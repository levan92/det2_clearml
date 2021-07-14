#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from config import add_custom_configs

def add_custom_configs(cfg: CfgNode):
    _C = cfg

    _C.MODEL.INSTANCE_NORM = True

    _C.INPUT.LARGE_SCALE_JITTER = CfgNode()
    _C.INPUT.LARGE_SCALE_JITTER.ENABLED = True
    _C.INPUT.LARGE_SCALE_JITTER.MIN_SCALE = 0.2
    _C.INPUT.LARGE_SCALE_JITTER.MAX_SCALE = 2.0
    
    _C.SOLVER.PERIODIC_CHECKPOINTER = CfgNode({"ENABLED": True})   
    _C.SOLVER.PERIODIC_CHECKPOINTER.PERIOD = _C.SOLVER.CHECKPOINT_PERIOD

    _C.SOLVER.BEST_CHECKPOINTER = CfgNode({"ENABLED": False})
    _C.SOLVER.BEST_CHECKPOINTER.METRIC = "bbox/AP50"
    _C.SOLVER.BEST_CHECKPOINTER.MODE = "max"

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if cfg.SOLVER.PERIODIC_CHECKPOINTER.ENABLED and comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.PERIODIC_CHECKPOINTER.PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if cfg.SOLVER.BEST_CHECKPOINTER:
            ret.append(hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer, cfg.SOLVER.BEST_CHECKPOINTER.METRIC, mode=cfg.SOLVER.BEST_CHECKPOINTER.MODE))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args, cl_task=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if cl_task:
        # cl_task.connect(cfg)
        cl_dict = cl_task.connect_configuration(name='hyperparams', configuration=cfg)
        cfg = CfgNode(init_dict=cl_dict)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args, cl_task=None):
    cfg = setup(args, cl_task=cl_task)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()

if __name__ == "__main__":
    from pathlib import Path

    import wget
    from clearml import Task

    from utils.s3_helper import S3_handler
    from utils.det2_helper import register_datasets, parse_datasets_args, extend_opts

    parser = default_argument_parser()

    parser.add_argument("--skip-clearml", help='flag to entirely skip any clearml action.', action='store_true')
    ## CLEARML ARGS
    parser.add_argument("--clearml-proj", default="det2", help="ClearML Project Name")
    parser.add_argument("--clearml-task-name", default="Task", help="ClearML Task Name")
    parser.add_argument("--clearml-task-type", default="data_processing", help="ClearML Task Type, e.g. training, testing, inference, etc", choices=['training','testing','inference','data_processing','application','monitor','controller','optimizer','service','qc','custom'])
    parser.add_argument("--docker-img", default="harbor.dsta.ai/nvidia/pytorch:21.03-py3", help="Base docker image to pull")
    parser.add_argument("--queue", default="1gpu", help="ClearML Queue")

    parser.add_argument("--skip-s3", help='flag to entirely skip any s3 action.', action='store_true')
    ## DOWNLOAD MODELS ARGS
    parser.add_argument(
        "--download-models",
        help="List of models to download",
        nargs='+'
    )
    parser.add_argument("--s3-models-bucket", help="S3 Bucket for models")
    parser.add_argument("--s3-models-path", help="S3 Models Path")

    ## DOWNLOAD DATA ARGS
    parser.add_argument(
        "--download-data",
        help="List of dataset to download",
        nargs='+'
    )
    parser.add_argument("--s3-data-bucket", help="S3 Bucket for data")
    parser.add_argument("--s3-data-path", help="S3 Data Path")
    parser.add_argument(
        "--custom-dsnames",
        help='Names of custom datasets (must match to those in args.datasets_train and args.datasets_test). Only for custom datasets that does not conform to repo dataset assumptions.',
        nargs='*'
        )
    parser.add_argument(
        "--custom-cocojsons",
        help='Paths to coco json file. Only for custom datasets that does not conform to repo dataset assumptions.',
        nargs='*'
        )
    parser.add_argument(
        "--custom-imgroots",
        help='Paths to img roots. Only for custom datasets that does not conform to repo dataset assumptions.',
        nargs='*'
        )
    # Datasets to register, will override config.yaml
    parser.add_argument(
        "--datasets-train",
        help="DATASETS.TRAIN"
    )
    parser.add_argument(
        "--datasets-test",
        help="DATASETS.TEST"
    )

    ## UPLOAD OUTPUT ARGS
    parser.add_argument("--s3-output-bucket", help="S3 Bucket for output")
    parser.add_argument("--s3-output-path", help="S3 Path to output")

    args = parser.parse_args()
    print("Command Line Args:", args)

    if not args.skip_s3:
        '''
        S3 handling to download weights and datasets
        '''
        AWS_ENDPOINT_URL=os.environ.get("AWS_ENDPOINT_URL", "https://ecs.dsta.ai")
        AWS_ACCESS_KEY=os.environ.get("AWS_ACCESS_KEY")
        AWS_SECRET_ACCESS=os.environ.get("AWS_SECRET_ACCESS")
        CERT_PATH=os.environ.get("CERT_PATH", "/usr/share/ca-certificates/extra/ca.dsta.ai.crt")
        CERT_DL_URL='http://gitlab.dsta.ai/ai-platform/getting-started/raw/master/config/ca.dsta.ai.crt'
        if CERT_DL_URL:
            wget.download(CERT_DL_URL)
            CERT_PATH = Path(CERT_DL_URL).name

        s3_handler = S3_handler(AWS_ENDPOINT_URL, AWS_ACCESS_KEY, AWS_SECRET_ACCESS, CERT_PATH)

        local_weight_dir = 'weights'
        local_data_dir = 'datasets'
        local_output_dir = 'output'

        local_weights_paths= s3_handler.dl_files(args.download_models, args.s3_models_bucket, args.s3_models_path, local_weight_dir, unzip=True)

        local_data_dirs = s3_handler.dl_dirs(args.download_data, args.s3_data_bucket, args.s3_data_path, local_data_dir, unzip=True)

    '''
    Datasets Registration
    '''
    # Register the custom datasets that don't conform to dataset format assumptions first
    already_reged = []
    if args.custom_dsnames:
        assert len(args.custom_dsnames)==len(args.custom_cocojsons)
        assert len(args.custom_dsnames)==len(args.custom_imgroots)
        for dsname, cjson, imroot in zip(args.custom_dsnames, args.custom_cocojsons, args.custom_imgroots):
            register_datasets(dsname, json_path=cjson, dataset_image_root=imroot)
            already_reged.append(dsname)
        # register_coco_instances("coco_train", {}, '/media/dh/HDD/coco/annotations/instances_train2017.json', '/media/dh/HDD/coco/train2017')
        # register_coco_instances("coco_val", {}, '//media/dh/HDD/coco/smallval/instances_val2017.json', '/media/dh/HDD/coco/smallval/images')
    # Then register remaining of train and test sets, assuming remainders all conform to dataset format.
    datasets_to_reg = []
    datasets_train = parse_datasets_args(args.datasets_train, datasets_to_reg)
    datasets_test = parse_datasets_args(args.datasets_test, datasets_to_reg)
    remainder_sets = list(set(datasets_to_reg)-set(already_reged))
    for dataset_to_reg in remainder_sets:
        register_datasets(dataset_to_reg, local_data_dir=local_data_dir)

    extend_opts(args.opts, 'DATASETS.TRAIN', datasets_train)
    extend_opts(args.opts, 'DATASETS.TEST', datasets_test)

    if not args.skip_clearml:
        cl_task = Task.init(project_name=args.clearml_proj,task_name=args.clearml_task_name, task_type=args.clearml_task_type)
    else:
        cl_task = None

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, cl_task),
    )

    if not args.skip_s3:
        '''
        S3 handling to upload outputs
        '''
        s3_handler.ul_dir(local_output_dir, args.s3_output_bucket, args.s3_output_path, f'{args.clearml_task_name}')
