#!/usr/bin/env python

if __name__ == "__main__":
    import os
    from pathlib import Path

    from clearml import Task

    from det2_default_argparser import default_argument_parser

    parser = default_argument_parser()
    parser.add_argument(
        "--skip-clearml",
        help="flag to entirely skip any clearml action.",
        action="store_true",
    )
    parser.add_argument(
        "--monitor-ps", help="flag to monitoring processes.", action="store_true"
    )
    parser.add_argument(
        "--clearml-run-locally",
        help="flag to run job locally but keep clearml expt tracking.",
        action="store_true",
    )
    ## CLEARML ARGS
    parser.add_argument("--clearml-proj", default="det2", help="ClearML Project Name")
    parser.add_argument("--clearml-task-name", default="Task", help="ClearML Task Name")
    parser.add_argument(
        "--clearml-task-type",
        default="data_processing",
        help="ClearML Task Type, e.g. training, testing, inference, etc",
        choices=[
            "training",
            "testing",
            "inference",
            "data_processing",
            "application",
            "monitor",
            "controller",
            "optimizer",
            "service",
            "qc",
            "custom",
        ],
    )
    parser.add_argument(
        "--clearml-output-uri",
        # default="s3://ecs.dsta.ai:80/clearml-models/default",
        help="ClearML output uri",
    )
    parser.add_argument(
        "--docker-img",
        default="harbor.dsta.ai/nvidia/pytorch:21.03-py3",
        help="Base docker image to pull",
    )
    parser.add_argument("--queue", default="1gpu", help="ClearML Queue")
    ### S3
    parser.add_argument(
        "--skip-s3", help="flag to entirely skip any s3 action.", action="store_true"
    )
    ## DOWNLOAD MODELS ARGS
    parser.add_argument(
        "--download-models", help="List of models to download", nargs="+"
    )
    parser.add_argument("--s3-models-bucket", help="S3 Bucket for models")
    parser.add_argument("--s3-models-path", help="S3 Models Path")
    ## Model weights to load for training
    parser.add_argument(
        "--model-weights", help="MODEL.WEIGHTS | Path to pretrained model weights"
    )
    parser.add_argument(
        "--from-scratch",
        help="MODEL.WEIGHTS set to empty string | Train model from scratch",
        action="store_true",
    )
    ## DOWNLOAD DATA ARGS
    parser.add_argument(
        "--download-data", help="List of dataset to download", nargs="+"
    )
    parser.add_argument(
        "--local-data-dir", help="Destination dataset files downloaded to", default='datasets'
    )
    parser.add_argument("--s3-data-bucket", help="S3 Bucket for data")
    parser.add_argument("--s3-data-path", help="S3 Data Path")
    parser.add_argument(
        "--s3-direct-read",
        help="DATASETS.S3.ENABLED | enable direct reading of images from S3 bucket without initial download.",
        action="store_true",
    )
    parser.add_argument(
        "--coco-dsnames",
        help="Names of custom datasets (must match to those in args.datasets_train and args.datasets_test).",
        nargs="*",
    )
    parser.add_argument(
        "--coco-jsons",
        help="Paths to coco json file.",
        nargs="*",
    )
    parser.add_argument(
        "--coco-imgroots",
        help="Paths to img roots.",
        nargs="*",
    )
    # Datasets to register, will override config.yaml
    parser.add_argument("--datasets-train", help="DATASETS.TRAIN")
    parser.add_argument("--datasets-test", help="DATASETS.TEST")
    # ## UPLOAD OUTPUT ARGS
    # parser.add_argument("--s3-output-bucket", help="S3 Bucket for output")
    # parser.add_argument("--s3-output-path", help="S3 Path to output")
    ## Hyperparams
    parser.add_argument("--num-classes", help="MODEL.ROI_HEADS.NUM_CLASSES")
    parser.add_argument("--test-eval-period", help="TEST.EVAL_PERIOD")
    parser.add_argument("--solver-ims-per-batch", help="SOLVER.IMS_PER_BATCH")
    parser.add_argument("--solver-base-lr", help="SOLVER.BASE_LR")
    parser.add_argument("--solver-gamma", help="SOLVER.GAMMA")
    parser.add_argument("--solver-warmup-iters", help="SOLVER.WARMUP_ITERS")
    parser.add_argument("--solver-steps", help="SOLVER.STEPS")
    parser.add_argument("--solver-max-iter", help="SOLVER.MAX_ITER")
    parser.add_argument("--dataloader-num-workers", help="DATALOADER.NUM_WORKERS")
    parser.add_argument("--model-anchor-sizes", help="MODEL.ANCHOR_GENERATOR.SIZES")
    parser.add_argument(
        "--model-anchor-ar", help="MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS"
    )

    args = parser.parse_args()
    print("Command Line Args:", args)

    environs_names = ["AWS_ENDPOINT_URL", "AWS_ACCESS_KEY", "AWS_SECRET_ACCESS", "CERT_PATH", "CERT_DL_URL"]
    environs = {var: os.environ.get(var) for var in environs_names}

    """
    clearml task init
    """
    if not args.skip_clearml:
        cl_task = Task.init(
            project_name=args.clearml_proj,
            task_name=args.clearml_task_name,
            task_type=args.clearml_task_type,
            output_uri=args.clearml_output_uri,
        )
        env_strs = ' '.join([ f"--env {k}={v}" for k, v in environs.items() ])
        cl_task.set_base_docker(
            f"{args.docker_img} --env GIT_SSL_NO_VERIFY=true {env_strs}"
        )
        if not args.clearml_run_locally:
            cl_task.execute_remotely(queue_name=args.queue, exit_process=True)
        cl_task_id = cl_task.task_id
    else:
        cl_task = None
        cl_task_id = None

    """
    S3 handling to download weights and datasets
    """
    local_weight_dir = "weights"
    local_data_dir = args.local_data_dir
    local_output_dir = "output"

    if not args.skip_s3:
        from utils.s3_helper import S3_handler

        environs['CERT_PATH'] = environs['CERT_PATH'] if environs['CERT_PATH'] else None
        if environs['CERT_DL_URL'] and environs['CERT_PATH'] and not Path(environs['CERT_PATH']).is_file():
            import utils.wget as wget
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            print(f'Downloading from {environs["CERT_DL_URL"]}')
            wget.download(environs['CERT_DL_URL'])
            environs['CERT_PATH'] = Path(environs['CERT_DL_URL']).name

        s3_handler = S3_handler(
            environs['AWS_ENDPOINT_URL'], environs['AWS_ACCESS_KEY'], environs['AWS_SECRET_ACCESS'], environs['CERT_PATH']
        )

        if args.download_models:
            local_weights_paths = s3_handler.dl_files(
                args.download_models,
                args.s3_models_bucket,
                args.s3_models_path,
                local_weight_dir,
                unzip=True,
            )

        if args.download_data:
            if args.s3_direct_read:
                local_data_dirs = s3_handler.dl_files(
                    args.download_data,
                    args.s3_data_bucket,
                    args.s3_data_path,
                    local_data_dir,
                    unzip=True,
                )
            else:
                local_data_dirs = s3_handler.dl_dirs(
                    args.download_data,
                    args.s3_data_bucket,
                    args.s3_data_path,
                    local_data_dir,
                    unzip=True,
                )

    """
    Datasets Registration
    """
    from utils.det2_helper import register_datasets, parse_datasets_args, extend_opts

    # Register the custom datasets that don't conform to dataset format assumptions first
    # already_reged = []
    # if args.custom_dsnames:
    #     assert len(args.custom_dsnames)==len(args.custom_cocojsons)
    #     assert len(args.custom_dsnames)==len(args.custom_imgroots)
    #     for dsname, cjson, imroot in zip(args.custom_dsnames, args.custom_cocojsons, args.custom_imgroots):
    #         register_datasets(dsname, json_path=cjson, dataset_image_root=imroot)
    #         already_reged.append(dsname)
    # register_coco_instances("coco_train", {}, '/media/dh/HDD/coco/annotations/instances_train2017.json', '/media/dh/HDD/coco/train2017')
    # register_coco_instances("coco_val", {}, '//media/dh/HDD/coco/smallval/instances_val2017.json', '/media/dh/HDD/coco/smallval/images')
    # Then register remaining of train and test sets, assuming remainders all conform to dataset format.
    datasets_to_reg = []
    datasets_train = parse_datasets_args(args.datasets_train, datasets_to_reg)
    datasets_test = parse_datasets_args(args.datasets_test, datasets_to_reg)
    # remainder_sets = list(set(datasets_to_reg)-set(already_reged))
    # for dataset_to_reg in remainder_sets:
    #     register_datasets(dataset_to_reg, local_data_dir=local_data_dir)

    extend_opts(args.opts, "DATASETS.TRAIN", datasets_train)
    extend_opts(args.opts, "DATASETS.TEST", datasets_test)
    if args.from_scratch:
        extend_opts(args.opts, "MODEL.WEIGHTS", "")
    else:
        extend_opts(args.opts, "MODEL.WEIGHTS", args.model_weights)

    extend_opts(args.opts, "MODEL.ROI_HEADS.NUM_CLASSES", args.num_classes)
    extend_opts(args.opts, "TEST.EVAL_PERIOD", args.test_eval_period)
    extend_opts(args.opts, "SOLVER.IMS_PER_BATCH", args.solver_ims_per_batch)
    extend_opts(args.opts, "SOLVER.BASE_LR", args.solver_base_lr)
    extend_opts(args.opts, "SOLVER.GAMMA", args.solver_gamma)
    extend_opts(args.opts, "SOLVER.WARMUP_ITERS", args.solver_warmup_iters)
    extend_opts(args.opts, "SOLVER.STEPS", args.solver_steps)
    extend_opts(args.opts, "SOLVER.MAX_ITER", args.solver_max_iter)
    extend_opts(args.opts, "DATALOADER.NUM_WORKERS", args.dataloader_num_workers)
    extend_opts(args.opts, "MODEL.ANCHOR_GENERATOR.SIZES", args.model_anchor_sizes)
    extend_opts(args.opts, "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS", args.model_anchor_ar)

    if args.s3_direct_read:
        extend_opts(args.opts, "DATASETS.S3.AWS_ENDPOINT_URL", environs['AWS_ENDPOINT_URL'])
        extend_opts(args.opts, "DATASETS.S3.AWS_ACCESS_KEY", environs['AWS_ACCESS_KEY'])
        extend_opts(args.opts, "DATASETS.S3.AWS_SECRET_ACCESS", environs['AWS_SECRET_ACCESS'])
        extend_opts(args.opts, "DATASETS.S3.REGION_NAME", "us-east-1")
        extend_opts(args.opts, "DATASETS.S3.BUCKET", args.s3_data_bucket)
        extend_opts(args.opts, "DATASETS.S3.CERT_PATH", environs['CERT_PATH'])

    """
    Launching detectron2 run
    """
    from detectron2.engine import launch

    from trainer import main

    if args.monitor_ps:
        from utils.psutil_helper import start_monitor

        start_monitor(freq=1)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, cl_task_id),
    )

    """
    S3 handling to upload outputs
    """
    if args.eval_only:
        from tester import coco_eval

        pred_path = Path(local_output_dir)
        pred_path = pred_path / "inference/coco_instances_results.json"
        if cl_task:
            cl_task.upload_artifact(
                name="predictions",
                artifact_object=str(pred_path),
            )
        if args.custom_cocojsons:
            for custom_json in args.custom_cocojsons:
                custom_json_path = Path(custom_json)
                if "val" in custom_json_path.stem:
                    data_dir = custom_json_path.parent
                    break
        else:
            data_dir = Path(local_data_dir)
        datasets_test = (
            datasets_test[0] if isinstance(datasets_test, tuple) else datasets_test
        )
        evals = coco_eval(pred_path, data_dir, val_str="val", subfolder=datasets_test)
        if cl_task:
            cl_task.upload_artifact(
                name="evaluations",
                artifact_object=evals,
            )
            cl_logger = cl_task.get_logger()
            for val_set, eval_values in evals.items():
                for metric, value in eval_values.items():
                    cl_logger.report_scalar(
                        title=val_set.replace("_coco-catified", ""),
                        series=metric,
                        value=value,
                        iteration=0,
                    )
