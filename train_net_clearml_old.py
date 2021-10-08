# this needs to be here for it to read your args
from clearml import Task, Logger

#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
from copy import deepcopy

"""
ARGUMENT PARSER
"""

parser = argparse.ArgumentParser()
parser.add_argument("--clearml-proj", default="DETECTRON2", help="ClearML Project Name")
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
    "--docker-img",
    default="harbor.dsta.ai/nvidia/pytorch:21.03-py3",
    help="Base docker image to pull",
)
parser.add_argument("--queue", default="1gpu", help="ClearML Queue")
parser.add_argument("--clearml-output-uri", help="ClearML Output URI")
parser.add_argument("--s3-models-bucket", help="S3 Bucket for models")
parser.add_argument("--s3-models-path", help="S3 Models Path")
parser.add_argument("--s3-data-bucket", help="S3 Bucket for data")
parser.add_argument("--s3-data-path", help="S3 Data Path")
parser.add_argument("--s3-output-bucket", help="S3 Bucket for output")
parser.add_argument("--s3-output-path", help="S3 Path to output")
parser.add_argument(
    "--custom-dsnames",
    help="Names of custom datasets (must match to those in args.datasets_train and args.datasets_test). Only for custom datasets that does not conform to repo dataset assumptions.",
    nargs="*",
)
parser.add_argument(
    "--custom-cocojsons",
    help="Paths to coco json file. Only for custom datasets that does not conform to repo dataset assumptions.",
    nargs="*",
)
parser.add_argument(
    "--custom-imgroots",
    help="Paths to img roots. Only for custom datasets that does not conform to repo dataset assumptions.",
    nargs="*",
)
parser.add_argument(
    "--config-file", default="", metavar="FILE", help="path to config file"
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Whether to attempt to resume from the checkpoint directory. "
    "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
)
parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
parser.add_argument(
    "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
)
parser.add_argument(
    "--num-machines", type=int, default=1, help="total number of machines"
)
parser.add_argument(
    "--machine-rank",
    type=int,
    default=0,
    help="the rank of this machine (unique per machine)",
)

# PyTorch still may leave orphan processes in multi-gpu training.
# Therefore we use a deterministic way to obtain port,
# so that users are aware of orphan processes by seeing the port occupied.
port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:{}".format(port),
    help="initialization URL for pytorch distributed backend. See "
    "https://pytorch.org/docs/stable/distributed.html for details.",
)
parser.add_argument("--download-data", help="List of dataset to download", nargs="+")

parser.add_argument(
    "--noclearml", help="flag to NOT send clearml job", action="store_true"
)
parser.add_argument("--num-classes", help="MODEL.ROI_HEADS.NUM_CLASSES")
parser.add_argument(
    "--model-weights",
    help="MODEL.WEIGHTS | if multiple are given, will generate test for each (eval mode only). This assumes all model weights are of same architecture as given config yaml file.",
    nargs="*",
)
parser.add_argument(
    "--output-dir",
    help="OUTPUT_DIR | Multiple must be given if model weights are multiple as well. Should have same length. Respective inference res from each model weight run will output to respective output dir. Defaults to outputs.",
    nargs="*",
)
parser.add_argument("--datasets-train", help="DATASETS.TRAIN")
parser.add_argument("--datasets-test", help="DATASETS.TEST")
parser.add_argument("--test-eval-period", help="TEST.EVAL_PERIOD")
parser.add_argument("--solver-ims-per-batch", help="SOLVER.IMS_PER_BATCH")
parser.add_argument("--solver-base-lr", help="SOLVER.BASE_LR")
parser.add_argument("--solver-warmup-iters", help="SOLVER.WARMUP_ITERS")
parser.add_argument("--solver-steps", help="SOLVER.STEPS")
parser.add_argument("--solver-max-iter", help="SOLVER.MAX_ITER")
parser.add_argument("--solver-checkpoint-period", help="SOLVER.CHECKPOINT_PERIOD")
parser.add_argument(
    "--anchor-sizes",
    help="MODEL.ANCHOR_GENERATOR.SIZES, default: [[32], [64], [128], [256], [512]]",
)
parser.add_argument(
    "--anchor-aspect-ratios",
    help="MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS, default: [[0.5, 1.0, 2.0]]",
)
parser.add_argument(
    "--min-size-train",
    help="INPUT.MIN_SIZE_TRAIN, default: (640, 672, 704, 736, 768, 800)",
)
parser.add_argument("--max-size-train", help="INPUT.MAX_SIZE_TRAIN, default: 1333")
parser.add_argument("--min-size-test", help="INPUT.MIN_SIZE_TEST, default: 800")
parser.add_argument("--max-size-test", help="INPUT.MAX_SIZE_TEST, default: 1333")
parser.add_argument(
    "--disable-crop", help="INPUT.CROP.ENABLED, default: enabled", action="store_false"
)
parser.add_argument("--crop-size", help="INPUT.CROP.SIZE, default: [0.5,1.0]")
parser.add_argument(
    "--lsj",
    help="INPUT.LARGE_SCALE_JITTER.ENABLED, default: disabled",
    action="store_true",
)
parser.add_argument(
    "opts",
    help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
    "See config references at "
    "https://detectron2.readthedocs.io/modules/config.html#config-references",
    default=None,
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

print("Command Line Args:", args)

AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "https://ecs.dsta.ai")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS = os.environ.get("AWS_SECRET_ACCESS")
CERT_PATH = os.environ.get(
    "CERT_PATH", "/usr/share/ca-certificates/extra/ca.dsta.ai.crt"
)
CERT_DL_URL = (
    "http://gitlab.dsta.ai/ai-platform/getting-started/raw/master/config/ca.dsta.ai.crt"
)

base_output_dir = "./output"
if args.model_weights and len(args.model_weights) > 1:
    assert args.eval_only, "Only give multiple model weights if on eval mode"
    assert len(args.output_dir) == len(
        args.model_weights
    ), "Multiple output dir must be of same length to multiple model weights."
    output_dirs = [Path(base_output_dir) / opd for opd in args.output_dir]
else:
    output_dirs = [base_output_dir]

"""
Clearml
"""
if not args.noclearml:
    task = Task.init(
        project_name=args.clearml_proj,
        task_name=args.clearml_task_name,
        task_type=args.clearml_task_type,
    )
    task.set_base_docker(
        f"{args.docker_img} --env GIT_SSL_NO_VERIFY=true --env AWS_ACCESS_KEY={AWS_ACCESS_KEY} --env AWS_SECRET_ACCESS={AWS_SECRET_ACCESS}"
    )
    task.execute_remotely(queue_name=args.queue, exit_process=True)

"""
S3 downloading
"""
import wget
import boto3
from botocore.client import Config

from utils import download_dir_from_s3, upload_dir_to_s3

if CERT_DL_URL:
    wget.download(CERT_DL_URL)
    CERT_PATH = Path(CERT_DL_URL).name

s3 = boto3.resource(
    "s3",
    endpoint_url=AWS_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
    verify=CERT_PATH,
)

# if args.model_weights:
local_weights_paths = []
for model_weight in args.model_weights:
    assert args.s3_models_bucket
    assert args.s3_models_path
    s3_models_parent = Path(args.s3_models_path)
    s3_weights_path = s3_models_parent / Path(model_weight)
    local_weights_path = "weights" / Path(model_weight)
    local_weights_path.parent.mkdir(parents=True, exist_ok=True)
    s3.Bucket(args.s3_models_bucket).download_file(
        str(s3_weights_path), str(local_weights_path)
    )
    assert local_weights_path.is_file()
    local_weights_paths.append(local_weights_path)
    print(f"Weights: {model_weight} downloaded from S3!")

local_data_dir = Path("datasets")
local_data_dir.mkdir(parents=True, exist_ok=True)
if args.download_data:
    assert args.s3_data_bucket
    assert args.s3_data_path
    s3_dataset_parent = Path(args.s3_data_path)
    for dataset in args.download_data:
        local_dataset_path = local_data_dir / dataset
        s3_dataset_path = s3_dataset_parent / dataset
        download_dir_from_s3(
            s3, args.s3_data_bucket, s3_dataset_path, local_dataset_path, untar=True
        )

"""
TRAINING
"""
from detectron2.engine import launch

from trainer import main
from utils import register_datasets, extend_opts, parse_datasets_args

# Register the custom datasets that don't conform to dataset format assumptions first
already_reged = []
if args.custom_dsnames:
    assert len(args.custom_dsnames) == len(args.custom_cocojsons)
    assert len(args.custom_dsnames) == len(args.custom_imgroots)
    for dsname, cjson, imroot in zip(
        args.custom_dsnames, args.custom_cocojsons, args.custom_imgroots
    ):
        register_datasets(dsname, json_path=cjson, dataset_image_root=imroot)
        already_reged.append(dsname)

# Then register remaining of train and test sets, assuming remainders all conform to dataset format.
datasets_to_reg = []
datasets_train = parse_datasets_args(args.datasets_train, datasets_to_reg)
datasets_test = parse_datasets_args(args.datasets_test, datasets_to_reg)
for dataset_to_reg in list(set(datasets_to_reg) - set(already_reged)):
    register_datasets(dataset_to_reg, local_data_dir=local_data_dir)

extend_opts(args.opts, "MODEL.ROI_HEADS.NUM_CLASSES", args.num_classes)
extend_opts(args.opts, "DATASETS.TRAIN", datasets_train)
extend_opts(args.opts, "DATASETS.TEST", datasets_test)
extend_opts(args.opts, "TEST.EVAL_PERIOD", args.test_eval_period)
extend_opts(args.opts, "SOLVER.IMS_PER_BATCH", args.solver_ims_per_batch)
extend_opts(args.opts, "SOLVER.BASE_LR", args.solver_base_lr)
extend_opts(args.opts, "SOLVER.WARMUP_ITERS", args.solver_warmup_iters)
extend_opts(args.opts, "SOLVER.STEPS", args.solver_steps)
extend_opts(args.opts, "SOLVER.MAX_ITER", args.solver_max_iter)
extend_opts(args.opts, "SOLVER.CHECKPOINT_PERIOD", args.solver_checkpoint_period)
extend_opts(args.opts, "MODEL.ANCHOR_GENERATOR.SIZES", args.anchor_sizes)
extend_opts(
    args.opts, "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS", args.anchor_aspect_ratios
)
extend_opts(args.opts, "INPUT.MIN_SIZE_TRAIN", args.min_size_train)
extend_opts(args.opts, "INPUT.MAX_SIZE_TRAIN", args.max_size_train)
extend_opts(args.opts, "INPUT.MIN_SIZE_TEST", args.min_size_test)
extend_opts(args.opts, "INPUT.MAX_SIZE_TEST", args.max_size_test)
extend_opts(args.opts, "INPUT.CROP.ENABLED", args.disable_crop)
extend_opts(args.opts, "INPUT.CROP.SIZE", args.crop_size)
extend_opts(args.opts, "INPUT.LARGE_SCALE_JITTER.ENABLED", args.lsj)

for local_wts, outputdir in zip(local_weights_paths, output_dirs):
    this_args = deepcopy(args)
    extend_opts(this_args.opts, "MODEL.WEIGHTS", str(local_wts))
    extend_opts(this_args.opts, "OUTPUT_DIR", str(outputdir))

    launch(
        main,
        this_args.num_gpus,
        num_machines=this_args.num_machines,
        machine_rank=this_args.machine_rank,
        dist_url=this_args.dist_url,
        args=(this_args,),
    )

"""
Upload Outputs
"""
s3_output_path = Path(args.s3_output_path) / f"{args.clearml_task_name}"
upload_dir_to_s3(s3, args.s3_output_bucket, base_output_dir, s3_output_path)
