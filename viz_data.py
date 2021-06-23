import cv2
from detectron2.config import get_cfg
from detectron2.engine import default_setup

from config import add_custom_configs
from trainer import Trainer
from utils import det2_viz, register_datasets, extend_opts, parse_datasets_args

def setup(args, freeze=True):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_configs(cfg)
    cfg.merge_from_file(args.config_file)
    if hasattr(args, 'opts') and args.opts is not None:
        cfg.merge_from_list(args.opts)
    if freeze:
        cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    train_loader = Trainer.build_train_loader(cfg)
    for bi, batch in enumerate(train_loader):
        for per_image in batch:
            end = det2_viz(per_image, cfg)
            if end:
                break
        if end:
            break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--datasets-dir",
        help='Path to parent folder of datasets',
        required=True
    )
    parser.add_argument(
    "--datasets-train",
    help="DATASETS.TRAIN"
    )
    parser.add_argument(
    "--datasets-test",
    help="DATASETS.TEST"
    )
    parser.add_argument(
    "--solver-ims-per-batch",
    help="SOLVER.IMS_PER_BATCH"
    )
    parser.add_argument(
    "--min-size-train",
    help="INPUT.MIN_SIZE_TRAIN, default: (640, 672, 704, 736, 768, 800)"
    )
    parser.add_argument(
    "--max-size-train",
    help="INPUT.MAX_SIZE_TRAIN, default: 1333"
    )
    parser.add_argument(
    "--min-size-test",
    help="INPUT.MIN_SIZE_TEST, default: 800"
    )
    parser.add_argument(
    "--max-size-test",
    help="INPUT.MAX_SIZE_TEST, default: 1333"
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

    datasets_to_reg = []
    datasets_train = parse_datasets_args(args.datasets_train, datasets_to_reg)
    datasets_test = parse_datasets_args(args.datasets_test, datasets_to_reg)

    for dataset_to_reg in list(set(datasets_to_reg)):
        register_datasets(dataset_to_reg, args.datasets_dir)

    extend_opts(args.opts, 'DATASETS.TRAIN', datasets_train)
    extend_opts(args.opts, 'DATASETS.TEST', datasets_test)
    extend_opts(args.opts, 'SOLVER.IMS_PER_BATCH', args.solver_ims_per_batch)
    extend_opts(args.opts, 'INPUT.MIN_SIZE_TRAIN', args.min_size_train)
    extend_opts(args.opts, 'INPUT.MAX_SIZE_TRAIN', args.max_size_train)
    extend_opts(args.opts, 'INPUT.MIN_SIZE_TEST', args.min_size_test)
    extend_opts(args.opts, 'INPUT.MAX_SIZE_TEST', args.max_size_test)

    main(args)
