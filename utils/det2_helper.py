import ast
from pathlib import Path

from detectron2.data.datasets import register_coco_instances


def parse_datasets_args(ds_args, datasets_to_reg):
    if ds_args:
        if ds_args.startswith("("):
            ds = ast.literal_eval(ds_args)
        else:
            ds = (ds_args,)
        datasets_to_reg.extend(ds)
    else:
        ds = None
    return ds


def get_root_and_json(dataset_name, local_data_dir):
    local_data_dir = Path(local_data_dir)
    set_name, phase = dataset_name.split("_", 1)
    dataset_dir = local_data_dir / set_name
    assert dataset_dir.is_dir(), dataset_dir
    dataset_image_root = dataset_dir / "images"
    json_path = dataset_dir / f"{phase}.json"
    return dataset_image_root, json_path, set_name, phase


def register_datasets(
    dataset_name, local_data_dir=None, json_path=None, dataset_image_root=None
):
    assert local_data_dir or json_path
    if local_data_dir is None:
        print("Registering", dataset_name)
        json_path = Path(json_path)
        dataset_image_root = Path(dataset_image_root)
    else:
        dataset_image_root, json_path, set_name, phase = get_root_and_json(
            dataset_name, local_data_dir
        )
        print("Registering", set_name, phase)

    assert dataset_image_root.is_dir(), dataset_image_root
    assert json_path.is_file(), json_path
    register_coco_instances(dataset_name, {}, json_path, dataset_image_root)


def extend_opts(opts, cfg_param, value):
    if value is not None:
        opts.extend([cfg_param, value])


import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils as utils

SCALE = 1.0


def det2_viz(det2_img, cfg):
    # Pytorch tensor is in (C, H, W) format
    img = det2_img["image"].permute(1, 2, 0).cpu().detach().numpy()
    img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
    visualizer = Visualizer(img, scale=SCALE)
    target_fields = det2_img["instances"].get_fields()
    vis = visualizer.overlay_instances(
        boxes=target_fields.get("gt_boxes", None),
        masks=target_fields.get("gt_masks", None),
        keypoints=target_fields.get("gt_keypoints", None),
    )
    out_img = vis.get_image()
    cv2.imshow("Press q to quit, Any to continue.", out_img[:, :, ::-1])
    if cv2.waitKey() == ord("q"):
        return True
    else:
        return False
