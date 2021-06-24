import os
import ast
import tarfile
import zipfile
from pathlib import Path

def download_dir_from_s3(s3_resource, bucket_name, remote_dir_name, local_dir, untar=True):
    buck = s3_resource.Bucket(bucket_name)
    for obj in buck.objects.filter(Prefix=str(remote_dir_name)+'/'):
        remote_rel_path = Path(obj.key).relative_to(remote_dir_name)
        local_fp = local_dir / remote_rel_path
        local_fp.parent.mkdir(parents=True, exist_ok=True)
        if not local_fp.is_file():
            print(f'Downloading {obj.key} from S3..')
            buck.download_file(obj.key, str(local_fp))
            if local_fp.suffix in ['.tar','.gz','.tgz']:
                print('Untarring..')
                tar = tarfile.open(local_fp)
                tar.extractall(local_dir)
                tar.close()
            elif local_fp.suffix in ['.zip']:
                print('Unzipping..')
                with zipfile.ZipFile(local_fp, 'r') as zip_ref:
                    zip_ref.extractall(local_dir)

def upload_dir_to_s3(s3_resource, bucket_name, local_dir, remote_dir):
    buck = s3_resource.Bucket(bucket_name)

    for root,dirs,files in os.walk(str(local_dir)):
        for file in files: 
            local_fp = Path(root)/file
            rel_path = Path(root).relative_to(local_dir)
            remote_fp = Path(remote_dir)/ rel_path / file
            print(f'Uploading {local_fp} to S3 {remote_dir}')
            buck.upload_file(str(local_fp), str(remote_fp))

from detectron2.data.datasets import register_coco_instances

def register_datasets(dataset_name, local_data_dir = None, json_path = None, dataset_image_root = None):
    assert local_data_dir or json_path
    if local_data_dir is None:
        print('Registering', dataset_name)
        json_path = Path(json_path)
        dataset_image_root = Path(dataset_image_root)
    else:
        local_data_dir = Path(local_data_dir)
        set_name, phase = dataset_name.rsplit('_',1)
        print('Registering', set_name, phase)
        dataset_dir = local_data_dir / set_name
        assert dataset_dir.is_dir(),dataset_dir
        dataset_image_root = dataset_dir / 'images'
        json_path = dataset_dir / f'{phase}.json'
    assert dataset_image_root.is_dir(),dataset_image_root
    assert json_path.is_file(),json_path
    register_coco_instances(dataset_name, {}, json_path, dataset_image_root)

def extend_opts(opts, cfg_param, value):
    if value is not None:
        opts.extend([cfg_param,value])

def parse_datasets_args(ds_args, datasets_to_reg):
    if ds_args:
        if ds_args.startswith('('):
            ds = ast.literal_eval(ds_args)
        else:
            ds = (ds_args,)
        datasets_to_reg.extend(ds)
    else:
        ds = None
    return ds

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils as utils

SCALE=1.0
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
    if cv2.waitKey() == ord('q'):
        return True
    else:
        return False