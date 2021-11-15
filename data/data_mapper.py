# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
import boto3

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

from .new_augs import LargeScaleJitter


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []

    if is_train:
        tfm_gens.append(T.RandomFlip(prob=0.5, horizontal=True))

    if is_train and cfg.INPUT.LARGE_SCALE_JITTER.ENABLED:
        tfm_gens.append(
            LargeScaleJitter(
                min_scale=cfg.INPUT.LARGE_SCALE_JITTER.MIN_SCALE,
                max_scale=cfg.INPUT.LARGE_SCALE_JITTER.MAX_SCALE,
                short_edge_length=min_size,
                max_size=max_size,
                sample_style=sample_style,
                interp=Image.BILINEAR,
                pad_value=128.0,
            )
        )
    else:
        tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        tfm_gens.extend(
            [
                T.RandomBrightness(0.5, 1.5),
                T.RandomContrast(0.5, 1.5),
                T.RandomSaturation(0.5, 1.2),
                T.RandomLighting(1.0),
            ]
        )
        logger.info("TransformGens used in training: " + str(tfm_gens))
        print("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def get_s3(s3_info):
    if s3_info is None:
        return None
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_info["endpoint_url"],
        aws_access_key_id=s3_info["aws_access_key_id"],
        aws_secret_access_key=s3_info["aws_secret_access_key"],
        region_name=s3_info["region_name"],
        verify=s3_info["verify"],
    )
    return s3


def read_image_s3(path, s3, bucket, format=None):
    """
    Adapted from detectron2.data.detection_utils.read_image

    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        path (str): image file path from bucket
        s3 (`S3.Client`): boto3 object
        bucket (str): bucketname
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """

    with io.BytesIO() as f:
        s3.download_fileobj(bucket, path, f)
        image = Image.open(f)
        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = utils._apply_exif_orientation(image)
        np_img = utils.convert_PIL_to_numpy(image, format)
    return np_img


class AugDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, s3_info, is_train=True):
        """
        s3_info (dict, optional): dictionary of information needed for s3 communication (endpoint_url, bucket, aws_access_key_id, aws_secret_access_key, region_name, verify)
        """
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info(
                "CropGen used in training: " + str(self.crop_gen)
            )
            print("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train
        if s3_info:
            self.s3_info = s3_info
        else:
            self.s3_info = None
        self.s3 = None

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        if self.s3_info:
            if self.s3 is None:
                self.s3 = get_s3(self.s3_info)
            # download from s3 into memory and directly read it
            image = read_image_s3(
                dataset_dict["file_name"],
                self.s3,
                self.s3_info["bucket"],
                format=self.img_format,
            )
        else:
            # detectron2 default image reading (from local filesystem)
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict or len(dataset_dict["annotations"]) == 0:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                self.min_box_side_len,
                self.proposal_topk,
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict
