# -*- coding = utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_custom_configs(cfg: CN):
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.MODEL.INSTANCE_NORM = True

    _C.INPUT.LARGE_SCALE_JITTER = CN()
    _C.INPUT.LARGE_SCALE_JITTER.ENABLED = True
    _C.INPUT.LARGE_SCALE_JITTER.MIN_SCALE = 0.2
    _C.INPUT.LARGE_SCALE_JITTER.MAX_SCALE = 2.0
