import sys

import numpy as np
from PIL import Image
from fvcore.transforms.transform import NoOpTransform, TransformList
from detectron2.data.transforms import (
    Augmentation,
    ResizeTransform,
    CropTransform,
    PadTransform,
)


class LargeScaleJitter(Augmentation):
    """
    Re-implemented the scaling portion from Large Scale Jitter augmentations in https://github.com/facebookresearch/detectron2/blob/master/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py

    This new LSJ merges logic from ResizeShortestEdge,  ResizeScale and FixedSizeCrop such that it calculates random target width and height based on given shortest and max edge length and thereafter does the random scaling, followed by cropping to target size.
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        short_edge_length,
        max_size=sys.maxsize,
        sample_style="range",
        interp=Image.BILINEAR,
        pad_value: float = 128.0,
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
            pad_value: the padding value.
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    def get_transform(self, image):
        input_size = image.shape[:2]
        h, w = input_size

        ## ResizeShortestEdge logic to get target width and height
        if self.is_range:
            size = np.random.randint(
                self.short_edge_length[0], self.short_edge_length[1] + 1
            )
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        target_width = int(neww + 0.5)
        target_height = int(newh + 0.5)
        output_size = (target_height, target_width)

        ## ResizeScale logic
        # Compute the image scale and scaled size.
        random_scale = np.random.uniform(self.min_scale, self.max_scale)
        random_scale_size = np.multiply(output_size, random_scale)
        scale = np.minimum(random_scale_size[0] / h, random_scale_size[1] / w)
        scaled_size = np.round(np.multiply(input_size, scale)).astype(int)

        resize_transform = ResizeTransform(
            h, w, scaled_size[0], scaled_size[1], self.interp
        )

        ## FixedSizeCrop logic
        # Add random crop if the image is scaled up.
        max_offset = np.subtract(scaled_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        crop_transform = CropTransform(
            offset[1],
            offset[0],
            output_size[1],
            output_size[0],
            scaled_size[1],
            scaled_size[0],
        )

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, scaled_size)
        pad_size = np.maximum(pad_size, 0)
        original_size = np.minimum(scaled_size, output_size)
        pad_transform = PadTransform(
            0,
            0,
            pad_size[1],
            pad_size[0],
            original_size[1],
            original_size[0],
            self.pad_value,
        )

        return TransformList([resize_transform, crop_transform, pad_transform])
