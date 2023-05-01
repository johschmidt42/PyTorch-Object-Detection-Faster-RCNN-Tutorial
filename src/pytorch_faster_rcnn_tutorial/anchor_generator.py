import logging
from typing import Tuple

import torch
from torch import nn
from torch.jit.annotations import Dict, List, Optional
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform

logger: logging.Logger = logging.getLogger(__name__)


class AnchorGenerator(nn.Module):
    # Slightly adapted AnchorGenerator from torchvision.
    # It returns anchors_over_all_feature_maps instead of anchors (concatenated for every feature layer)

    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(
        self,
        scales,
        aspect_ratios,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cpu",
    ) -> torch.Tensor:
        scales = torch.as_tensor(data=scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(data=aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(
        self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        assert len(grid_sizes) == len(strides) == len(cell_anchors)

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = (
                torch.arange(0, grid_width, dtype=torch.float32, device=device)
                * stride_width
            )
            shifts_y = (
                torch.arange(0, grid_height, dtype=torch.float32, device=device)
                * stride_height
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def cached_grid_anchors(
        self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(
        self, image_list: ImageList, feature_maps: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        self._cache.clear()
        return anchors_over_all_feature_maps


def get_anchor_boxes(
    image: torch.Tensor,
    rcnn_transform: GeneralizedRCNNTransform,
    feature_map_size: tuple,
    anchor_size: Tuple[Tuple[int]] = ((128, 256, 512),),
    aspect_ratios: Tuple[Tuple[float]] = ((1.0,),),
):
    """
    Returns the anchors for a given image and feature map.
    image should be a torch.Tensor with shape [C, H, W].
    feature_map_size should be a tuple with shape (C, H, W]).
    Only one feature map supported at the moment.

    Example:

    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    transform = GeneralizedRCNNTransform(min_size=1024,
                                         max_size=1024,
                                         image_mean=[0.485, 0.456, 0.406],
                                         image_std=[0.229, 0.224, 0.225])

    image = dataset[0]['x'] # ObjectDetectionDataSet

    anchors = get_anchor_boxes(image,
                               transform,
                               feature_map_size=(512, 16, 16),
                               anchor_size=((128, 256, 512),),
                               aspect_ratios=((1.0, 2.0),)
                               )
    """

    image_transformed = rcnn_transform([image])

    features = [torch.rand(size=feature_map_size)]

    anchor_gen = AnchorGenerator(anchor_size, aspect_ratios)
    anchors = anchor_gen(image_list=image_transformed[0], feature_maps=features)

    return anchors[0]
