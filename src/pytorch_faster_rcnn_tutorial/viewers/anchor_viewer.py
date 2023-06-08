from typing import Any, Dict, Tuple

import numpy as np
import torch
from napari.layers import Shapes
from torch.utils.data import Dataset
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_convert

from pytorch_faster_rcnn_tutorial.anchor_generator import get_anchor_boxes
from pytorch_faster_rcnn_tutorial.transformations import re_normalize
from pytorch_faster_rcnn_tutorial.viewers.dataset_viewer import (
    DatasetViewer,
    make_bbox_napari,
)


def get_center_bounding_box(boxes: torch.tensor):
    """
    Returns the center points of given bounding boxes.
    """
    return box_convert(boxes=boxes, in_fmt="xyxy", out_fmt="cxcywh")[:, :2]


class AnchorViewer(DatasetViewer):
    """
    Anchor Viewer for object detection datasets (FasterRCNN).
    """

    def __init__(
        self,
        dataset: Dataset,
        rcnn_transform: GeneralizedRCNNTransform,
        feature_map_size: tuple,
        anchor_size: Tuple[tuple] = ((128, 256, 512),),
        aspect_ratios: Tuple[tuple] = ((1.0,),),
    ):
        super().__init__(torch_dataset=dataset, target_type=Shapes)

        self.rcnn_transform: GeneralizedRCNNTransform = rcnn_transform
        self.feature_map_size: tuple = feature_map_size
        self.anchor_size: Tuple[tuple] = anchor_size
        self.aspect_ratios: Tuple[tuple] = aspect_ratios

        self.image = None
        self.anchor_boxes = None
        self.first_anchor = None
        self.center_points = None
        self.anchor_points = None

    def get_anchors(self):
        """
        Returns the anchor boxes for the current image.
        """
        return get_anchor_boxes(
            image=self.image,
            rcnn_transform=self.rcnn_transform,
            feature_map_size=self.feature_map_size,
            anchor_size=self.anchor_size,
            aspect_ratios=self.aspect_ratios,
        )

    def get_first_anchor(self):
        """
        Returns the first anchor box for the current image.
        """
        num_anchor_boxes_per_location = len(self.anchor_size[0]) * len(
            self.aspect_ratios[0]
        )
        return [self.anchor_boxes[idx] for idx in range(num_anchor_boxes_per_location)]

    def get_center_points(self):
        """
        Returns the center points of the anchor boxes for the current image.
        """
        return get_center_bounding_box(self.anchor_boxes)

    def transform_image(self):
        image_transformed = self.rcnn_transform([self.image])
        x = image_transformed[0].tensors[0]

        # make sure it's a numpy.ndarray on the cpu
        x = x.cpu().numpy()

        # from [C, H, W] to [H, W, C] - only for RGB images.
        x = np.moveaxis(x, source=0, destination=-1)

        # Re-normalize
        x = re_normalize(x)

        return x

    @staticmethod
    def transform_boxes(boxes):
        return [make_bbox_napari(box) for box in boxes]

    def get_data(self, sample) -> Dict[str, Any]:
        """
        Returns the image for the current sample.
        """
        self.image = sample["x"]
        data = self.transform_image()

        return {"data": data, "name": "Image"}

    def get_target(self, sample) -> Dict[str, Any]:
        """
        Returns the anchor boxes for the current sample.
        """
        self.anchor_boxes = self.get_anchors()
        self.first_anchor = self.get_first_anchor()
        self.center_points = self.get_center_points()
        self.anchor_points = self.center_points.unique(dim=0)

        boxes = self.transform_boxes(self.first_anchor)

        return {
            "data": boxes,
            "name": "Anchor Boxes",
            "face_color": "transparent",
            "edge_color": "red",
            "edge_width": 2,
        }
