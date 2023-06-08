import logging
from typing import Any, Dict, List, Optional

import numpy as np
from napari.layers import Shapes
from torch.utils.data import Dataset
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from pytorch_faster_rcnn_tutorial.transformations import re_normalize
from pytorch_faster_rcnn_tutorial.utils import color_mapping_func
from pytorch_faster_rcnn_tutorial.viewers.dataset_viewer import (
    DatasetViewer,
    make_bbox_napari,
)

logger: logging.Logger = logging.getLogger(__name__)


class ObjectDetectionViewer(DatasetViewer):
    """
    Viewer for object detection datasets (FasterRCNN).
    """

    def __init__(
        self,
        dataset: Dataset,
        color_mapping: Dict[int, str],
        rcnn_transform: Optional[GeneralizedRCNNTransform] = None,
    ):
        super().__init__(torch_dataset=dataset, target_type=Shapes)

        # color mapping
        self.color_mapping: Dict[int, str] = color_mapping

        # rcnn_transformation
        self.rcnn_transform: Optional[GeneralizedRCNNTransform] = rcnn_transform

    def get_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the data from the sample and transform it to be napari compatible.
        """
        logger.info(f"Input sample: {sample['x_name']}\nShape: {sample['x'].shape}")

        # RCNN-transformer
        if self.rcnn_transform is not None:
            sample: Dict[str, Any] = self._rcnn_transformer(
                sample=sample, transform=self.rcnn_transform
            )
            logger.info(
                f"Transformed input sample: {sample['x_name']}\nShape: {sample['x'].shape}"
            )

        data = sample["x"]
        data = self._transform_x(data)

        name: str = sample["x_name"]

        return {"data": data, "name": name}

    def get_target(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the target from the sample and transform it to be napari compatible.
        """
        logger.info(f"Target sample: {sample['y_name']}\n{sample['y']}")

        # RCNN-transformer
        if self.rcnn_transform is not None:
            sample: Dict[str, Any] = self._rcnn_transformer(
                sample=sample, transform=self.rcnn_transform
            )
            logger.info(f"Transformed target sample: {sample['y_name']}\n{sample['y']}")

        data = sample["y"]
        data = self._transform_y(data)

        # transform bboxes to make them napari compatible
        boxes: List[np.ndarray] = [make_bbox_napari(box) for box in data["boxes"]]
        labels: List[str] = data["labels"]
        colors: List[str] = color_mapping_func(
            labels=data["labels"], mapping=self.color_mapping
        )

        name: str = sample["y_name"]

        # save to metadata
        metadata: dict = {
            "boxes": boxes,
            "labels": labels,
            "colors": colors,
            "color_mapping": self.color_mapping,
        }

        # add text parameters
        text_parameters: Dict[str, Any] = self._get_text_parameters()

        # add properties
        properties: Dict[str, List[str]] = {"labels": labels}

        # add scores if available
        if "scores" in data.keys():
            scores: List[str] = data["scores"]
            text_parameters["text"] = "label: {labels}\nscore: {scores:.2f}"
            properties["scores"] = scores
            metadata["scores"] = scores

        return {
            "data": boxes,
            "name": name,
            "face_color": "transparent",
            "edge_color": colors,
            "edge_width": 2,
            "metadata": metadata,
            "text": text_parameters,
            "properties": properties,
        }

    def _transform_x(self, x):
        # make sure it's a numpy.ndarray on the cpu
        x = x.cpu().numpy()

        # from [C, H, W] to [H, W, C] - only for RGB images.
        if self._check_if_rgb(x):
            x = np.moveaxis(x, source=0, destination=-1)

        # Re-normalize
        x = re_normalize(x)

        return x

    @staticmethod
    def _transform_y(y):
        # make sure it's numpy.ndarrays on the cpu()
        y = {key: value.cpu().numpy() for key, value in y.items()}

        return y

    @staticmethod
    def _check_if_rgb(x):
        """
        Checks if the shape of the first dim (channel dim) is 3
        """
        # TODO: RGBA images have 4 channels -> raise Error
        if x.shape[0] == 3:
            return True
        else:
            raise AssertionError(
                f"The channel dimension is supposed to be 3 for RGB images."
                f"This image has a channel dimension of size {x.shape[0]}"
            )

    @staticmethod
    def _get_text_parameters() -> Dict[str, Any]:
        """
        Get the text parameters for the labels.
        """
        return {
            "text": "{labels}",
            "size": 10,
            "color": "white",
            "anchor": "upper_left",
            "translation": [-1, 0],
        }

    @staticmethod
    def _rcnn_transformer(
        sample: Dict[str, Any], transform: GeneralizedRCNNTransform
    ) -> Dict[str, Any]:
        # dict unpacking
        x, x_name, y, y_name = (
            sample["x"],
            sample["x_name"],
            sample["y"],
            sample["y_name"],
        )

        x, y = transform([x], [y])
        x, y = x.tensors[0], y[0]

        return {"x": x, "y": y, "x_name": x_name, "y_name": y_name}


class ObjectDetectionViewerSingle(ObjectDetectionViewer):
    @staticmethod
    def _rcnn_transformer(
        sample: Dict[str, Any], transform: GeneralizedRCNNTransform
    ) -> Dict[str, Any]:
        x, x_name = sample["x"], sample["x_name"]

        x, _ = transform([x])
        x, _ = x.tensors[0], _

        return {"x": x, "x_name": x_name}

    def get_target(self, sample: Dict[str, Any]) -> None:
        return None
