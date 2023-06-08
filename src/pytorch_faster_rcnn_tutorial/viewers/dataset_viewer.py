# python -W ignore script.py
import abc
import logging
from abc import ABC
from typing import Any, Dict, Optional, Union

import napari
import numpy as np
from napari.layers import Image, Shapes
from torch.utils.data import Dataset

logger: logging.Logger = logging.getLogger(__name__)


def make_bbox_napari(bbox: np.ndarray, reverse: bool = False) -> np.ndarray:
    """
    Get the coordinates of the 4 corners of a
    bounding box - expected to be in 'xyxy' format.
    Result can be put directly into a napari shapes layer.

    Order: top-left, bottom-left, bottom-right, top-right
    numpy style [y, x]

    """
    if reverse:
        x = bbox[:, 1]
        y = bbox[:, 0]

        x1 = x.min()
        y1 = y.min()
        x2 = x.max()
        y2 = y.max()

        return np.array([x1, y1, x2, y2])

    else:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        bbox_rect = np.array([[y1, x1], [y2, x1], [y2, x2], [y1, x2]])
        return bbox_rect


class DatasetViewer(ABC):
    """
    Abstract class for visualizing image datasets with napari.
    Target type can be either 'Image' (e.g. semantic segmentation) or 'Shapes' (object detection).
    This abstract class is meant to be inherited from and the following methods need to be implemented.

    Abstract methods:
        - get_data
        - get_target
    """

    def __init__(
        self, torch_dataset: Dataset, target_type: Union[Image, Shapes, None] = None
    ):
        self.dataset: Dataset = torch_dataset
        self.index: int = 0
        self.viewer: napari.Viewer = self._init_napari_viewer()

        self.image_layer: Optional[Any] = None
        self.target_layer: Optional[Any] = None

        self.target_type: Union[Image, Shapes, None] = target_type

        # Key-bindings
        @self.viewer.bind_key("n")
        def _next(viewer):
            """
            Press 'n' to get the next sample
            """
            self._increase_index()
            self.show_image()

        @self.viewer.bind_key("b")
        def _back(viewer):
            """
            Press 'b' to get the previous sample
            """
            self._decrease_index()
            self.show_image()

    @staticmethod
    def _init_napari_viewer() -> napari.Viewer:
        """
        Initialize the napari viewer
        """
        viewer: napari.Viewer = napari.Viewer()
        return viewer

    def _increase_index(self) -> None:
        """
        Increase the index by 1
        """
        self.index += 1
        if self.index >= len(self.dataset):
            self.index: int = 0

    def _decrease_index(self) -> None:
        """
        Decrease the index by 1
        """
        self.index -= 1
        if self.index < 0:
            self.index: int = len(self.dataset) - 1

    def _create_image_layer(self, x) -> Image:
        """
        Create a new image layer
        """
        return self.viewer.add_image(**x)

    def _update_image_layer(self, x: Dict[str, Any]) -> None:
        """
        Update the image layer with the new data.
        Instead of creating a new image layer, we can also update the existing one.
        """
        self.image_layer.data = x["data"]
        self.image_layer.name = x["name"]

    def _create_target_layer(self, y: Dict[str, Any]) -> Union[Image, Shapes]:
        """
        Create a new target layer.
        The target layer can be either an 'Image' or a 'Shapes' layer.
        """

        # remove old target layer
        if self.target_layer is not None:
            self.viewer.layers.remove(self.target_layer)

        if self.target_type == Image:
            # semantic segmentation -> 'Image' layer
            return self.viewer.add_image(**y)
        elif self.target_type == Shapes:
            # object detection -> 'Shapes' layer
            if y["data"]:
                shapes: Shapes = self.viewer.add_shapes(**y)
                shapes.editable = False
                return shapes
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

    def show_image(self) -> None:
        """
        Show the image at the current index.
        """

        sample: Any = self.dataset[self.index]

        # x
        x: Dict[str, Any] = self.get_data(sample=sample)

        if self.image_layer not in self.viewer.layers:
            self.image_layer: Image = self._create_image_layer(x=x)
        else:
            self._update_image_layer(x=x)

        # y
        y: Dict[str, Any] = self.get_target(sample=sample)

        if y is not None:
            self.target_layer: Union[Image, Shapes] = self._create_target_layer(y=y)

        # reset the camera view
        self.viewer.reset_view()

    @abc.abstractmethod
    def get_data(self, sample) -> Dict[str, Any]:
        """
        Get the data of the sample (e.g. image data)
        Output dict must contain the following keys:
            - "data": the image data
            - "name": the name of the image
        """
        ...

    @abc.abstractmethod
    def get_target(self, sample) -> Dict[str, Any]:
        """
        Get the target of the sample (e.g. bounding boxes, image data, names, etc.)
        Output dict must contain the following keys:
            - "data": the target data
            - "name": the name of the target
        """
        ...
