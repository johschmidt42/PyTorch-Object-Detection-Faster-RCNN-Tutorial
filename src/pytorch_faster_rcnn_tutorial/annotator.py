import logging
import pathlib
from typing import Any, Dict, List

import numpy as np
from napari.layers import Shapes

from pytorch_faster_rcnn_tutorial.datasets import ObjectDetectionDatasetSingle
from pytorch_faster_rcnn_tutorial.transformations import re_normalize
from pytorch_faster_rcnn_tutorial.utils import save_json
from pytorch_faster_rcnn_tutorial.viewers.dataset_viewer import (
    DatasetViewer,
    make_bbox_napari,
)

logger: logging.Logger = logging.getLogger(__name__)


class Annotator(DatasetViewer):
    """
    Very simple annotation tool for object detection datasets (FasterRCNN).
    Based on napari.
    """

    def __init__(
        self,
        dataset: ObjectDetectionDatasetSingle,
        color_mapping: Dict[str, Dict[str, Any]] = None,
    ):
        super().__init__(torch_dataset=dataset, target_type=Shapes)

        self.color_mapping = color_mapping if color_mapping is not None else {}

        # TODO: add a way to load the annotations from a json file like before

        # Key-bindings
        # overwriting the navigation key-bindings of the super class 'DatasetViewer' to add the save_boxes function
        @self.viewer.bind_key("n", overwrite=True)
        def _next(viewer):
            """
            Press 'n' to get the next sample
            """
            self._increase_index()
            self.save_boxes()
            self.show_image()

        @self.viewer.bind_key("b", overwrite=True)
        def _back(viewer):
            """
            Press 'b' to get the previous sample
            """
            self._decrease_index()
            self.save_boxes()
            self.show_image()

    def save_boxes(self) -> None:
        """
        Save the boxes of the current image to the metadata of the image layer.
        """
        if self.image_layer is None:
            return None

        shapes_layers: List[Shapes] = self._get_all_shapes_layer()

        for layer in shapes_layers:
            self.image_layer.metadata[self.image_layer.name][layer.name] = layer.data
            layer.data = []

    def get_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the data from the sample and transform it to be napari compatible.
        """
        logger.info(f"Input sample: {sample['x_name']}\nShape: {sample['x'].shape}")

        data = sample["x"]
        data = self._transform_x(data)

        name: str = sample["x_name"]

        return {"data": data, "name": name}

    def _get_all_shapes_layer(self) -> List[Shapes]:
        """
        Get all shapes layers from the napari viewer.
        """
        return [layer for layer in self.viewer.layers if isinstance(layer, Shapes)]

    def get_target(self, sample: Dict[str, Any]) -> None:
        """
        Get the target from the metadata of the image layer OR initialize it.
        """
        shapes_layers: List[Shapes] = self._get_all_shapes_layer()

        # init metadata
        if self.image_layer.name not in self.image_layer.metadata.keys():
            self.image_layer.metadata[self.image_layer.name] = {}
            return None

        for layer in shapes_layers:
            for box in self.image_layer.metadata[self.image_layer.name][layer.name]:
                layer.add(box)

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

    def add_class(self, label: str, color: str) -> None:
        """
        Adds a class to the color mapping.
        Creates a new shapes layer with the given color.
        """

        self.color_mapping[label] = {"color": color}
        self.viewer.add_shapes(
            edge_color=self.color_mapping[label]["color"],
            edge_width=2,
            face_color="transparent",
            name=label,
        )

    def export(self, directory: pathlib.Path) -> None:
        """
        Saves all available annotations to disk in JSON format.
        Every image gets its own JSON file.
        Every JSON file contains a list of labels and a list of boxes.
        The boxes are in the format [xyxy].
        """
        for image_name, data in self.image_layer.metadata.items():
            boxes_per_image = []
            labels_per_image = []

            for label, boxes in data.items():
                if len(boxes) == 0:
                    continue
                num_labels = len(boxes)
                boxes_np = np.array(boxes)
                boxes_per_image.append(boxes_np)
                labels_per_image.append(np.repeat(np.array([label]), num_labels))

            if boxes_per_image:
                boxes_per_image = np.concatenate(boxes_per_image, axis=0)
                labels_per_image = np.concatenate(labels_per_image, axis=0)

                boxes_per_image = [
                    make_bbox_napari(box, reverse=True).tolist()
                    for box in boxes_per_image
                ]

                labels = labels_per_image.tolist()

                name: pathlib.Path = pathlib.Path(image_name).with_suffix(".json")

                file: dict = {"labels": labels, "boxes": boxes_per_image}

                save_json(file, path=directory / name)

                logger.info(f"Annotation {str(name)} saved to {directory}")
