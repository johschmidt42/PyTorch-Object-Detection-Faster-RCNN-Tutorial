import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

import napari
import numpy as np
import torch
from napari.layers import Shapes
from skimage.io import imread
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from datasets import ObjectDetectionDataSet, ObjectDetectionDatasetSingle
from transformations import re_normalize
from utils import color_mapping_func
from utils import enable_gui_qt


def make_bbox_napari(bbox, reverse=False):
    """
    Get the coordinates of the 4 corners of a
    bounding box - expected to be in 'xyxy' format.
    Result can be put directly into a napari shapes layer.

    Order: top-left, bottom-left, bottom-right, top-right
    numpy style [y, x]

    """
    if reverse:
        x = (bbox[:, 1])
        y = (bbox[:, 0])

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

        bbox_rect = np.array(
            [[y1, x1], [y2, x1], [y2, x2], [y1, x2]]
        )
        return bbox_rect


def get_center_bounding_box(boxes: torch.tensor):
    """Returns the center points of given bounding boxes."""
    from torchvision.ops import box_convert
    return box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')[:, :2]


class ViewerBase:
    def napari(self):
        # IPython magic
        enable_gui_qt()

        # napari
        if self.viewer:
            try:
                del self.viewer
            except AttributeError:
                pass
        self.index = 0

        # Init napari instance
        self.viewer = napari.Viewer()

        # Show current sample
        self.show_sample()

        # Key-bindings
        # Press 'n' to get the next sample
        @self.viewer.bind_key('n')
        def next(viewer):
            self.increase_index()  # Increase the index
            self.show_sample()  # Show next sample

        # Press 'b' to get the previous sample
        @self.viewer.bind_key('b')
        def prev(viewer):
            self.decrease_index()  # Decrease the index
            self.show_sample()  # Show next sample

    def increase_index(self):
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0

    def decrease_index(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.dataset) - 1

    def show_sample(self):
        """Overwrite method"""
        pass

    def create_image_layer(self, x, x_name):
        return self.viewer.add_image(x, name=str(x_name))

    def update_image_layer(self, image_layer, x, x_name):
        """Replace the data and the name of a given image_layer"""
        image_layer.data = x
        image_layer.name = str(x_name)

    def get_all_shape_layers(self):
        return [layer for layer in self.viewer.layers if isinstance(layer, Shapes)]

    def remove_all_shape_layers(self):
        all_shape_layers = self.get_all_shape_layers()
        for shape_layer in all_shape_layers:
            self.remove_layer(shape_layer)

    def remove_layer(self, layer):
        self.viewer.layers.remove(layer)


class DatasetViewer(ViewerBase):
    def __init__(self,
                 dataset: ObjectDetectionDataSet,
                 color_mapping: Dict,
                 rccn_transform: GeneralizedRCNNTransform = None):
        self.dataset = dataset
        self.index = 0
        self.color_mapping = color_mapping

        # napari viewer instance
        self.viewer = None

        # rccn_transformer
        self.rccn_transform = rccn_transform

        # current image & shape layer
        self.image_layer = None
        self.shape_layer = None

    def show_sample(self):

        # Get a sample from the dataset
        sample = self.get_sample_dataset(self.index)

        # RCNN-transformer
        if self.rccn_transform is not None:
            sample = self.rcnn_transformer(sample, self.rccn_transform)

        # Transform the sample to numpy, cpu and correct format to visualize
        x, x_name = self.transform_x(sample)
        y, y_name = self.transform_y(sample)

        # Create an image layer
        if self.image_layer not in self.viewer.layers:
            self.image_layer = self.create_image_layer(x, x_name)
        else:
            self.update_image_layer(self.image_layer, x, x_name)

        # Create a shape layer
        if self.shape_layer not in self.viewer.layers:
            self.shape_layer = self.create_shape_layer(y, y_name)
        else:
            self.update_shape_layer(self.shape_layer, y, y_name)

        # Reset view
        self.viewer.reset_view()

        # self.viewer.layers.select_previous()  # focus on inp layer
        # self.viewer.status = f'index: {self.index}, x_name: {x_name}, y_name: {y_name}'

    def get_sample_dataset(self, index):
        return self.dataset[index]

    def transform_x(self, sample):
        # dict unpacking
        x, x_name = sample['x'], sample['x_name']

        # make sure it's a numpy.ndarray on the cpu
        x = x.cpu().numpy()

        # from [C, H, W] to [H, W, C] - only for RGB images.
        if self.check_if_rgb(x):
            x = np.moveaxis(x, source=0, destination=-1)

        # Re-normalize
        x = re_normalize(x)

        return x, x_name

    def transform_y(self, sample):
        # dict unpacking
        y, y_name = sample['y'], sample['y_name']

        # make sure it's numpy.ndarrays on the cpu()
        y = {key: value.cpu().numpy() for key, value in y.items()}

        return y, y_name

    def get_boxes(self, y):
        boxes = y['boxes']

        # transform bboxes to make them napari compatible
        boxes_napari = [make_bbox_napari(box) for box in boxes]

        return boxes_napari

    def get_labels(self, y):
        return y['labels']

    def get_colors(self, y):
        return color_mapping_func(y['labels'], self.color_mapping)

    def get_scores(self, y):
        return y['scores']

    def get_text_parameters(self):
        return {
            'text': '{labels}',
            'size': 10,
            'color': 'white',
            'anchor': 'upper_left',
            'translation': [-1, 0],
        }

    def create_shape_layer(self, y, y_name):
        boxes = self.get_boxes(y)
        labels = self.get_labels(y)
        colors = self.get_colors(y)

        # add properties to shape layer
        # this is required to get the right text for the TextManager
        # the TextManager displays the text on top of the bounding box
        # in this case that's the label

        text_parameters = self.get_text_parameters()  # dict
        properties = {'labels': labels}

        if 'scores' in y.keys():
            scores = self.get_scores(y)
            text_parameters['text'] = 'label: {labels}\nscore: {scores:.2f}'
            properties['scores'] = scores

        # add shape layer
        shape_layer = self.viewer.add_shapes(data=boxes,
                                             face_color='transparent',
                                             edge_color='red',
                                             edge_width=2,
                                             properties=properties,
                                             name=y_name,
                                             text=text_parameters)

        # save some information in the metadata
        self.save_to_metadata(shape_layer, 'boxes', boxes)
        self.save_to_metadata(shape_layer, 'labels', labels)
        self.save_to_metadata(shape_layer, 'colors', colors)

        # add scores
        if 'scores' in y.keys():
            self.save_to_metadata(shape_layer, 'scores', scores)

        # update color
        self.set_colors_of_shapes(shape_layer, self.color_mapping)

        return shape_layer

    def update_shape_layer(self, shape_layer, y, y_name):
        """Remove all shapes and replace the data and the properties"""
        # remove all shapes from layer
        self.select_all_shapes(shape_layer)
        shape_layer.remove_selected()

        boxes = self.get_boxes(y)
        labels = self.get_labels(y)
        colors = self.get_colors(y)

        if 'scores' in y.keys():
            scores = self.get_scores(y)

        # set the current properties
        # this is a workaround for a bug https://github.com/napari/napari/issues/2239
        shape_layer.current_properties['labels'] = labels
        if 'scores' in y.keys():
            shape_layer.current_properties['scores'] = scores

        # add shapes to layer
        shape_layer.add(boxes)

        # set the properties correctly (also part of the workaround)
        shape_layer.properties['labels'] = labels
        if 'scores' in y.keys():
            shape_layer.properties['scores'] = scores

        # override information in the metadata
        self.reset_metadata(shape_layer)
        self.save_to_metadata(shape_layer, 'boxes', boxes)
        self.save_to_metadata(shape_layer, 'labels', labels)
        self.save_to_metadata(shape_layer, 'colors', colors)

        # add scores
        if 'scores' in y.keys():
            self.save_to_metadata(shape_layer, 'scores', scores)

        # update color
        self.set_colors_of_shapes(shape_layer, self.color_mapping)

        # change the name
        shape_layer.name = y_name

    def save_to_metadata(self, shape_layer, key, value):
        shape_layer.metadata[key] = value

    def reset_metadata(self, shape_layer):
        shape_layer.metadata = {}

    def check_if_rgb(self, x):
        # TODO: Check if rgb
        return True

    def get_unique_labels(self, shapes_layer):
        return set(shapes_layer.metadata['labels'])

    def select_all_shapes(self, shape_layer):
        """Selects all shapes within a shape_layer instance."""
        shape_layer.selected_data = set(range(shape_layer.nshapes))

    def select_all_shapes_label(self, shape_layer, label):
        """Select all shapes of certain label"""
        # TODO: Check if label exists
        indices = set(self.get_indices_of_shapes(shape_layer, label))
        shape_layer.selected_data = indices

    def get_indices_of_shapes(self, shapes_layer, label):
        return list(np.argwhere(shapes_layer.properties['labels'] == label).flatten())

    def set_colors_of_shapes(self, shape_layer, color_mapping):
        """Iterate over unique labels and assign a color according to color_mapping."""
        for label in self.get_unique_labels(shape_layer):  # get unique labels
            color = color_mapping[label]  # get color from mapping
            self.set_color_of_shapes(shape_layer, label, color)

    def set_color_of_shapes(self, shapes_layer, label, color):
        """Assign a color to every shape of a certain label"""
        self.select_all_shapes_label(shapes_layer, label)  # select only the correct shapes
        shapes_layer.current_edge_color = color  # change the color of the selected shapes

    def gui_text_properties(self, shape_layer):
        container = self.create_gui_text_properties(shape_layer)
        self.viewer.window.add_dock_widget(container, name='text_properties', area='left')

    def gui_score_slider(self, shape_layer):
        if 'nms_slider' in self.viewer.window._dock_widgets.keys():
            self.remove_gui('nms_slider')
            self.shape_layer.events.name.disconnect(callback=self.shape_layer.events.name.callbacks[0])

        container = self.create_gui_score_slider(shape_layer)
        self.slider = container
        self.viewer.window.add_dock_widget(container, name='score_slider', area='left')

    def gui_nms_slider(self, shape_layer):
        if 'score_slider' in self.viewer.window._dock_widgets.keys():
            self.remove_gui('score_slider')
            self.shape_layer.events.name.disconnect(callback=self.shape_layer.events.name.callbacks[0])

        container = self.create_gui_nms_slider(shape_layer)
        self.slider = container
        self.viewer.window.add_dock_widget(container, name='nms_slider', area='left')

    def remove_gui(self, name):
        widget = self.viewer.window._dock_widgets[name]
        self.viewer.window.remove_dock_widget(widget)

    def create_gui_text_properties(self, shape_layer):
        from magicgui.widgets import Combobox, Container, Slider

        TextColor = Combobox(choices=shape_layer._colors, name='text color', value='white')
        TextSize = Slider(min=1, max=50, name='text size', value=1)

        container = Container(widgets=[TextColor, TextSize])

        def change_text_color(event):
            # This changes the text color
            shape_layer.text.color = str(TextColor.value)

        def change_text_size(event):
            # This changes the text size
            shape_layer.text.size = int(TextSize.value)

        TextColor.changed.connect(change_text_color)
        TextSize.changed.connect(change_text_size)

        return container

    def create_gui_score_slider(self, shape_layer):
        from magicgui.widgets import FloatSlider, Container, Label

        slider = FloatSlider(min=0.0, max=1.0, step=0.05, name='Score', value=0.0)
        slider_label = Label(name='Score_threshold', value=0.0)

        container = Container(widgets=[slider, slider_label])

        def change_boxes(event, shape_layer=shape_layer):
            # remove all shapes from layer
            self.select_all_shapes(shape_layer)
            shape_layer.remove_selected()

            # create mask and new information
            mask = np.where(shape_layer.metadata['scores'] > slider.value)
            new_boxes = np.asarray(shape_layer.metadata['boxes'])[mask]
            new_labels = shape_layer.metadata['labels'][mask]
            new_scores = shape_layer.metadata['scores'][mask]

            # set the current properties as workaround
            shape_layer.current_properties['labels'] = new_labels
            shape_layer.current_properties['scores'] = new_scores

            # add shapes to layer
            if new_boxes.size > 0:
                shape_layer.add(list(new_boxes))

            # set the properties
            shape_layer.properties['labels'] = new_labels
            shape_layer.properties['scores'] = new_scores

            # change label
            slider_label.value = str(slider.value)

        slider.changed.connect(change_boxes)

        # invoke scoring
        container.Score.value = 0.0

        # event triggered when the name of the shape layer changes
        self.shape_layer.events.name.connect(change_boxes)

        return container

    def create_gui_nms_slider(self, shape_layer):
        from magicgui.widgets import FloatSlider, Container, Label
        from torchvision.ops import nms
        slider = FloatSlider(min=0.0, max=1.0, step=0.01, name='NMS')
        slider_label = Label(name='IoU_threshold')

        container = Container(widgets=[slider, slider_label])

        def change_boxes(event, shape_layer=shape_layer):
            # remove all shapes from layer
            self.select_all_shapes(shape_layer)
            shape_layer.remove_selected()

            # create mask and new information
            boxes = torch.tensor([make_bbox_napari(box, reverse=True) for box in shape_layer.metadata['boxes']])
            scores = torch.tensor(shape_layer.metadata['scores'])

            if boxes.size()[0] > 0:
                mask = nms(boxes, scores, slider.value)  # torch.tensor
                mask = (np.array(mask),)

                new_boxes = np.asarray(shape_layer.metadata['boxes'])[mask]
                new_labels = shape_layer.metadata['labels'][mask]
                new_scores = shape_layer.metadata['scores'][mask]

                # set the current properties as workaround
                shape_layer.current_properties['labels'] = new_labels
                shape_layer.current_properties['scores'] = new_scores

                # add shapes to layer
                if new_boxes.size > 0:
                    shape_layer.add(list(new_boxes))

                # set the properties
                shape_layer.properties['labels'] = new_labels
                shape_layer.properties['scores'] = new_scores

                # set temporary information
                shape_layer.metadata['boxes_nms'] = list(new_boxes)
                shape_layer.metadata['labels_nms'] = new_labels
                shape_layer.metadata['scores_nms'] = new_scores

            # change label
            slider_label.value = str(slider.value)

        slider.changed.connect(change_boxes)

        # invoke nms
        container.NMS.value = 1.0

        # event triggered when the name of the shape layer changes
        self.shape_layer.events.name.connect(change_boxes)

        return container

    def rcnn_transformer(self, sample, transform):
        # dict unpacking
        x, x_name, y, y_name = sample['x'], sample['x_name'], sample['y'], sample['y_name']

        x, y = transform([x], [y])
        x, y = x.tensors[0], y[0]

        return {'x': x, 'y': y, 'x_name': x_name, 'y_name': y_name}


class DatasetViewerSingle(DatasetViewer):
    def __init__(self,
                 dataset: ObjectDetectionDatasetSingle,
                 rccn_transform: GeneralizedRCNNTransform = None):
        self.dataset = dataset
        self.index = 0

        # napari viewer instance
        self.viewer = None

        # rccn_transformer
        self.rccn_transform = rccn_transform

        # current image & shape layer
        self.image_layer = None
        self.shape_layer = None

    def show_sample(self):

        # Get a sample from the dataset
        sample = self.get_sample_dataset(self.index)

        # RCNN-transformer
        if self.rccn_transform is not None:
            sample = self.rcnn_transformer(sample, self.rccn_transform)

        # Transform the sample to numpy, cpu and correct format to visualize
        x, x_name = self.transform_x(sample)

        # Create an image layer
        if self.image_layer not in self.viewer.layers:
            self.image_layer = self.create_image_layer(x, x_name)
        else:
            self.update_image_layer(self.image_layer, x, x_name)

        # Reset view
        self.viewer.reset_view()

    def rcnn_transformer(self, sample, transform):
        # dict unpacking
        x, x_name = sample['x'], sample['x_name']

        x, _ = transform([x])
        x, _ = x.tensors[0], _

        return {'x': x, 'x_name': x_name}


class Annotator(ViewerBase):
    def __init__(self,
                 image_ids: pathlib.Path,
                 annotation_ids: pathlib.Path = None,
                 color_mapping: Dict = {}):

        self.image_ids = image_ids
        self.annotation_ids = annotation_ids

        self.index = 0
        self.color_mapping = color_mapping

        # napari viewer instance
        self.viewer = None

        # current image & shape layers
        self.image_layer = None
        self.shape_layers = []

        # init annotations
        self.annotations = self.init_annotations()

        # load annotations from disk
        if self.annotation_ids is not None:
            self.load_annotations()

        # edge width for shapes
        self.edge_width = 2.0

        # current annotation object
        self.annotation_object = None

    def init_annotations(self):
        @dataclass
        class AnnotationObject:
            name: str
            boxes: np.ndarray
            labels: np.ndarray

            def __bool__(self):
                return True if self.boxes.size > 0 else False

        return [AnnotationObject(name=image_id.stem, boxes=np.array([]), labels=np.array([])) for image_id in
                self.image_ids]

    def increase_index(self):
        self.index += 1
        if self.index >= len(self.image_ids):
            self.index = 0

    def decrease_index(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.image_ids) - 1

    def show_sample(self):
        # Get an image_id
        image_id = self.get_image_id(self.index)

        # Load the image
        x = self.load_x(image_id)

        # Transform the image
        x = self.transform_x(x)

        # Create or update image layer
        if self.image_layer not in self.viewer.layers:
            self.image_layer = self.create_image_layer(x, image_id)
        else:
            self.update_image_layer(self.image_layer, x, image_id)

        # Save annotations in annotation_object (any changes will be saved/overwritten)
        self.save_annotations(self.annotation_object)

        # Update current annotation object
        self.annotation_object = self.get_annotation_object(self.index)

        # Remove all shape layers
        self.remove_all_shape_layers()

        # Create new shape layers
        self.shape_layers = self.create_shape_layers(self.annotation_object)

        # Reset view
        self.viewer.reset_view()

    def get_image_id(self, index):
        return self.image_ids[index]

    def get_annotation_object(self, index):
        return self.annotations[index]

    def transform_x(self, x):
        # Re-normalize
        x = re_normalize(x)

        return x

    def load_x(self, image_id):
        return imread(image_id)

    def load_annotations(self):
        # generate a list of names, annotation file must have the same name (stem) as the image.
        annotation_object_names = [annotation_object.name for annotation_object in self.annotations]
        # iterate over the annotation_ids
        for annotation_id in self.annotation_ids:
            annotation_name = annotation_id.stem

            index_list = self.get_indices_of_sequence(annotation_name, annotation_object_names)
            if index_list:
                # TODO: check if it finds more than one index
                idx = index_list[0]  # get index value of index_list
                annotation_file = torch.load(annotation_id)  # read file

                # store them as np.ndarrays
                boxes = np.array(annotation_file['boxes'])  # get boxes
                boxes = np.array([make_bbox_napari(box) for box in boxes])  # transform to napari boxes
                labels = np.array(annotation_file['labels'])  # get labels

                # add information to annotation object
                self.annotations[idx].boxes = boxes
                self.annotations[idx].labels = labels

    def get_indices_of_sequence(self, string, sequence):
        return [idx for idx, element in enumerate(sequence) if element == string]

    def get_annotations_from_shape_layers(self):
        all_shape_layers = self.get_all_shape_layers()
        if all_shape_layers:
            all_boxes = []
            all_labels = []
            for shape_layer in all_shape_layers:
                boxes = np.array(shape_layer.data)  # numpy.ndarray
                num_labels = len(boxes)
                label = shape_layer.metadata['label']  # read the label from the metadata
                all_boxes.append(boxes)
                all_labels.append(np.repeat(np.array([label]), num_labels))

            all_boxes = np.concatenate(all_boxes, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            return all_boxes, all_labels

    def save_annotations(self, annotation_object):
        # get the annotation information from every shape layer
        information = self.get_annotations_from_shape_layers()

        if information:
            boxes, labels = information  # tuple unpacking

            # update the current annotation object
            self.update_annotation_object(annotation_object, boxes, labels)

    def update_annotation_object(self, annotation_object, boxes, labels):
        annotation_object.boxes = boxes
        annotation_object.labels = labels

    def create_shape_layers(self, annotation_object):
        unique_labels = np.unique(annotation_object.labels)

        shape_layers = [self.create_shape_layer(label, annotation_object) for label in unique_labels]

        return shape_layers

    def create_shape_layer(self, label, annotation_object):
        mask = annotation_object.labels == label

        boxes = annotation_object.boxes[mask]

        layer = self.viewer.add_shapes(data=boxes,
                                       edge_color=self.color_mapping.get(label, 'black'),
                                       edge_width=self.edge_width,
                                       face_color='transparent',
                                       name=str(label))

        layer.metadata['label'] = label

        return layer

    def add_class(self, label, color: str):
        self.color_mapping[label] = color
        layer = self.viewer.add_shapes(edge_color=self.color_mapping.get(label, 'black'),
                                       edge_width=self.edge_width,
                                       face_color='transparent',
                                       name=str(label))

        layer.metadata['label'] = label

    def export(self, directory: pathlib.Path, name: str = None):
        """Saves the current annotations to disk."""
        self.save_annotations(self.annotation_object)  # Save annotations in current annotation_object

        boxes = [make_bbox_napari(box, reverse=True) for box in self.annotation_object.boxes]
        labels = self.annotation_object.labels
        if name is None:
            name = pathlib.Path(self.annotation_object.name).with_suffix('.pt')

        file = {
            'labels': labels,
            'boxes': boxes
        }

        torch.save(file, directory / name)

        print(f'Annotation {str(name)} saved to {directory}')

    def export_all(self, directory: pathlib.Path):
        """Saves all available annotations to disk."""
        self.save_annotations(self.annotation_object)  # Save annotations in current annotation_object

        for annotation_object in self.annotations:
            if annotation_object:
                boxes = [make_bbox_napari(box, reverse=True) for box in annotation_object.boxes]
                labels = annotation_object.labels
                name = pathlib.Path(annotation_object.name).with_suffix('.pt')

                file = {
                    'labels': labels,
                    'boxes': boxes
                }

                torch.save(file, directory / name)

                print(f'Annotation {str(name)} saved to {directory}')


class AnchorViewer(ViewerBase):
    def __init__(self,
                 image: torch.tensor,
                 rcnn_transform: GeneralizedRCNNTransform,
                 feature_map_size: tuple,
                 anchor_size: Tuple[tuple] = ((128, 256, 512),),
                 aspect_ratios: Tuple[tuple] = ((1.0,),),
                 ):
        self.image = image
        self.rcnn_transform = rcnn_transform
        self.feature_map_size = feature_map_size
        self.anchor_size = anchor_size
        self.aspect_ratios = aspect_ratios

        self.anchor_boxes = None

        # napari viewer instance
        self.viewer = None

    def napari(self):
        # IPython magic
        enable_gui_qt()

        # napari
        if self.viewer:
            try:
                del self.viewer
            except AttributeError:
                pass

        # Init napari instance
        self.viewer = napari.Viewer()

        # Show image
        self.show_sample()

    def get_anchors(self):
        from anchor_generator import get_anchor_boxes
        return get_anchor_boxes(self.image,
                                self.rcnn_transform,
                                self.feature_map_size,
                                self.anchor_size,
                                self.aspect_ratios)

    def get_first_anchor(self):
        num_anchor_boxes_per_location = len(self.anchor_size[0]) * len(self.aspect_ratios[0])
        return [self.anchor_boxes[idx] for idx in range(num_anchor_boxes_per_location)]

    def get_center_points(self):
        return get_center_bounding_box(self.anchor_boxes)

    def show_sample(self):
        self.anchor_boxes = self.get_anchors()
        self.first_anchor = self.get_first_anchor()
        self.center_points = self.get_center_points()
        self.anchor_points = self.center_points.unique(dim=0)

        # Transform the image to numpy, cpu and correct format to visualize
        image = self.transform_image(self.image)
        boxes = self.transform_boxes(self.first_anchor)

        # Create an image layer
        self.viewer.add_image(image, name='Image')

        # Create a shape layer
        self.viewer.add_shapes(data=boxes,
                               face_color='transparent',
                               edge_color='red',
                               edge_width=2,
                               name='Boxes',
                               )

        # Create a point layer
        self.viewer.add_points(data=self.anchor_points)

        # Reset view
        self.viewer.reset_view()

    def transform_image(self, x):
        image_transformed = self.rcnn_transform([self.image])
        x = image_transformed[0].tensors[0]

        # make sure it's a numpy.ndarray on the cpu
        x = x.cpu().numpy()

        # from [C, H, W] to [H, W, C] - only for RGB images.
        x = np.moveaxis(x, source=0, destination=-1)

        # Re-normalize
        x = re_normalize(x)

        return x

    def transform_boxes(self, boxes):
        return [make_bbox_napari(box) for box in boxes]
