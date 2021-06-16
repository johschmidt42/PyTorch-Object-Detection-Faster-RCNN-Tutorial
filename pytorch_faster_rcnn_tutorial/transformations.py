from functools import partial
from typing import List, Callable

import albumentations as A
import numpy as np
import torch
from sklearn.externals._pilutil import bytescale
from torchvision.ops import nms


def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (no clipping)"""
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out


def normalize(inp: np.ndarray, mean: float, std: float):
    """Normalize based on mean and standard deviation."""
    inp_out = (inp - mean) / std
    return inp_out


def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255
                 ):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


def clip_bbs(inp: np.ndarray,
             bbs: np.ndarray):
    """
    If the bounding boxes exceed one dimension, they are clipped to the dim's maximum.
    Bounding boxes are expected to be in xyxy format.
    Example: x_value=224 but x_shape=200 -> x1=199
    """

    def clip(value: int, max: int):

        if value >= max - 1:
            value = max - 1
        elif value <= 0:
            value = 0

        return value

    output = []
    for bb in bbs:
        x1, y1, x2, y2 = tuple(bb)
        x_shape = inp.shape[1]
        y_shape = inp.shape[0]

        x1 = clip(x1, x_shape)
        y1 = clip(y1, y_shape)
        x2 = clip(x2, x_shape)
        y2 = clip(y2, y_shape)

        output.append([x1, y1, x2, y2])

    return np.array(output)


def map_class_to_int(labels: List[str], mapping: dict):
    """Maps a string to an integer."""
    labels = np.array(labels)
    dummy = np.empty_like(labels)
    for key, value in mapping.items():
        dummy[labels == key] = value

    return dummy.astype(np.uint8)


def apply_nms(target: dict, iou_threshold):
    """Non-maximum Suppression"""
    boxes = torch.tensor(target['boxes'])
    labels = torch.tensor(target['labels'])
    scores = torch.tensor(target['scores'])

    if boxes.size()[0] > 0:
        mask = nms(boxes, scores, iou_threshold=iou_threshold)
        mask = (np.array(mask),)

        target['boxes'] = np.asarray(boxes)[mask]
        target['labels'] = np.asarray(labels)[mask]
        target['scores'] = np.asarray(scores)[mask]

    return target


def apply_score_threshold(target: dict, score_threshold):
    """Removes bounding box predictions with low scores."""
    boxes = target['boxes']
    labels = target['labels']
    scores = target['scores']

    mask = np.where(scores > score_threshold)
    target['boxes'] = boxes[mask]
    target['labels'] = labels[mask]
    target['scores'] = scores[mask]

    return target


class Repr:
    """Evaluatable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'


class FunctionWrapperSingle(Repr):
    """A function wrapper that returns a partial for input only."""

    def __init__(self, function: Callable, *args, **kwargs):
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray): return self.function(inp)


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, input: bool = True, target: bool = False, *args, **kwargs):
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, inp: np.ndarray, tar: dict):
        if self.input: inp = self.function(inp)
        if self.target: tar = self.function(tar)
        return inp, tar


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target


class ComposeSingle(Compose):
    """Composes transforms for input only."""

    def __call__(self, inp: np.ndarray):
        for t in self.transforms:
            inp = t(inp)
        return inp


class AlbumentationWrapper(Repr):
    """
    A wrapper for the albumentation package.
    Bounding boxes are expected to be in xyxy format (pascal_voc).
    Bounding boxes cannot be larger than the spatial image's dimensions.
    Use Clip() if your bounding boxes are outside of the image, before using this wrapper.
    """

    def __init__(self, albumentation: Callable, format: str = 'pascal_voc'):
        self.albumentation = albumentation
        self.format = format

    def __call__(self, inp: np.ndarray, tar: dict):
        # input, target
        transform = A.Compose([
            self.albumentation
        ], bbox_params=A.BboxParams(format=self.format, label_fields=['class_labels']))

        out_dict = transform(image=inp, bboxes=tar['boxes'], class_labels=tar['labels'])

        input_out = np.array(out_dict['image'])
        boxes = np.array(out_dict['bboxes'])
        labels = np.array(out_dict['class_labels'])

        tar['boxes'] = boxes
        tar['labels'] = labels

        return input_out, tar


class Clip(Repr):
    """
    If the bounding boxes exceed one dimension, they are clipped to the dim's maximum.
    Bounding boxes are expected to be in xyxy format.
    Example: x_value=224 but x_shape=200 -> x1=199
    """

    def __call__(self, inp: np.ndarray, tar: dict):
        new_boxes = clip_bbs(inp=inp, bbs=tar['boxes'])
        tar['boxes'] = new_boxes

        return inp, tar
