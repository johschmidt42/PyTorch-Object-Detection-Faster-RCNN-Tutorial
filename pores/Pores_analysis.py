# %% Imports
import pathlib

import numpy as np

from datasets import ObjectDetectionDataSet
from transformations import ComposeDouble, Clip, FunctionWrapperDouble
from transformations import normalize_01
from utils import get_filenames_of_path

# %% hyper-parameters
params = {'SEED': 42,
          'MIN_SIZE': 256,
          'MAX_SIZE': 256,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225]}

# %% root directory
root = pathlib.Path(r"F:\Juan\object_detection\training_data")

# %% input and target files
inputs = get_filenames_of_path(root / 'input_224x224')
targets = get_filenames_of_path(root / 'target_224x224')

inputs.sort()
targets.sort()

# %% transformations and augmentations
transforms = ComposeDouble([
    Clip(),
    # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
    # AlbumentationWrapper(albumentation=A.VerticalFlip(p=0.5)),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# %% random seed
from pytorch_lightning import seed_everything

seed_everything(params['SEED'])

# %% dataset
dataset = ObjectDetectionDataSet(inputs=inputs,
                                 targets=targets,
                                 transform=transforms,
                                 use_cache=False,
                                 convert_to_format=None,
                                 mapping=None)

# %% visualize dataset
color_mapping = {
    1: 'red',
    2: 'blue'
}

from visual import DatasetViewer

from torchvision.models.detection.transform import GeneralizedRCNNTransform

transform = GeneralizedRCNNTransform(min_size=params['MIN_SIZE'],
                                     max_size=params['MAX_SIZE'],
                                     image_mean=params['IMG_MEAN'],
                                     image_std=params['IMG_STD'])

datasetviewer = DatasetViewer(dataset, color_mapping, rccn_transform=transform)
datasetviewer.napari()
datasetviewer.viewer.window._qt_window.setGeometry(-1280, 0, 1280, 1400)
datasetviewer.viewer.window._qt_window.raise_()  # raise to foreground

# %% anchor boxes
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from visual import AnchorViewer

transform = GeneralizedRCNNTransform(min_size=params['MIN_SIZE'],
                                     max_size=params['MAX_SIZE'],
                                     image_mean=params['IMG_MEAN'],
                                     image_std=params['IMG_STD'])

image = dataset[0]['x']  # ObjectDetectionDataSet
from backbone_resnet import get_resnet_backbone
import torch

dummy_size = (1, 3, 256, 256)
dummy_input = torch.randn(dummy_size)
backbone = get_resnet_backbone(backbone_name='resnet18')

with torch.no_grad():
    output = backbone(dummy_input)

import torchsummary

torchsummary.summary(model=backbone, input_size=dummy_size[1:], device='cpu')

params = {
    'ANCHOR_SIZE': ((16, 18, 20),),
    'ASPECT_RATIOS': ((1.0,),),
}

# 16, 18, 20, 22


feature_map_size = (128, 32, 32)
anchorviewer = AnchorViewer(image=image,
                            rcnn_transform=transform,
                            feature_map_size=feature_map_size,
                            anchor_size=params['ANCHOR_SIZE'],
                            aspect_ratios=params['ASPECT_RATIOS']
                            )
anchorviewer.napari()
anchorviewer.viewer.window._qt_window.setGeometry(-1280, 0, 1280, 1400)
anchorviewer.viewer.window._qt_window.raise_()  # raise to foreground

point_layer = anchorviewer.viewer.layers[-1]
point_layer.selected_data = {i for i in range(point_layer.data.__len__())}
point_layer.current_size = 1
