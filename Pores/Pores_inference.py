# %% imports
import ast
import pathlib

import neptune
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.transform import resize, rescale

from api_key_neptune import get_api_key
from datasets import ObjectDetectionDatasetSingle, ObjectDetectionDataSet
from transformations import ComposeSingle, FunctionWrapperSingle, normalize_01, ComposeDouble, FunctionWrapperDouble
from utils import get_filenames_of_path, collate_single

# %% hyper-parameters
params = {
    'DOWNLOAD': False,
    'EXPERIMENT': 'POR-22',
    'INPUT_DIR': r'F:\Juan\object_detection\training_data',
    'PREDICTIONS_PATH': r'F:\Juan\object_detection\predictions',
    'MODEL_DIR': r'F:\Juan\object_detection\POR-22\checkpoints\epoch=65-step=197.ckpt',
    # 'DOWNLOAD_PATH': r'C:\Users\johan\Desktop\Johannes\Pores',
    'OWNER': 'johschmidt42',
    'PROJECT': 'Pores',
}

# %% input files
inputs = get_filenames_of_path(pathlib.Path(params['INPUT_DIR']) / 'input_224x224_bb')
inputs.sort()

inputs = inputs[-6:]

# input_image = [inputs[-2]]
# from skimage.io import imread
#
# img = imread(input_image[0])

# %% transformations
transforms = ComposeSingle([
    # FunctionWrapperSingle(''),
    FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
    FunctionWrapperSingle(normalize_01)
])

# %% create dataset and dataloader
dataset = ObjectDetectionDatasetSingle(inputs=inputs,
                                       transform=transforms,
                                       use_cache=False,
                                       )

dataloader_prediction = DataLoader(dataset=dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_single)

# %% import experiment from neptune
api_key = get_api_key()  # get the personal api key
project_name = f'{params["OWNER"]}/{params["PROJECT"]}'
project = neptune.init(project_qualified_name=project_name, api_token=api_key)  # get project
experiment_id = params['EXPERIMENT']  # experiment id
experiment = project.get_experiments(id=experiment_id)[0]
parameters = experiment.get_parameters()
properties = experiment.get_properties()

# %% view dataset
from visual import DatasetViewerSingle

from torchvision.models.detection.transform import GeneralizedRCNNTransform

transform = GeneralizedRCNNTransform(min_size=int(parameters['MIN_SIZE']),
                                     max_size=int(parameters['MAX_SIZE']),
                                     image_mean=ast.literal_eval(parameters['IMG_MEAN']),
                                     image_std=ast.literal_eval(parameters['IMG_STD']))

datasetviewer = DatasetViewerSingle(dataset, rccn_transform=None)
datasetviewer.napari()
datasetviewer.viewer.window._qt_window.setGeometry(-1280, 0, 1280, 1400)
datasetviewer.viewer.window._qt_window.raise_()  # raise to foreground

# %% download model from neptune
if params['DOWNLOAD']:
    download_path = pathlib.Path(params['DOWNLOAD_PATH'])
    model_name = 'best_model.pt'
    if not (download_path / model_name).is_file():
        experiment.download_artifact(path=model_name, destination_dir=download_path)  # download model

    model_state_dict = torch.load(download_path / model_name)
else:
    checkpoint = torch.load(params['MODEL_DIR'])
    model_state_dict = checkpoint['hyper_parameters']['model'].state_dict()

# %% model init
from faster_RCNN import get_fasterRCNN_resnet

model = get_fasterRCNN_resnet(num_classes=int(parameters['CLASSES']),
                              backbone_name=parameters['BACKBONE'],
                              anchor_size=ast.literal_eval(parameters['ANCHOR_SIZE']),
                              aspect_ratios=ast.literal_eval(parameters['ASPECT_RATIOS']),
                              fpn=ast.literal_eval(parameters['FPN']),
                              min_size=int(parameters['MIN_SIZE']),
                              max_size=int(parameters['MAX_SIZE'])
                              )

# %% load weights
model.load_state_dict(model_state_dict)

# %% inference
model.eval()
for sample in dataloader_prediction:
    x, x_name = sample
    with torch.no_grad():
        pred = model(x)
        pred = {key: value.numpy() for key, value in pred[0].items()}
        name = pathlib.Path(x_name[0])
        torch.save(pred, pathlib.Path(params['PREDICTIONS_PATH']) / name.with_suffix('.pt'))

# %% create dataset
predictions = get_filenames_of_path(pathlib.Path(params['PREDICTIONS_PATH']))
predictions.sort()

transforms_prediction = ComposeDouble([
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

dataset_prediction = ObjectDetectionDataSet(inputs=inputs,
                                            targets=predictions,
                                            transform=transforms_prediction,
                                            use_cache=False)

# %% visualize predictions
from visual import DatasetViewer

color_mapping = {
    1: 'red',
    2: 'blue'
}

datasetviewer_prediction = DatasetViewer(dataset_prediction, color_mapping)
datasetviewer_prediction.napari()
datasetviewer_prediction.viewer.window._qt_window.setGeometry(-1280, 0, 1280, 1400)
datasetviewer_prediction.viewer.window._qt_window.raise_()  # raise to foreground


datasetviewer_prediction.gui_score_slider(datasetviewer_prediction.shape_layer)
# datasetviewer_prediction.gui_nms_slider(datasetviewer_prediction.shape_layer)
datasetviewer_prediction.gui_text_properties(datasetviewer_prediction.shape_layer)

# TODO: There seems to be something wrong with the color mapping when using the nms slider
