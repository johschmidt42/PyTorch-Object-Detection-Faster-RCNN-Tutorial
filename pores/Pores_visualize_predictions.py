import numpy as np
import pathlib
import torch

input_file = pathlib.Path(
    r'C:\Users\johan\Desktop\JuanPrada\Clone6 +_- dTAG - Images\rescaled\Rep1 + WT\CL6 dTag -2.tif')
prediction_file = pathlib.Path(
    r'C:\Users\johan\Desktop\JuanPrada\Clone6 +_- dTAG - Images\results\Rep1 + WT\CL6 dTag -2.pt')

from skimage.io import imread

img = imread(input_file)
tar = torch.load(prediction_file)

boxes1_raw = tar['boxes'][tar['labels'] == 1]
boxes2_raw = tar['boxes'][tar['labels'] == 2]

from visual import make_bbox_napari

boxes1_napari = [make_bbox_napari(box) for box in boxes1_raw]
boxes2_napari = [make_bbox_napari(box) for box in boxes2_raw]

from utils import enable_gui_qt
import napari

enable_gui_qt()

viewer = napari.Viewer()
img_viewer = viewer.add_image(img)
shape1 = viewer.add_shapes(boxes1_napari, edge_color='red', edge_width=1, face_color='transparent')
shape2 = viewer.add_shapes(boxes2_napari, edge_color='green', edge_width=1, face_color='transparent')
