# %% Test the model
x, y, x_name, y_name = next(iter(dataloader_train))

task.model.eval()
with torch.no_grad():
    out = task.model(x)

# %% visualize dataset
color_mapping = {
    1: 'green',
    2: 'red'
}

from visual import DatasetViewer

b = DatasetViewer(dataset_valid, color_mapping, rgb=True)
b.napari()


# %% Inference

batch = dataset_valid[0]
x, y, x_name, y_name = batch['x'], batch['y'], batch['x_name'], batch['y_name']

with torch.no_grad():
    task.model.eval()
    output = task.model([x])[0]

boxes = output['boxes']
scores = output['scores']
labels = output['labels']


def threshold_scores(tensor, value):
    mask = torch.where(tensor > value)
    return mask


# Only take boxes with an objectiveness score of > 0.5
mask = threshold_scores(scores, 0.9)

scores_t = scores[mask]
boxes_t = boxes[mask]
labels_t = labels[mask]


def label_split(boxes, scores, labels):
    per_label = {}
    for label in labels.unique():
        mask = torch.where(labels == label.item())
        b = boxes[mask]
        s = scores[mask]
        per_label[label.item()] = {'boxes': b, 'scores': s}

    return per_label


d = label_split(boxes_t, scores_t, labels_t)

from annotator import make_bbox_napari

boxes_nap_1 = [make_bbox_napari(box) for box in d[1]['boxes']]
# boxes_nap_2 = [make_bbox_napari(box) for box in d[2]['boxes']]

import napari
import numpy as np
from utils import enable_gui_qt

enable_gui_qt()
viewer = napari.Viewer()

from transformations import re_normalize

img = re_normalize(x.numpy())
img = np.moveaxis(img, 0, -1)

viewer.add_image(img)

viewer.add_shapes(boxes_nap_1, face_color='transparent', edge_color='green', name='1')
# viewer.add_shapes(boxes_nap_2, face_color='transparent', edge_color='red', name='2')
