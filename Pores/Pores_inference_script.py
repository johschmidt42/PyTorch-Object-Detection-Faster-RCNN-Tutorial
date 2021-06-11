# %% imports

import ast
import pathlib
import os

import neptune
import torch

from api_key_neptune import get_api_key
from utils import get_filenames_of_path
from tqdm import tqdm

# device
device = torch.device('cuda')  # Device

# hyper-parameters
params = {'EXPERIMENT': 'POR-22',
          'DOWNLOAD': True,
          'DOWNLOAD_PATH': r'C:\Users\johan\Desktop\Johannes\Pores',
          'OWNER': 'johschmidt42',
          'PROJECT': 'Pores',
          }

api_key = get_api_key()  # get the personal api key
project_name = f'{params["OWNER"]}/{params["PROJECT"]}'
project = neptune.init(project_qualified_name=project_name, api_token=api_key)  # get project
experiment_id = params['EXPERIMENT']  # experiment id
experiment = project.get_experiments(id=experiment_id)[0]
parameters = experiment.get_parameters()
properties = experiment.get_properties()

# download model from neptune
if params['DOWNLOAD']:
    download_path = pathlib.Path(params['DOWNLOAD_PATH'])
    model_name = 'best_model.pt'
    if not (download_path / model_name).is_file():
        experiment.download_artifact(path=model_name, destination_dir=download_path)  # download model

    model_state_dict = torch.load(download_path / model_name)
else:
    checkpoint = torch.load(params['MODEL_DIR'])
    model_state_dict = checkpoint['hyper_parameters']['model'].state_dict()

# model init
from faster_RCNN import get_fasterRCNN_resnet

model = get_fasterRCNN_resnet(num_classes=int(parameters['CLASSES']),
                              backbone_name=parameters['BACKBONE'],
                              anchor_size=ast.literal_eval(parameters['ANCHOR_SIZE']),
                              aspect_ratios=ast.literal_eval(parameters['ASPECT_RATIOS']),
                              fpn=ast.literal_eval(parameters['FPN']),
                              min_size=int(parameters['MIN_SIZE']),
                              max_size=int(parameters['MAX_SIZE'])
                              )

# load weights
model.load_state_dict(model_state_dict)

# model to device
model = model.to(device)

# inputs
root_path = r'C:\Users\johan\Desktop\JuanPrada\Clone6 +_- dTAG - Images\rescaled'
folder_names = ['Rep1 + WT', 'Rep2', 'Rep3', 'Rescue Experiment March2021']

# inference batch processing
for folder_name in folder_names:
    inputs = get_filenames_of_path(pathlib.Path(root_path) / folder_name)
    inputs.sort()

    from skimage.io import imread

    for img_path in inputs:
        new_folder_path = pathlib.Path(root_path).parent / 'results' / folder_name
        new_name = (new_folder_path / img_path.stem).with_suffix('.pt')

        if not new_name.is_file():

            print('reading', img_path.name)
            img = imread(img_path)
            print('img_shape', img.shape)


            from skimage.util import view_as_windows

            block_view = view_as_windows(img, window_shape=(256, 256, 3), step=(128, 128, 3))  # 0-224, 128-352, 224-448

            from torchvision.transforms import functional as F

            # iterate over 3 dim
            import numpy as np

            patch_indices = [idx for idx in np.ndindex(block_view.shape[:3])]
            patch_boxes = []
            patch_labels = []
            patch_scores = []

            model.eval()

            print('inference')
            progressbar = tqdm(range(len(patch_indices)))
            for i, idx in zip(progressbar, patch_indices):
                patch = [F.to_tensor(block_view[idx]).to(device)]  # lazy preprocessing
                with torch.no_grad():
                    output = model(patch)  # Inference
                    output = output[0]  # Only one image at a time
                    patch_boxes.append(output['boxes'].cpu().numpy())  # to numpy
                    patch_labels.append(output['labels'].cpu().numpy())
                    patch_scores.append(output['scores'].cpu().numpy())  # to numpy

            from experiments.Pores.Pores_utils import remove_boxes_score, remove_boxes_edge
            from experiments.Pores.Pores_utils import translate_boxes_to_full_image

            print('post-processing')
            # Remove low score predictions
            patch_scores_keep = remove_boxes_score(patch_scores, score=0.6)

            patch_boxes_removed_score = [patch_box[mask] for patch_box, mask in zip(patch_boxes, patch_scores_keep)]
            patch_labels_removed_score = [patch_label[mask] for patch_label, mask in zip(patch_labels, patch_scores_keep)]
            patch_scores_removed_score = [patch_score[mask] for patch_score, mask in zip(patch_scores, patch_scores_keep)]

            # Remove edge boxes predictions
            patch_boxes_keep = remove_boxes_edge(patch_boxes_removed_score, pixels=14)

            patch_boxes_removed_edge = [patch_box[mask] if list(mask) else patch_box for patch_box, mask in
                                        zip(patch_boxes_removed_score, patch_boxes_keep)]
            patch_labels_removed_edge = [patch_label[mask] if list(mask) else patch_label for patch_label, mask in
                                         zip(patch_labels_removed_score, patch_boxes_keep)]
            patch_scores_removed_edge = [patch_score[mask] if list(mask) else patch_score for patch_score, mask in
                                         zip(patch_scores_removed_score, patch_boxes_keep)]

            # Translate boxes predictions to full image
            patch_boxes_translated = translate_boxes_to_full_image(patch_indices, patch_boxes_removed_edge)

            # Transform to single prediction format (concatenation)
            patch_boxes_all = np.concatenate(patch_boxes_translated)
            patch_labels_all = np.concatenate(patch_labels_removed_edge)
            patch_scores_all = np.concatenate(patch_scores_removed_edge)

            # Remove redundant boxes predictions with NMS
            from torchvision.ops import nms

            patch_boxes_keep_nms = nms(boxes=torch.from_numpy(patch_boxes_all), scores=torch.from_numpy(patch_scores_all),
                                       iou_threshold=0.5)
            # TODO: NMS should be performed per class, right?
            patch_boxes_all_nms = patch_boxes_all[patch_boxes_keep_nms]
            patch_labels_all_nms = patch_labels_all[patch_boxes_keep_nms]
            patch_scores_all_nms = patch_scores_all[patch_boxes_keep_nms]

            # %% save the predictions

            result = {
                'boxes': patch_boxes_all_nms,
                'labels': patch_labels_all_nms,
                'scores': patch_scores_all_nms
            }

            if not new_folder_path.is_dir():
                os.mkdir(new_folder_path)

            print('saving', result, new_name)
            torch.save(result, new_name)
