# %% Imports
import pathlib
import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader

from datasets import ObjectDetectionDataSet
from utils import get_filenames_of_path, collate_double
from transformations import ComposeDouble, AlbumentationWrapper, Clip, FunctionWrapperDouble
from transformations import normalize_01

# %% hyper-parameters
params = {'BATCH_SIZE': 6,
          'LR': 0.01,
          'PRECISION': 32,
          'SAMPLE': False,
          'CLASSES': 3,
          'SEED': 42,
          'PROJECT': 'Pores',
          'EXPERIMENT': 'pores',
          'MAXEPOCHS': 500,
          'BACKBONE': 'resnet18',
          'FPN': True,
          'ANCHOR_SIZE': ((16, 18, 20), (16, 18, 20), (16, 18, 20), (16, 18, 20)),
          'ASPECT_RATIOS': ((1.0,),),
          'MIN_SIZE': 256,
          'MAX_SIZE': 256,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }

# %% root directory
root = pathlib.Path(r'C:\Users\johan\Desktop\Johannes\Pores')

# %% input and target files
inputs = get_filenames_of_path(root / 'input_224x224_bb')
targets = get_filenames_of_path(root / 'target_224x224_bb_camilla')

inputs.sort()
targets.sort()

# %% training transformations and augmentations
transforms_training = ComposeDouble([
    Clip(),
    AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
    AlbumentationWrapper(albumentation=A.VerticalFlip(p=0.5)),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# %% validation transformations
transforms_validation = ComposeDouble([
    Clip(),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# %% random seed
from pytorch_lightning import seed_everything

seed_everything(params['SEED'])

# %% training validation test split
if params['SAMPLE']:
    inputs_train, inputs_valid = inputs[:4], inputs[4:8]
    targets_train, targets_valid = targets[:4], targets[4:8]
else:
    inputs_train, inputs_valid = inputs[:-6], inputs[-6:]
    targets_train, targets_valid = targets[:-6], targets[-6:]

# %% dataset training
dataset_train = ObjectDetectionDataSet(inputs=inputs_train,
                                       targets=targets_train,
                                       transform=transforms_training,
                                       use_cache=True,
                                       convert_to_format=None,
                                       mapping=None)

# %% dataset validation
dataset_valid = ObjectDetectionDataSet(inputs=inputs_valid,
                                       targets=targets_valid,
                                       transform=transforms_validation,
                                       use_cache=True,
                                       convert_to_format=None,
                                       mapping=None)

# %% dataloader training
dataloader_train = DataLoader(dataset=dataset_train,
                              batch_size=params['BATCH_SIZE'],
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_double)

# %% dataloader validation
dataloader_valid = DataLoader(dataset=dataset_valid,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              collate_fn=collate_double)

# %% neptune logger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from api_key_neptune import get_api_key

api_key = get_api_key()

neptune_logger = NeptuneLogger(
    api_key=api_key,
    project_name=f'johschmidt42/{params["PROJECT"]}',
    experiment_name=params['EXPERIMENT'],
    params=params
)

# %% model init
from faster_RCNN import get_fasterRCNN_resnet

model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
                              fpn=params['FPN'],
                              min_size=params['MIN_SIZE'],
                              max_size=params['MAX_SIZE'])

# %% task init
from faster_RCNN import FasterRCNN_lightning

task = FasterRCNN_lightning(model=model, lr=params['LR'], iou_threshold=params['IOU_THRESHOLD'])

# %% callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=30, mode='max')


# %% trainer init
from pytorch_lightning import Trainer
trainer = Trainer(gpus=1,
                  precision=params['PRECISION'],  # try 16 with enable_pl_optimizer=False
                  callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
                  default_root_dir=r"C:\Users\johan\Desktop\Johannes\Experiments",  # where checkpoints are saved to
                  logger=neptune_logger,
                  log_every_n_steps=1,
                  num_sanity_val_steps=0,
                  enable_pl_optimizer=False,  # False seems to be necessary for half precision
                  benchmark=True,  # good if the input sizes do not change, will increase speed
                  )


# %% start training
trainer.max_epochs = params['MAXEPOCHS']
trainer.fit(task,
            train_dataloader=dataloader_train,
            val_dataloaders=dataloader_valid)

# %% start testing
"""
Start testing here
trainer.test(ckpt_path='best', test_dataloaders='dataloader_test')
"""

# %% log packages
from utils import log_packages_neptune

log_packages_neptune(neptune_logger)

# %% log model
from utils import log_model_neptune

checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
log_model_neptune(checkpoint_path=checkpoint_path,
                  save_directory=pathlib.Path.home(),
                  name='best_model.pt',
                  neptune_logger=neptune_logger)
