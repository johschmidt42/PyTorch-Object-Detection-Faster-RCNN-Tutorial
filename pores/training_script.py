# imports
import os
import pathlib
import sys
import warnings

import albumentations as A
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers.neptune import NeptuneLogger
from torch.utils.data import DataLoader

from datasets import ObjectDetectionDataSet
from faster_RCNN import FasterRCNN_lightning
from faster_RCNN import get_fasterRCNN_resnet
from transformations import ComposeDouble, AlbumentationWrapper, Clip, FunctionWrapperDouble
from transformations import normalize_01
from utils import get_filenames_of_path, collate_double
from utils import log_model_neptune
from utils import log_packages_neptune

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# hyper-parameters
params = {'BATCH_SIZE': 6,
          'LR': 0.01,
          'PRECISION': 32,
          'CLASSES': 3,
          'SEED': 42,
          'PROJECT': 'Pores',
          'EXPERIMENT': 'pores',
          'OWNER': 'johschmidt42',
          'MAXEPOCHS': 500,
          'PATIENCE': 30,
          'BACKBONE': 'resnet18',
          'FPN': True,
          'ANCHOR_SIZE': ((16, 18, 20), (16, 18, 20), (16, 18, 20), (16, 18, 20)),
          'ASPECT_RATIOS': ((1.0,),),
          'MIN_SIZE': 256,
          'MAX_SIZE': 256,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5,
          'GPU': None,  # None -> training on cpu
          'DEFAULT_ROOT_DIR': pathlib.Path.home()
          }

# api key
assert os.environ['NEPTUNE'], 'Did you set your env var NEPTUNE?'
api_key = os.environ['NEPTUNE']

# root directory
root = pathlib.Path(r'pores/training_data')

# input and target files
inputs = get_filenames_of_path(root / 'input_224x224')
targets = get_filenames_of_path(root / 'target_224x224')

assert len(inputs) > 0, f'No files found in: {root / "input_224x224"}'
assert len(targets) > 0, f'No files found in: {root / "target_224x224"}'

inputs.sort()
targets.sort()

# training transformations and augmentations
transforms_training = ComposeDouble([
    Clip(),
    AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
    AlbumentationWrapper(albumentation=A.VerticalFlip(p=0.5)),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# validation transformations
transforms_validation_test = ComposeDouble([
    Clip(),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# random seed
seed_everything(params['SEED'])

# training validation test split
inputs_train, inputs_valid, inputs_test = inputs[:16], inputs[16:20], inputs[20:]
targets_train, targets_valid, targets_test = targets[:16], targets[16:20], targets[20:24]

# dataset training
dataset_train = ObjectDetectionDataSet(inputs=inputs_train,
                                       targets=targets_train,
                                       transform=transforms_training,
                                       use_cache=True,
                                       convert_to_format=None,
                                       mapping=None)

# dataset validation
dataset_valid = ObjectDetectionDataSet(inputs=inputs_valid,
                                       targets=targets_valid,
                                       transform=transforms_validation_test,
                                       use_cache=True,
                                       convert_to_format=None,
                                       mapping=None)

# dataset test
dataset_test = ObjectDetectionDataSet(inputs=inputs_test,
                                      targets=targets_test,
                                      transform=transforms_validation_test,
                                      use_cache=True,
                                      convert_to_format=None,
                                      mapping=None)

# dataloader training
dataloader_train = DataLoader(dataset=dataset_train,
                              batch_size=params['BATCH_SIZE'],
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_double)

# dataloader validation
dataloader_valid = DataLoader(dataset=dataset_valid,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              collate_fn=collate_double)

# dataloader test
dataloader_test = DataLoader(dataset=dataset_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_double)

# neptune logger
neptune_logger = NeptuneLogger(
    api_key=api_key,
    project_name=f'{params["OWNER"]}/{params["PROJECT"]}',
    experiment_name=params['EXPERIMENT'],
    params=params
)

assert neptune_logger.name  # http GET request to check if the project exists

# model init
model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
                              fpn=params['FPN'],
                              min_size=params['MIN_SIZE'],
                              max_size=params['MAX_SIZE'])

# task init
task = FasterRCNN_lightning(model=model, lr=params['LR'], iou_threshold=params['IOU_THRESHOLD'])

# callbacks
checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=params['PATIENCE'], mode='max')

# trainer init
trainer = Trainer(gpus=params['GPU'],
                  precision=params['PRECISION'],  # try 16 with enable_pl_optimizer=False
                  callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
                  default_root_dir=params['DEFAULT_ROOT_DIR'],  # where checkpoints are saved to
                  logger=neptune_logger,
                  log_every_n_steps=1,
                  num_sanity_val_steps=0,
                  benchmark=True,  # good if the input sizes do not change, will increase speed
                  )

# start training
trainer.max_epochs = params['MAXEPOCHS']
trainer.fit(task,
            train_dataloader=dataloader_train,
            val_dataloaders=dataloader_valid)

# start testing
trainer.test(ckpt_path='best', test_dataloaders=dataloader_test)

# log packages
log_packages_neptune(neptune_logger)

# log model
checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
log_model_neptune(checkpoint_path=checkpoint_path,
                  save_directory=pathlib.Path.home(),
                  name='best_model.pt',
                  neptune_logger=neptune_logger)

# stop logger
neptune_logger.experiment.stop()
print('Finished')