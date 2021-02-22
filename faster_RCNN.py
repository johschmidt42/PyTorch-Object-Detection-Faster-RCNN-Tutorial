from itertools import chain
from typing import Tuple, List
from backbone_resnet import get_resnet_backbone, get_resnet_fpn_backbone

import pytorch_lightning as pl
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from utils import from_dict_to_BoundingBox


def get_anchor_generator(anchor_size: Tuple[tuple] = None, aspect_ratios: Tuple[tuple] = None):
    """Returns the anchor generator."""
    if anchor_size is None:
        anchor_size = ((16,), (32,), (64,), (128,))
    if aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_size)

    anchor_generator = AnchorGenerator(sizes=anchor_size,
                                       aspect_ratios=aspect_ratios)
    return anchor_generator


def get_roi_pool(featmap_names: List[str] = None, output_size: int = 7, sampling_ratio: int = 2):
    """Returns the ROI Pooling"""
    if featmap_names is None:
        # default for resnet with FPN
        featmap_names = ['0', '1', '2', '3']

    roi_pooler = MultiScaleRoIAlign(featmap_names=featmap_names,
                                    output_size=output_size,
                                    sampling_ratio=sampling_ratio)

    return roi_pooler


def get_fasterRCNN(backbone: torch.nn.Module,
                   anchor_generator: AnchorGenerator,
                   roi_pooler: MultiScaleRoIAlign,
                   num_classes: int,
                   image_mean: List[float] = [0.485, 0.456, 0.406],
                   image_std: List[float] = [0.229, 0.224, 0.225],
                   min_size: int = 512,
                   max_size: int = 1024,
                   **kwargs
                   ):
    """Returns the Faster-RCNN model. Default normalization: ImageNet"""
    model = FasterRCNN(backbone=backbone,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       num_classes=num_classes,
                       image_mean=image_mean,  # ImageNet
                       image_std=image_std,  # ImageNet
                       min_size=min_size,
                       max_size=max_size,
                       **kwargs
                       )
    model.num_classes = num_classes
    model.image_mean = image_mean
    model.image_std = image_std
    model.min_size = min_size
    model.max_size = max_size

    return model


def get_fasterRCNN_resnet(num_classes: int,
                          backbone_name: str,
                          anchor_size: List[float],
                          aspect_ratios: List[float],
                          fpn: bool = True,
                          min_size: int = 512,
                          max_size: int = 1024,
                          **kwargs
                          ):
    """Returns the Faster-RCNN model with resnet backbone with and without fpn."""

    # Backbone
    if fpn:
        backbone = get_resnet_fpn_backbone(backbone_name=backbone_name)
    else:
        backbone = get_resnet_backbone(backbone_name=backbone_name)

    # Anchors
    anchor_size = anchor_size
    aspect_ratios = aspect_ratios * len(anchor_size)
    anchor_generator = get_anchor_generator(anchor_size=anchor_size, aspect_ratios=aspect_ratios)

    # ROI Pool
    with torch.no_grad():
        backbone.eval()
        random_input = torch.rand(size=(1, 3, 512, 512))
        features = backbone(random_input)

    if isinstance(features, torch.Tensor):
        from collections import OrderedDict

        features = OrderedDict([('0', features)])

    featmap_names = [key for key in features.keys() if key.isnumeric()]

    roi_pool = get_roi_pool(featmap_names=featmap_names)

    # Model
    return get_fasterRCNN(backbone=backbone,
                          anchor_generator=anchor_generator,
                          roi_pooler=roi_pool,
                          num_classes=num_classes,
                          min_size=min_size,
                          max_size=max_size,
                          **kwargs)


class FasterRCNN_lightning(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 lr: float = 0.0001,
                 iou_threshold: float = 0.5
                 ):
        super().__init__()

        # Model
        self.model = model

        # Classes (background inclusive)
        self.num_classes = self.model.num_classes

        # Learning rate
        self.lr = lr

        # IoU threshold
        self.iou_threshold = iou_threshold

        # Transformation parameters
        self.mean = model.image_mean
        self.std = model.image_std
        self.min_size = model.min_size
        self.max_size = model.max_size

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Batch
        x, y, x_name, y_name = batch  # tuple unpacking

        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())

        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        # Batch
        x, y, x_name, y_name = batch

        # Inference
        preds = self.model(x)

        gt_boxes = [from_dict_to_BoundingBox(target, name=name, groundtruth=True) for target, name in zip(y, x_name)]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [from_dict_to_BoundingBox(pred, name=name, groundtruth=False) for pred, name in zip(preds, x_name)]
        pred_boxes = list(chain(*pred_boxes))

        return {'pred_boxes': pred_boxes, 'gt_boxes': gt_boxes}

    def validation_epoch_end(self, outs):
        gt_boxes = [out['gt_boxes'] for out in outs]
        gt_boxes = list(chain(*gt_boxes))
        pred_boxes = [out['pred_boxes'] for out in outs]
        pred_boxes = list(chain(*pred_boxes))

        from metrics.pascal_voc_evaluator import get_pascalvoc_metrics
        from metrics.enumerators import MethodAveragePrecision
        metric = get_pascalvoc_metrics(gt_boxes=gt_boxes,
                                       det_boxes=pred_boxes,
                                       iou_threshold=self.iou_threshold,
                                       method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                       generate_table=True)

        per_class, mAP = metric['per_class'], metric['mAP']
        self.log('Validation_mAP', mAP)

        for key, value in per_class.items():
            self.log(f'Validation_AP_{key}', value['AP'])

    def test_step(self, batch, batch_idx):
        # Batch
        x, y, x_name, y_name = batch

        # Inference
        preds = self.model(x)

        gt_boxes = [from_dict_to_BoundingBox(target, name=name, groundtruth=True) for target, name in zip(y, x_name)]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [from_dict_to_BoundingBox(pred, name=name, groundtruth=False) for pred, name in zip(preds, x_name)]
        pred_boxes = list(chain(*pred_boxes))

        return {'pred_boxes': pred_boxes, 'gt_boxes': gt_boxes}

    def test_epoch_end(self, outs):
        gt_boxes = [out['gt_boxes'] for out in outs]
        gt_boxes = list(chain(*gt_boxes))
        pred_boxes = [out['pred_boxes'] for out in outs]
        pred_boxes = list(chain(*pred_boxes))

        from metrics.pascal_voc_evaluator import get_pascalvoc_metrics
        from metrics.enumerators import MethodAveragePrecision
        metric = get_pascalvoc_metrics(gt_boxes=gt_boxes,
                                       det_boxes=pred_boxes,
                                       iou_threshold=self.iou_threshold,
                                       method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                       generate_table=True)

        per_class, mAP = metric['per_class'], metric['mAP']
        self.log('Test_mAP', mAP)

        for key, value in per_class.items():
            self.log(f'Test_AP_{key}', value['AP'])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=0.005)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='max',
                                                                  factor=0.75,
                                                                  patience=30,
                                                                  min_lr=0)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'Validation_mAP'}