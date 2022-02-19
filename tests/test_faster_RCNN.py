import pytest
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from pytorch_faster_rcnn_tutorial.backbone_resnet import ResNetBackbones
from pytorch_faster_rcnn_tutorial.faster_RCNN import (
    get_anchor_generator,
    get_faster_rcnn_resnet,
    get_roi_pool,
)


def test_get_anchor_generator():
    anchor_generator = get_anchor_generator()
    assert type(anchor_generator) == AnchorGenerator


def test_get_roi_pool():
    roi_pooler = get_roi_pool()
    assert type(roi_pooler) == MultiScaleRoIAlign


@pytest.mark.parametrize(argnames="backbone_name", argvalues=list(ResNetBackbones))
@pytest.mark.parametrize(argnames="fpn", argvalues=[True, False])
def test_get_faster_rcnn_resnet(backbone_name, fpn):
    faster_rcnn = get_faster_rcnn_resnet(
        num_classes=2,
        backbone_name=backbone_name,
        anchor_size=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),),
        fpn=fpn,
    )
    assert type(faster_rcnn) == FasterRCNN
