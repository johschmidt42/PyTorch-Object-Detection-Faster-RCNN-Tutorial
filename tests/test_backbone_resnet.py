import pytest
import torch

from pytorch_faster_rcnn_tutorial.backbone_resnet import (
    BackboneWithFPN,
    ResNetBackbones,
    get_resnet_backbone,
    get_resnet_fpn_backbone,
)


@pytest.mark.parametrize(argnames="backbone_name", argvalues=list(ResNetBackbones))
def test_get_resnet_backbone(backbone_name):
    backbone = get_resnet_backbone(backbone_name=backbone_name)
    assert type(backbone) == torch.nn.Sequential


def test_get_resnet_backbone_error():
    with pytest.raises(ValueError):
        get_resnet_backbone(backbone_name="resnet-999")


@pytest.mark.parametrize(argnames="backbone_name", argvalues=list(ResNetBackbones))
def test_get_resnet_fpn_backbone(backbone_name):
    backbone_fpn = get_resnet_fpn_backbone(
        backbone_name=backbone_name, pretrained=True, trainable_layers=5
    )
    assert type(backbone_fpn) == BackboneWithFPN
    assert backbone_fpn.out_channels == 256
