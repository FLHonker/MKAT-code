import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url
from .mobilenet import MobileNetV2
from ._deeplab import DeepLabHead, DeepLabV3, FCN, FCNHead
# from torchvision.models.segmentation.fcn import FCNHead, FCN
from torchvision.models import resnet, mobilenet_v2

__all__ = ['DeepLab', 'FCNs', 'deeplabv3_mobilenet', 'deeplabv3_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet18']

def _segm_mobilenet(name, backbone_name, num_classes, aux, output_stride=8, pretrained_backbone=True):
    # if output_stride==8:
    #     aspp_dilate = [12, 24, 36]
    # else:
    #     aspp_dilate = [6, 12, 18]
    
    backbone = MobileNetV2(output_stride=output_stride, pretrained=pretrained_backbone)

    # return_layers = {'features': 'out'}
    return_layers = {'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }

    inplanes = 320
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, None)
    return model


def _segm_resnet(name, backbone_name, num_classes, aux, output_stride=16, pretrained_backbone=True):
    # if output_stride==8:
    #     replace_stride_with_dilation=[False, True, True]
    #     aspp_dilate = [12, 24, 36]
    # else:
    #     replace_stride_with_dilation=[False, False, True]
    #     aspp_dilate = [6, 12, 18]

    if backbone_name in ['resnet50', 'resnet101', 'resnet152']:
        replace_stride_with_dilation = [False, False, True]
        inplanes = 2048
    else:
        replace_stride_with_dilation = None
        inplanes = 512

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    return_layers = {'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }

    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, None)
    return model


def DeepLab(progress=True, backbone='resnet50', num_classes=19, dropout_p=0.0, aux_loss=None, **kwargs):
    if backbone == 'mobilenet':
        model = _segm_mobilenet("deeplab", "mobilenet_v2", num_classes=num_classes, aux=aux_loss, **kwargs)
    else:
        model = _segm_resnet("deeplab", backbone_name=backbone, num_classes=num_classes, aux=aux_loss, **kwargs)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout_p
        elif isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01
    return model


def FCNs(progress=True, backbone='resnet50', num_classes=19, dropout_p=0.0, aux_loss=None, **kwargs):
    if backbone == 'mobilenet':
        model = _segm_mobilenet("fcn", "mobilenet_v2", num_classes=num_classes, aux=aux_loss, **kwargs)
    else:
        model = _segm_resnet("fcn", backbone_name=backbone, num_classes=num_classes, aux=aux_loss, **kwargs)
    # for m in model.modules():
    #     if isinstance(m, nn.Dropout):
    #         m.p = dropout_p
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.momentum = 0.01
    return model