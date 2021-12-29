from .deeplabv3 import DeepLab, FCNs
from .mkat import MKAT_F, DeepLab_AUX

def get_model(args):
    name = args.model.lower().split('_')
    if 'deeplabv3' in name:
        return DeepLab(backbone=name[1], num_classes=args.num_classes, dropout_p=0.5, pretrained_backbone=True)
    elif 'fcn' in name:
        return FCNs(backbone=name[1], num_classes=args.num_classes, dropout_p=0.5, pretrained_backbone=True)
    else:
        return eval(args.model)(num_classes=args.num_classes, pretrained_backbone=True)
