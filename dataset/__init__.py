from .cityscapes import Cityscapes
from .nyu import NYUv2, NYUv2Depth
from utils import ext_transforms
import torch


""" Segmentation """

def get_dataloader(args):

    if args.dataset.lower() in ['nyuv2']:
        train_loader = torch.utils.data.DataLoader(
            NYUv2(args.data_root, split='train',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(args.base_size),
                            ext_transforms.ExtRandomCrop(args.crop_size, pad_if_needed=True),
                            ext_transforms.ExtRandomHorizontalFlip(),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            NYUv2(args.data_root, split='test',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(args.base_size),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    elif args.dataset.lower() == 'cityscapes':
        train_loader = torch.utils.data.DataLoader(
            Cityscapes(args.data_root, split='train',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(args.base_size),
                            ext_transforms.ExtRandomCrop(args.crop_size, pad_if_needed=True),
                            ext_transforms.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                            ext_transforms.ExtRandomHorizontalFlip(),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            Cityscapes(args.data_root, split='val',
                        transform=ext_transforms.ExtCompose([
                            ext_transforms.ExtResize(args.base_size),
                            ext_transforms.ExtToTensor(),
                            ext_transforms.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])),
            batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    return train_loader, test_loader
