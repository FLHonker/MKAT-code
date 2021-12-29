from __future__ import print_function
import os
import argparse
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import network
from dataset import get_dataloader
from utils.stream_metrics import StreamSegMetrics
from utils.visualizer import VisdomPlotter
from utils.lr_scheduler import PolyLR
from utils.loss import mutual_calibration

vp = None

def train(args, teacher, student, device, train_loader, encoders, optimizer, lr_scheduler, aux_head, optimizer_aux, scheduler_aux, epoch):
    teacher.eval()
    student.train()
    seg_metrics = StreamSegMetrics(args.num_classes)
    loss = 0

    tqdm_epcoh = tqdm(train_loader, ncols=100)
    for batch_idx, (data, target) in enumerate(tqdm_epcoh):
        data, target = data.to(device), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        optimizer_aux.zero_grad()
        with torch.no_grad():
            fm_t, out_t = teacher(data)
            out_t = out_t.detach()
            fm_t = [f.detach() for f in fm_t.values()]
        
        fm_s, out_s = student(data)
        fm_s = [f for f in fm_s.values()]
        
        _, _, feat_s, feat_t = encoders(fm_s[-2], fm_t[-2])
        out_aux_t = aux_head(feat_t, (target.size(1), target.size(2)))
        # loss_kl, loss_aux = mutual_calibration(out_aux_t, out_s, out_t, target, alpha=args.alpha, T=args.T)
        # discriminate feat_s
        with torch.no_grad():
            out_aux_s = aux_head(feat_s, (target.size(1), target.size(2)))
        loss_kl, loss_aux = mutual_calibration(out_aux_t, out_aux_s, out_t, out_s, target, alpha=args.alpha, T=args.T)

        seg_metrics.update(out_aux_t.max(1)[1].detach().cpu().numpy().astype('uint8'), target.detach().cpu().numpy().astype('uint8'))
        # update aux_head
        loss_aux.backward()
        optimizer_aux.step()
        scheduler_aux.step()
        
        loss_kd = F.l1_loss(feat_s, feat_t.detach())
        # loss_kd = F.l1_loss(out_aux_s, out_aux_t.detach())
        loss = loss_kl + args.beta * loss_kd 
        # update student and encoders
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        tqdm_epcoh.set_description('Epoch:[{}], Loss: {:.4f}'.format(epoch, loss.item()))
    
    results = seg_metrics.get_results()
    print('\nTrain set: Acc= {:.4f}, mIoU= {:.4f}, mAP= {:.4f} \n'.format(results['Overall Acc'], results['Mean IoU'], results['Mean Acc']))
    return loss.item()

def test(args, model, device, test_loader):
    model.eval()
    model.to(device)
    seg_metrics = StreamSegMetrics(args.num_classes)

    with torch.no_grad():
        tqdm_epcoh = tqdm(test_loader, ncols=100)
        tqdm_epcoh.set_description('test')
        for batch_idx, (data, target) in enumerate(tqdm_epcoh):
            data, target = data.to(device), target.to(device, dtype=torch.long)
            _, output = model(data)
            seg_metrics.update(output.max(1)[1].detach().cpu().numpy().astype('uint8'), target.detach().cpu().numpy().astype('uint8'))

    results = seg_metrics.get_results()

    print('\nTest set: Acc= {:.4f}, mIoU= {:.4f}, mAP= {:.4f} \n'.format(results['Overall Acc'], results['Mean IoU'], results['Mean Acc']))
    
    return results

    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MKAT-Seg')
    parser.add_argument('--num_classes', type=int, default=19, help='class num')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 90)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--data_root', type=str, default=None, required=True)
    parser.add_argument('--dataset', type=str, default='cityscapes', 
                        help='dataset name')
    parser.add_argument('--base_size', type=int, default=1024)
    parser.add_argument('--crop_size', type=int, default=768)
    parser.add_argument('--teacher', type=str, default='',
                        help='teacher model name (default: deeplabv3_resnet50)')
    parser.add_argument('--student', type=str, default='',
                        help='student model name (default: deeplabv3_mobilenet)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu_ids', type=str, default='', help='CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--t_ckpt', type=str, default=None, required=True)
    parser.add_argument('--s_ckpt', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--power', type=float, default=0.9, help='power of PolyLR')
    parser.add_argument('--T', type=float, help='temperature for KL')
    parser.add_argument('--alpha', type=float, help='alpha for KL loss')
    parser.add_argument('--beta', type=float, help='beta for KD loss')
    parser.add_argument('--kn_nz', type=int, default=256, help='channels of knowledge latent features')
    parser.add_argument('--head_ch', type=int, default=512, help='input channels of aux_header')
    parser.add_argument('--kn_list', type=int, nargs='+', help='knowledge used for aggregation')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args)

    global vp
    vp = VisdomPlotter('8097', 'KD-seg-%s'%args.dataset)
    vp.add_table('settings-{}: {}->{}'.format(args.dataset, args.teacher, args.student), vars(args))

    train_loader, test_loader = get_dataloader(args)
    args.model = args.student
    student = network.get_model(args)
    student = torch.nn.DataParallel(student).to(device)
    args.model = args.teacher
    teacher = network.get_model(args)
    teacher = torch.nn.DataParallel(teacher).to(device)
    teacher.module.load_state_dict(torch.load(args.t_ckpt))
    print('loaded teacher model from %s.' % args.t_ckpt)
    teacher.eval()

    data = torch.randn((args.batch_size, 3, args.crop_size, args.crop_size)).to(device)
    with torch.no_grad():
        g_t, _ = teacher(data)
        g_s, _ = student(data)
        g_t = [f.shape for f in list(g_t.values())]
        g_s = [f.shape for f in list(g_s.values())]

    aux_head = network.DeepLab_AUX(agg_ch=args.kn_nz*len(args.kn_list), in_channels=args.head_ch, num_classes=args.num_classes).to(device)
    encoders = network.MKAT_F(s_shape=g_s[3], t_shape=g_t[3], nz=args.kn_nz, kn_list=args.kn_list).to(device)

    if args.s_ckpt is not None:
        checkpoint = torch.load(args.s_ckpt)
        student.module.load_state_dict(checkpoint['model'])
        aux_head.load_state_dict(checkpoint['aux_head'])
        encoders.load_state_dict(checkpoint['encoders'])
        print('student loaded from %s' % args.s_ckpt)

    trainable_params = [{'params': filter(lambda p:p.requires_grad, student.module.get_backbone_params()), 'lr': 0.1*args.lr},
                        {'params': filter(lambda p:p.requires_grad, student.module.get_decoder_params()), 'lr': args.lr} ]
    # trainable_params = []
    trainable_params.append({'params': encoders.enc_s_params(), 'lr': 0.1*args.lr})
    trainable_params.append({'params': encoders.enc_t_params(), 'lr': 0.1*args.lr})

    optimizer = optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer_aux = optim.SGD(aux_head.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    best_mIoU = 0
    best_results = None
    loss_list = []
    miou_list = []

    scheduler = PolyLR(optimizer=optimizer, max_iters=args.epochs * len(train_loader), power=args.power) 
    scheduler_aux = PolyLR(optimizer=optimizer_aux, max_iters=args.epochs * len(train_loader), power=args.power) 

    if args.test_only:
        results = test(args, student, device, test_loader)
        return

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, teacher, student, device, train_loader, encoders, optimizer, scheduler, aux_head, optimizer_aux, scheduler_aux, epoch)
        results = test(args, student, device, test_loader)
        vp.add_scalar('train loss', epoch, train_loss)
        vp.add_scalar('mIoU', epoch, results['Mean IoU'])
        vp.add_scalar('lr_rate', epoch, optimizer.param_groups[0]['lr'])
        loss_list.append(train_loss)
        miou_list.append(results['Mean IoU'])

        if results['Mean IoU'] > best_mIoU:
            best_mIoU = results['Mean IoU']
            best_results = results
            save_model = {
                'aux_head': aux_head.state_dict(), 
                'encoders': encoders.state_dict(),
                'model': student.module.state_dict()
            }
            torch.save(save_model, "checkpoint/student/MKAT-{}-{}-{}.pt".format(args.dataset, args.teacher, args.student))
            print('Saving a best checkpoint...')
    
    print("-" * 25)
    print('\nBEST: Acc= {:.4f}, mIoU= {:.4f}, mAP= {:.4f} \n'.format(best_results['Overall Acc'], best_results['Mean IoU'], best_results['Mean Acc']))

if __name__ == '__main__':
    main()