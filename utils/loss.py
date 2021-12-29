import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['ScaleInvariantLoss', 'FocalLoss', 'soft_cross_entropy',
			'criterion_mutual', 'mutual_calibration']

# ---------------------------------------------
# 				Segmentation loss
# ---------------------------------------------
class ScaleInvariantLoss(nn.Module):
    """This criterion is used in depth prediction task.
    **Parameters:**
        - **la** (int, optional): Default value is 0.5. No need to change.
        - **ignore_index** (int, optional): Value to ignore.
    **Shape:**
        - **inputs**: $(N, H, W)$.
        - **targets**: $(N, H, W)$.
        - **output**: scalar.
    """
    def __init__(self, la=0.5, ignore_index=255):
        super(ScaleInvariantLoss, self).__init__()
        self.la = la
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        size = inputs.size()
        if len(size) > 2:
            inputs = inputs.view(size[0], -1)
            targets = targets.view(size[0], -1)
        
        inv_mask = targets.eq(self.ignore_index)
        nums = (1-inv_mask.float()).sum(1)

        log_d = torch.log(inputs) - torch.log(targets)
        log_d[inv_mask] = 0

        loss = torch.div(torch.pow(log_d, 2).sum(1), nums) - \
            self.la * torch.pow(torch.div(log_d.sum(1), nums), 2)

        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def kldiv(logits, targets, reduction='batchmean'):
    p = F.log_softmax(logits, dim=1)
    q = F.softmax(targets, dim=1)
    return F.kl_div(p, q, reduction=reduction)

def soft_cross_entropy(logits, target, T=1.0, size_average=True, target_is_prob=False):
    """ Cross Entropy for soft targets
    
    **Parameters:**
        - **logits** (Tensor): logits score (e.g. outputs of fc layer)
        - **targets** (Tensor): logits of soft targets
        - **T** (float): temperature　of distill
        - **size_average**: average the outputs
        - **target_is_prob**: set True if target is already a probability.
    """
    if target_is_prob:
        p_target = target
    else:
        p_target = F.softmax(target/T, dim=1)
    
    logp_pred = F.log_softmax(logits/T, dim=1)
    # F.kl_div(logp_pred, p_target, reduction='batchmean')*T*T
    ce = torch.sum(-p_target * logp_pred, dim=1)
    if size_average:
        return ce.mean() * T * T
    else:
        return ce * T * T


def criterion_aux(out_aux, out_s, out_t, target, alpha=0.5, T=6.):
	loss_kl = F.kl_div(F.log_softmax(out_aux, dim=1),
					F.softmax(out_t.detach()/T, dim=1), reduction='mean') * T * T
	loss_ce = F.cross_entropy(out_aux, target, ignore_index=255)
	loss = loss_ce + alpha * loss_kl
	return loss 

def criterion_mutual(out_a, out_s, out_t, target, alpha=0.5, T=10.):
	loss_kl_a = F.kl_div(F.log_softmax(out_a, dim=1),
					F.softmax(out_t.detach()/T, dim=1), reduction='mean') * T * T
	loss_kl_s = F.kl_div(F.log_softmax(out_s, dim=1),
					F.softmax(out_t.detach()/T, dim=1), reduction='mean') * T * T
	loss_ce_s = F.cross_entropy(out_s, target, ignore_index=255)
	loss_ce_a = F.cross_entropy(out_a, target, ignore_index=255)
	loss_aux = loss_ce_a + alpha * loss_kl_a
	loss_skl = loss_ce_s #+ alpha * loss_kl_s
	return loss_skl, loss_aux

def kl_loss(out, out_t, T=6.):
	return F.kl_div(F.log_softmax(out, dim=1),
					F.softmax(out_t/T, dim=1), reduction='mean') * T * T

# Mutual Label Calibration
def mutual_calibration(out_at, out_as, out_t, out_s, target, alpha=0.5, T=10.):

    ''' calibration for student '''

    pred_at = torch.argmax(out_at, dim=1)
    pred_as = torch.argmax(out_as, dim=1)
    pred_s = torch.argmax(out_s, dim=1)
    pred_t = torch.argmax(out_t, dim=1)
    mp_at = pred_at == target   # postive mask
    mp_s = pred_s == target     # negative mask
    mr_s = mp_s == mp_at    # right mask 
    # mc_s = not mr_s     # calibration mask

    mr_s = mr_s.int()
    mc_s = 1 - mr_s # calibration mask

    # ent_as = entropy_map(out_as)
    # ent_at = entropy_map(out_at)
    # ent_t = entropy_map(out_t)

    # entmap_stack = torch.stack((ent_as, ent_at, ent_t), dim=1)
    # entmap_mask = torch.argmax(entmap_stack, dim=1)

    # out_stack = torch.stack((out_as, out_at, out_t), dim=2)
    #TODO:
    # calibr_out = out_stack[:, :, entmap_mask, :, :]
    
    # index_mask = torch.ones_like(target) * 255
    # calibr_tar = torch.where(mr_s, target, index_mask)
    loss_ce_s = F.cross_entropy(out_s, target, ignore_index=255)

    out_s2 = torch.mul(out_s, mc_s.unsqueeze(1)) + mr_s.unsqueeze(1)
    out_t2 = torch.mul(out_t, mc_s.unsqueeze(1)) + mr_s.unsqueeze(1)
    loss_kl_s = kl_loss(out_s2, out_t2)

    loss_s = loss_ce_s + alpha * loss_kl_s

    ''' calibration for aux '''

    loss_ce_a = F.cross_entropy(out_at, target, ignore_index=255)
    out_a2 = torch.mul(out_at, mr_s.unsqueeze(1)) + mc_s.unsqueeze(1)
    out_t2 = torch.mul(out_t, mr_s.unsqueeze(1)) + mc_s.unsqueeze(1)
    loss_kl_a = kl_loss(out_a2, out_t2)

    loss_a = loss_ce_a + alpha * loss_kl_a

    return loss_s, loss_a


def entropy_map(out_map, normalize=True):
	eps = 1e-6
	pred = F.softmax(out_map, dim=1)
	map = -torch.sum(pred.mul(torch.log(pred)), dim=1)

	if normalize:
		norm = torch.norm(map, dim=(1,2), keepdim=False)
		map = torch.div(map, norm+eps)
	return map


def get_mask(out_a, out_b, target):
	label_a = torch.argmax(out_a, dim=1)
	label_b = torch.argmax(out_b, dim=1)
	pos_a = label_a == target
	pos_b = label_b == target
	
