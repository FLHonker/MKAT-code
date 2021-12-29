import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import math
from itertools import chain

from ._deeplab import ASPP

# -------------------------------------------
#          Multi-Knowledge Aggregation
# -------------------------------------------

class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args
	def forward(self, x):
		return x.view((x.size(0),)+self.shape)

class DeepLab_AUX(nn.Module):
	def __init__(self, agg_ch, in_channels, num_classes=11, aspp_dilate=[6, 12, 18]):
		super(DeepLab_AUX, self).__init__()
		# self.agg = nn.Sequential(
		# 	nn.Conv2d(agg_ch, in_channels, 1),
        #     nn.BatchNorm2d(in_channels),
		# 	nn.ReLU()
		# )

		self.aspp = ASPP(in_channels, aspp_dilate)
		self.head = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, num_classes, 1)
		)
    
	def forward(self, x, input_size):
		# ka = self.agg(x)
		out = self.aspp(x)
		out = self.head(out)
		out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
		return out


class MKAT_F(nn.Module):
	def __init__(self, s_shape, t_shape, nz=256, kn_list=range(5)):
		super(MKAT_F, self).__init__()
		self.nz = nz
		self.kn_list = kn_list
		self.num_k = len(kn_list)
		agg_ch = self.num_k * nz
		in_channels = 512

		def conv1x1(in_channels, out_channels, stride=1):
			return nn.Conv2d(
				in_channels, out_channels,
				kernel_size=1, padding=0,
				bias=False, stride=stride)

		at_shape = (s_shape[0], 1, s_shape[2] * s_shape[3])
		jac_shape = (s_shape[0], 3, 768, 768)
		af_shape = s_shape
		sa_shape = s_shape
		ca_shape = s_shape
		cm_shape = s_shape
		gm_shape = s_shape

		self.at_enc_s = nn.Sequential(
            conv1x1(at_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.af_enc_s = nn.Sequential(
			conv1x1(af_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.sa_enc_s = nn.Sequential(
			conv1x1(sa_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.ca_enc_s = nn.Sequential(
			conv1x1(ca_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.cm_enc_s = nn.Sequential(
			conv1x1(cm_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.gm_enc_s = nn.Sequential(
			conv1x1(gm_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.jac_enc_s = nn.Sequential(
			nn.Conv2d(jac_shape[1], nz//8, 5, 1),
			nn.BatchNorm2d(nz//8),
            nn.ReLU6(inplace=True),
			nn.Conv2d(nz//8, nz//4, 5, 3, 1),
			nn.BatchNorm2d(nz//4),
            nn.ReLU6(inplace=True),
			conv1x1(nz//4, nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		at_shape = (t_shape[0], 1, t_shape[2] * t_shape[3])
		jac_shape = (t_shape[0], 3, 768, 768)
		af_shape = t_shape
		sa_shape = t_shape
		ca_shape = t_shape
		cm_shape = t_shape
		gm_shape = t_shape
		self.at_enc_t = nn.Sequential(
            conv1x1(at_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.af_enc_t = nn.Sequential(
			conv1x1(af_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.sa_enc_t = nn.Sequential(
			conv1x1(sa_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.ca_enc_t = nn.Sequential(
			conv1x1(ca_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.cm_enc_t = nn.Sequential(
			conv1x1(cm_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.gm_enc_t = nn.Sequential(
			conv1x1(gm_shape[1], nz, 1), 
			nn.BatchNorm2d(nz),
			nn.ReLU()
		)

		self.agg_s = nn.Sequential(
			nn.Conv2d(agg_ch, in_channels, 1),
            nn.BatchNorm2d(in_channels),
			nn.ReLU()
		)

		self.agg_t = nn.Sequential(
			nn.Conv2d(agg_ch, in_channels, 1),
            nn.BatchNorm2d(in_channels),
			nn.ReLU()
		)

		self._initialize_weights()
		
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				# m.weight.data.normal_(0, math.sqrt(2. / n))
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	@staticmethod
	def adapt_wh(f_s, f_t):
		s_H, t_H = f_s.shape[2], f_t.shape[2]
		if s_H > t_H:
			f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
		elif s_H < t_H:
			f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
		else:
			pass
		return f_s, f_t

	def forward(self, f_s, f_t):
		f_s, f_t = self.adapt_wh(f_s, f_t)
		at_s, at_t = self.at(f_s), self.at(f_t)
		af_s, af_t = self.af(f_s), self.af(f_t)
		cm_s, cm_t = self.cm(f_s), self.cm(f_t)
		sa_s, sa_t = self.sa(f_s), self.sa(f_t)
		ca_s, ca_t = self.ca(f_s), self.ca(f_t)
		gm_s, gm_t = self.gram(f_s), self.gram(f_t)

		at_em_s, at_em_t = self.at_enc_s(at_s), self.at_enc_t(at_t)
		af_em_s, af_em_t = self.af_enc_s(af_s), self.af_enc_t(af_t)
		cm_em_s, cm_em_t = self.cm_enc_s(cm_s), self.cm_enc_t(cm_t)
		sa_em_s, sa_em_t = self.sa_enc_s(sa_s), self.sa_enc_t(sa_t)
		ca_em_s, ca_em_t = self.ca_enc_s(ca_s), self.ca_enc_t(ca_t)
		gm_em_s, gm_em_t = self.gm_enc_s(gm_s), self.gm_enc_t(gm_t)

		stack_s = [at_em_s, af_em_s, cm_em_s, sa_em_s, ca_em_s, gm_em_s]
		stack_t = [at_em_t, af_em_t, cm_em_t, sa_em_t, ca_em_t, gm_em_t]

		feat_stack_s = torch.cat([stack_s[i] for i in self.kn_list], dim=1) # 
		feat_stack_t = torch.cat([stack_t[i] for i in self.kn_list], dim=1) # 

		feat_s = self.agg_s(feat_stack_s)
		feat_t = self.agg_t(feat_stack_t)

		return feat_stack_s, feat_stack_t, feat_s, feat_t


	''' get params '''

	def enc_s_params(self):
		return chain(self.at_enc_s.parameters(), self.af_enc_s.parameters(), self.ca_enc_s.parameters(), 
					 self.sa_enc_s.parameters(), self.cm_enc_s.parameters(), self.gm_enc_s.parameters(), self.agg_s.parameters())

	def enc_t_params(self):
		return chain(self.at_enc_t.parameters(), self.af_enc_t.parameters(), self.ca_enc_t.parameters(), 
					 self.sa_enc_t.parameters(), self.cm_enc_t.parameters(), self.gm_enc_t.parameters(), self.agg_t.parameters())

	''' ---- 6/7 forms of knowledge ---- '''

	@staticmethod
	# attention 
	def at(f, p=2):
 		return F.normalize(f.pow(p).mean(1).view(f.size(0), -1)).reshape((f.size(0), 1, f.size(2), f.size(3)))

	@staticmethod
	# correlation matrix -- dual affinity
	def cm(f, P_order=2, gamma=0.4):
		f = F.normalize(f, p=2, dim=-1)
		f_trans = torch.transpose(f, 2, 3)
		sim_mat = torch.matmul(f_trans, torch.matmul(f, f_trans)) 	# (H*W)x[(W*H)x(H*W)] = (H*W)
		corr_mat1 = torch.zeros_like(sim_mat)

		for p in range(P_order+1):
			corr_mat1 += math.exp(-2*gamma) * (2*gamma)**p / \
						math.factorial(p) * torch.pow(sim_mat, p)
		corr_mat1 = torch.transpose(corr_mat1, 2, 3)

		sim_mat2  = torch.matmul(f, torch.matmul(f_trans, f)) 	# (W*H)x[(H*W)x(W*H)] = (W*H)
		corr_mat2 = torch.zeros_like(sim_mat2)

		for p in range(P_order+1):
			corr_mat2 += math.exp(-2*gamma) * (2*gamma)**p / \
						math.factorial(p) * torch.pow(sim_mat2, p)
		
		corr_mat = corr_mat1 + corr_mat2
		
		return corr_mat
	
	@staticmethod
	# grad cam
	def cam(out, f, target):
		target_out = torch.gather(out, 2, torch.unsqueeze(target, 1))
		grad_fm    = grad(outputs=target_out, inputs=f,
						grad_outputs=torch.ones_like(target_out),
						create_graph=True, retain_graph=True, only_inputs=True)[0]
		weights = F.adaptive_avg_pool2d(grad_fm, 1)
		cam = torch.sum(torch.mul(weights, grad_fm), dim=1, keepdim=True)
		cam = F.relu(cam)
		cam = cam.view(cam.size(0), -1)
		norm_cam = F.normalize(cam, p=2, dim=1)
		return norm_cam

	@staticmethod
	# grad norm
	def jacobian_grad(out, img, target):
		target_out = torch.gather(out, 2, torch.unsqueeze(target, 1))
		grad_ 	   = grad(outputs=target_out, inputs=img,
						grad_outputs=torch.ones_like(target_out),
						create_graph=True, retain_graph=True, only_inputs=True)[0]
		norm_grad  = F.normalize(grad_.view(grad_.size(0), -1), p=2, dim=1)
		return norm_grad

	@staticmethod
	# attention feature norm
	def af(f, eps=1e-6):
		fm_norm = torch.norm(f, dim=(2,3), keepdim=True)
		af 		= torch.div(f, fm_norm + eps)
		return af

	@staticmethod
	# spatial attention
	def sa(f, gamma=0.4):
		m_batchsize, C, height, width = f.size()
		proj_query = f.view(m_batchsize, -1, width*height).permute(0, 2, 1)
		proj_key = f.view(m_batchsize, -1, width*height)
		energy = torch.bmm(proj_query, proj_key)
		attention = F.softmax(energy, dim=-1)
		proj_value = f.view(m_batchsize, -1, width*height)

		out = torch.bmm(proj_value, attention.permute(0, 2, 1))
		out = out.view(m_batchsize, C, height, width)
		# out = gamma*out + f
		return out

	@staticmethod
	# channel attention
	def ca(f, gamma=0.4):
		m_batchsize, C, height, width = f.size()
		proj_query = f.view(m_batchsize, C, -1)
		proj_key = f.view(m_batchsize, C, -1).permute(0, 2, 1)
		energy = torch.bmm(proj_query, proj_key)
		energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
		attention = F.softmax(energy_new, dim=-1)
		proj_value = f.view(m_batchsize, C, -1)

		out = torch.bmm(attention, proj_value)
		out = out.view(m_batchsize, C, height, width)
		# out = gamma*out + f
		return out

	# gram matrix
	@staticmethod
	def gram(f):
		shape = f.shape
		f = f.view(f.size(0), f.size(1), -1)
		fm = F.normalize(f, dim=2)
		gram_matrix = torch.bmm(fm, fm.transpose(1, 2))
		trans_gram = torch.bmm(gram_matrix, f)
		trans_gram = trans_gram.view(shape)

		return trans_gram
  