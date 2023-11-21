"""
Global consistent quantization loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IC_Loss(nn.Module):
	"""Instance contrastive learning loss"""
	def __init__(self, batch_size, tau_ic, mask_neg):
		super(IC_Loss, self).__init__()
		self.batch_size = batch_size
		self.tau_ic = tau_ic
		self.mask_neg = mask_neg
		self.sim_metric = torch.nn.CosineSimilarity(dim=-1)
		self.CE_Loss = torch.nn.CrossEntropyLoss()

	def forward(self, Xa, Xb, labels):
		XaXb = torch.cat([Xa, Xb], dim=0)
		Sim = self.sim_metric(XaXb.unsqueeze(1), XaXb.unsqueeze(0))
		Diag_a = torch.diag(Sim, self.batch_size)
		Diag_b = torch.diag(Sim, -self.batch_size)
		Pos = torch.cat([Diag_a, Diag_b]).view(2*self.batch_size, 1)
		Neg = Sim[self.mask_neg].view(2*self.batch_size, -1)

		logits = torch.cat((Pos, Neg), dim=1)
		logits *= 1/self.tau_ic
		loss = self.CE_Loss(logits, labels)

		return loss, Neg


class KL_Loss(nn.Module):
	"""Kullback-Leibler divergence loss"""
	def __init__(self, temperature):
		super(KL_Loss, self).__init__()
		self.T = temperature

	def forward(self, logits_p, logits_q):
		# Symmetric KL Divergence
		logits_p_1 = F.log_softmax(logits_p/self.T, dim=1)
		logits_q_1 = F.softmax(logits_q/self.T, dim=1)
		loss_1 = nn.KLDivLoss(reduction='batchmean')(logits_p_1, logits_q_1)

		logits_q_2 = F.log_softmax(logits_q/self.T, dim=1)
		logits_p_2 = F.softmax(logits_p/self.T, dim=1)
		loss_2 = nn.KLDivLoss(reduction='batchmean')(logits_q_2, logits_p_2)

		return loss_1*0.5+loss_2*0.5
	
class CL_Loss(nn.Module):
	"""Sub-embedding Classification Loss"""
	def __init__(self, batch_size):
		super(CL_Loss, self).__init__()
		self.batch_size = batch_size
		self.NLL_Loss = nn.NLLLoss()
		
	def forward(self, logits_a, logits_b, labels):
		#import ipdb; ipdb.set_trace()
		labels_repeat = labels.repeat(self.batch_size)
		loss_a = self.NLL_Loss(logits_a, labels_repeat)
		loss_b = self.NLL_Loss(logits_b, labels_repeat)
		loss = loss_a + loss_b 
		return loss 

class CQC_Loss(nn.Module):
	"""Cross quantization loss from SPQ"""
	def __init__(self, batch_size, tau_cqc, mask_neg):
		super(CQC_Loss, self).__init__()
		self.batch_size = batch_size
		self.tau_cqc = tau_cqc
		#self.device = device
		self.COSSIM = nn.CosineSimilarity(dim=-1)
		self.CE = nn.CrossEntropyLoss(reduction="sum")
		self.mask = mask_neg
		#self.get_corr_mask = self._get_correlated_mask().type(T.bool)

	def forward(self, Xa, Xb, Za, Zb, labels):

		XaZb = torch.cat([Xa, Zb], dim=0)
		XbZa = torch.cat([Xb, Za], dim=0)

		Cossim_ab = self.COSSIM(XaZb.unsqueeze(1), XaZb.unsqueeze(0))
		Rab = torch.diag(Cossim_ab, self.batch_size)
		Lab = torch.diag(Cossim_ab, -self.batch_size)
		Pos_ab = torch.cat([Rab, Lab]).view(2 * self.batch_size, 1)
		Neg_ab = Cossim_ab[self.mask].view(2 * self.batch_size, -1)

		Cossim_ba = self.COSSIM(XbZa.unsqueeze(1), XbZa.unsqueeze(0))
		Rba = torch.diag(Cossim_ba, self.batch_size)
		Lba = torch.diag(Cossim_ba, -self.batch_size)    
		Pos_ba = torch.cat([Rba, Lba]).view(2 * self.batch_size, 1)
		Neg_ba = Cossim_ba[self.mask].view(2 * self.batch_size, -1)


		logits_ab = torch.cat((Pos_ab, Neg_ab), dim=1)
		logits_ab /= self.tau_cqc

		logits_ba = torch.cat((Pos_ba, Neg_ba), dim=1)
		logits_ba /= self.tau_cqc

		#labels = torch.zeros(2 * self.batch_size).to(self.device).long()
		
		loss = self.CE(logits_ab, labels) + self.CE(logits_ba, labels)
		return loss / (2 * self.batch_size)