"""
The unified learning objective for self-supervised consistent quantization
"""

from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict

from lib.loss_global import IC_Loss, KL_Loss, CL_Loss, CQC_Loss
from lib.loss_part import PN_Loss, ER_Loss


class SSCQ_Loss(torch.nn.Module):
	"""Unified Learning Objective for Self-Supervised Consistent Quantization"""
	def __init__(self, args, device, batch_size, N_books, L_word, tau_ic, tau_cc, tau_pn, fusion_type, N_top_part_neighbor, pn_use_pos):
		super(SSCQ_Loss, self).__init__()
		self.device = device
		self.args = args
		self.batch_size = batch_size
		self.N_books = N_books
		self.L_word = L_word
		self.tau_ic = tau_ic
		self.tau_cc = tau_cc
		self.tau_pn = tau_pn
		self.tau_cqc = 0.5 # from paper
		self.fusion_type = fusion_type
		self.N_top_part_neighbor = N_top_part_neighbor
		self.pn_use_pos = pn_use_pos

		self.sim_metric = torch.nn.CosineSimilarity(dim=-1)
		self.mask_neg = self.generate_neg_mask().type(torch.bool) # get mask for negative samples
		self.IC_Loss = IC_Loss(self.batch_size, self.tau_ic, self.mask_neg).to(self.device)
		self.KL_Loss = KL_Loss(self.tau_cc).to(self.device)
		self.PN_Loss = PN_Loss(self.batch_size, self.L_word, self.N_books, self.tau_pn, self.N_top_part_neighbor, self.mask_neg, self.pn_use_pos).to(self.device)
		self.ER_Loss = ER_Loss(self.batch_size, self.N_books).to(self.device)
		self.CQC_Loss = CQC_Loss(self.batch_size, self.tau_cqc, self.mask_neg).to(self.device)
		self.CL_Loss = CL_Loss(self.batch_size)

		self.time_dict = defaultdict(list) 
	def generate_neg_mask(self):
		"""Get mask for negative samples"""
		diag = np.eye(2 * self.batch_size)
		l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
		l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
		mask = torch.from_numpy((diag + l1 + l2))
		mask = (1 - mask).type(torch.bool)
		return mask.to(self.device)

	def forward(self, Fa, Fb, Za, Zb, Pa, Pb, Ca, Cb):
	
		labels = torch.zeros(2*self.batch_size).to(self.device).long()
		labels_cls = torch.arange(self.N_books).to(self.device).long()

		#### Global consistent quantization
		# Quantized representation instance contrastive loss
		time1 = time.time()
		loss_icz, Neg_z = self.IC_Loss(Za, Zb, labels)
		time2 = time.time()-time1
		if self.args.get_timing:
			self.time_dict['icz'].append(time2)
			

		# Embedding representation instance contrastive loss
		time1 = time.time()
		loss_icf, Neg_f = self.IC_Loss(Fa, Fb, labels)
		time2 = time.time()-time1 
		if self.args.get_timing:
			self.time_dict['icf'].append(time2)
		
		# Codebook classification loss
		if self.args.W_codebook_classfiy > 0:
			loss_cl = self.CL_Loss(Ca, Cb, labels_cls)
		else:
			loss_cl = torch.tensor(0)

		# Cross quantization loss 
		if self.args.W_cqc > 0:
			loss_cqc = self.CQC_Loss(Fa, Fb, Za, Zb, labels)
		else:
			loss_cqc = torch.tensor(0)
		# Fused representation consistent contrastive regularization
		time1 = time.time()
		if self.fusion_type == 'cross': # tau_cc=2.0, lambda_cc=40.0
			logits_p_z, logits_q_z = Neg_z[:self.batch_size], Neg_z[self.batch_size:]
			logits_p_f, logits_q_f = Neg_f[:self.batch_size], Neg_f[self.batch_size:]
			loss_cc = self.KL_Loss(logits_p_z, logits_q_f)
			loss_cc += self.KL_Loss(logits_p_f, logits_q_z)
		elif self.fusion_type == 'quantized':
			logits_p_z, logits_q_z = Neg_z[:self.batch_size], Neg_z[self.batch_size:]
			loss_cc = self.KL_Loss(logits_p_z, logits_q_z)
		elif self.fusion_type == 'embedding':
			logits_p_f, logits_q_f = Neg_f[:self.batch_size], Neg_f[self.batch_size:]
			loss_cc = self.KL_Loss(logits_p_f, logits_q_f)
			
		else:
			if self.fusion_type == 'sum':
				FaZa = torch.add(Fa, Za)
				FbZb = torch.add(Fb, Zb)
			elif self.fusion_type == 'concatenate':
				FaZa = torch.cat([Fa, Za], dim=1)
				FbZb = torch.cat([Fb, Zb], dim=1)
			elif self.fusion_type == 'max':
				FaZa = torch.max(Fa, Za)
				FbZb = torch.max(Fb, Zb)

			fused_rep = torch.cat([FaZa, FbZb], dim=0)
			Sim_fused = self.sim_metric(fused_rep.unsqueeze(1), fused_rep.unsqueeze(0))
			Neg_fused = Sim_fused[self.mask_neg].view(2*self.batch_size, -1)
			logits_p, logits_q = Neg_fused[:self.batch_size], Neg_fused[self.batch_size:]
			loss_cc = self.KL_Loss(logits_p, logits_q)
		time2 = time.time()-time1 
		if self.args.get_timing:
			self.time_dict['cc'].append(time2)

		#### Part consistent quantization
		# Part neighbor discriminative learning loss
		time1 = time.time()
		loss_pn = self.PN_Loss(Za, Zb)
		time2 = time.time()-time1 
		if self.args.get_timing:
			self.time_dict['pn'].append(time2)
		# Entropy maximization based codeword diversity regularization
		time1 = time.time()
		loss_cd = self.ER_Loss(Pa, Pb)
		time2 = time.time()-time1 
		if self.args.get_timing:
			self.time_dict['cd'].append(time2)

		return loss_icz, loss_icf, loss_cc, loss_pn, loss_cd, loss_cl, loss_cqc

