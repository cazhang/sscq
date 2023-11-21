"""
Head modules: Projection head and Quantization head
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.utils import soft_quantization

class Projection_Head(nn.Module):
	def __init__(self, dim_in, N_books, L_word):
		super(Projection_Head, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(dim_in, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, N_books*L_word),
		)
		nn.init.normal_(self.mlp[0].weight, 0, 0.01)
		nn.init.xavier_uniform_(self.mlp[2].weight)

	def forward(self, x):
		f = self.mlp(x)
		return f


class Quantization_Head(nn.Module):
	def __init__(self, N_words, N_books, L_word, tau_sq):
		super(Quantization_Head, self).__init__()
		self.codebooks = nn.Parameter(Variable((torch.randn(N_words, N_books*L_word)).type(torch.float32), requires_grad=True))
		nn.init.xavier_uniform_(self.codebooks)
		
		# add CLS head here:
		self.mlp = nn.Sequential(
			nn.Linear(L_word, L_word*4),
			nn.ReLU(inplace=True),
			nn.Linear(L_word*4, N_books),
			nn.LogSoftmax(dim=1),
		)
		nn.init.normal_(self.mlp[0].weight, 0, 0.01)
		nn.init.xavier_uniform_(self.mlp[2].weight)

		self.N_books = N_books
		self.L_word = L_word
		self.tau_sq = tau_sq

	def forward(self, f):
		z, p = soft_quantization(f, self.codebooks, self.N_books, self.tau_sq)
		
		# predict codebook class 
		#import ipdb; ipdb.set_trace()
		f_split = f.view(-1, self.L_word)
		logits_codebook = self.mlp(f_split)
		return f, z, p, logits_codebook


