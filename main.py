"""
Self-Supervised Consistent Quantization for Fully Unsupervised Image Retrieval.
Computer Vision Group, Cambridge Research Lab, Toshiba Europe Limited.
Guile.
April, 2022.
"""

from email.policy import strict
import os
import math
import random
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from dataset.data_manager import load_data
from models import Projection_Head, Quantization_Head, ResNet18_Cifar, resnet18, resnet50, vgg16
from lib.losses import SSCQ_Loss
from utils.logger import setup_logger
from utils.retrieval import perform_retrieval, perform_intra_inter_loss, draw_tsne_fig
from datetime import datetime

# Hyper-parameter settings
parser = argparse.ArgumentParser('SSCQ', add_help=False)

parser.add_argument('--gpu_id', default='0', type=str, help='Used GPU ids.')
parser.add_argument('--random_seed', default=2022, type=int, help='Used random seed.')
parser.add_argument('--run', default=1, type=int, help='The i-th run of the experiment.')
parser.add_argument('--data_dir', default='/home/gwu/Desktop/data', type=str, help='Path of the dataset to be loaded.')
parser.add_argument('--dataset', default='cifar10',type=str, help='Dataset for model training and evaluation.')
parser.add_argument('--output_dir', default='./outputs/', type=str, help='Path to save logs and checkpoints.')
parser.add_argument('--exp_name', default='test', type=str, help='Path to save logs and checkpoints.')
parser.add_argument('--ckpt_name', default='test.pth', type=str, help='Path to save logs and checkpoints.')


parser.add_argument('--backbone', default='resnet18_cifar', type=str, help='Backbone architecture of the model')
parser.add_argument('--batch_size', default=256, type=int, help='Training mini-batch size.')
parser.add_argument('--eval_epoch', default=10, type=int, help='Compute mAP for Every N-th epoch.')
parser.add_argument('--print_iter', default=50, type=int, help='Print training loss every N iterations.')
parser.add_argument('--epoch_max', default=800, type=int, help='Number of training epochs')
parser.add_argument('--epoch_warm_lr', default=10, type=int, help='Number of warm-up epochs for decaying learning rate')
parser.add_argument('--lr', default=5e-4, type=float, help='Initial learning rate')
parser.add_argument('--lr_warmup_from', default=1e-5, type=float, help='Warm-up learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='Weight decay for the learning rate')

parser.add_argument('--N_books', default=8, type=int, help='Number of the codebooks.')
parser.add_argument('--N_words', default=16, type=int, help='Number of the codewords. It should be a power of two.')
parser.add_argument('--L_word', default=16, type=int, help='Dimensionality of the codeword.')
parser.add_argument('--Top_N', default=1000, type=int, help='Top N number of images to be retrieved for evaluation.')

parser.add_argument('--T_instance_contrastive', default=0.5, type=float, help='Temperature parameter for instance contrastive learning (default: 0.5-1.0).')
parser.add_argument('--T_soft_quantization', default=0.2, type=float, help='Temperature parameter for soft-quantization (default: 0.2).')
parser.add_argument('--T_consistent_contrastive', default=0.2, type=float, help='Temperature parameter for consistent contrastive regularization (default: 0.2).')
parser.add_argument('--T_part_neighbor', default=0.5, type=float, help='Temperature parameter for part neighbor discriminative learning (default: 0.2-0.5).')
parser.add_argument('--W_consistent_contrastive', default=0.4, type=float, help='Weight for consistent contrastive regularization (default: 0.4).')
parser.add_argument('--W_part_neighbor', default=0.1, type=float, help='Weight for part neighbor discriminative learning (default: 0.1).')
parser.add_argument('--W_codeword_diversity', default=0.2, type=float, help='Weight for codeword diversity regularization (default: 0.2).')
parser.add_argument('--N_top_part_neighbor', default=20, type=int, help='Number of top-Nk neighbors for part discriminative learning (default: 20)')
parser.add_argument('--fusion_type', default='sum', type=str, help='Fusion strategy (sum, concatenate, max, cross, quantized) for consistent contrastive regularization.')

parser.add_argument('--W_codebook_classfiy', default=0.0, type=float, help='Weight for codebook classification (default: 0.1).')
parser.add_argument('--W_icz', default=0., type=float, help='Weight for codebook classification (default: 0.1).')
parser.add_argument('--W_icf', default=0., type=float, help='Weight for codebook classification (default: 0.1).')
parser.add_argument('--W_cqc', default=0., type=float, help='Weight for codebook classification (default: 0.1).')

parser.add_argument('--pn_use_pos', action='store_true', help='Use pos sample in pn loss.')

parser.add_argument('--evaluate_only', action='store_true', help='Perform evaluation without training.')
parser.add_argument('--get_intra_inter_loss', action='store_true', help='Perform evaluation without training.')
parser.add_argument('--draw_tsne_figure', action='store_true', help='Perform evaluation without training.')

parser.add_argument('--save_pr_curve_data', action='store_true', help='Save data for pr curves and p-at-topk curves.')
parser.add_argument('--get_timing', action='store_true', help='Log timing of each loss term.')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# Set fixed random seed
if args.random_seed is not None:
	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)
torch.backends.cudnn.benchmark = True


def main():
	device = torch.device('cuda')
	#device = torch.cuda.device(0)
	dataset = args.dataset
	args.data_dir = os.path.join(args.data_dir, dataset)
	data_dir = args.data_dir
	batch_size = args.batch_size

	N_books = args.N_books
	N_words = args.N_words
	L_word = args.L_word
	tau_ic = args.T_instance_contrastive
	tau_sq = args.T_soft_quantization
	tau_cc = args.T_consistent_contrastive
	tau_pn = args.T_part_neighbor
	lambda_cc = args.W_consistent_contrastive
	lambda_pn = args.W_part_neighbor
	lambda_cd = args.W_codeword_diversity
	lambda_cl = args.W_codebook_classfiy
	lambda_icz = args.W_icz
	lambda_icf = args.W_icf
	lambda_cqc = args.W_cqc


	N_bits = int(N_books * np.log2(N_words))
	print('Image Retrieval with {:d} bits on {:s}'.format(N_bits, dataset))

	now = datetime.now()
	dt_string = now.strftime("%m%d_%H_%M_%S")

	# Set logger
	output_path = os.path.join(args.output_dir, './{}-{}-bit{}e{}lr{}seed{}-{}-{}'.format(args.dataset, args.backbone, N_bits, args.epoch_max, args.lr, args.random_seed, args.exp_name, dt_string))
	setup_logger(output_path)
	#print('-'*20)
	#print(args)
	#print('-'*20)

	##############################
	# Dataset configuration
	##############################
	trainloader, databaseloader, queryloader, trainset, databaseset, testset = load_data(data_dir, dataset, batch_size, args.random_seed)


	##############################
	# Model configuration
	##############################
	if args.backbone == 'resnet18_cifar': # modified resnet-18 for cifar-10 with 32x32 input image size
		head_proj = Projection_Head(512,  N_books, L_word,)
		head_quan = Quantization_Head(N_words, N_books, L_word, tau_sq)
		feature_extractor = ResNet18_Cifar()
	elif args.backbone =='resnet18': # standard resnet-18 for 224x224 input image size
		head_proj = Projection_Head(512,  N_books, L_word,)
		head_quan = Quantization_Head(N_words, N_books, L_word, tau_sq)
		feature_extractor = resnet18(pretrained=False)
	elif args.backbone =='resnet50': # standard resnet-50 for 224x224 input image size
		head_proj = Projection_Head(2048,  N_books, L_word,)
		head_quan = Quantization_Head(N_words, N_books, L_word, tau_sq)
		feature_extractor = resnet50(pretrained=False)
	elif args.backbone =='vgg': # standard vgg16 for 224x224 input image size (also compatible with 32x32 image size on cifar-10 (16 bits with lr of 1e-4))
		head_proj = Projection_Head(4096,  N_books, L_word,)
		head_quan = Quantization_Head(N_words, N_books, L_word, tau_sq)
		backbone_layer_train = 'ALL' if args.dataset=='cifar10' else 0
		feature_extractor = vgg16(pretrained=True, dataset=args.dataset, backbone_layer_train=backbone_layer_train)
		if N_bits==16 and args.dataset=='cifar10' and args.lr>1e-4:
			raise ValueError('Please use a small learning rate.')
	else:
		raise ValueError('Not known backbone architecture. Please change the backbone.')

	model = nn.Sequential(feature_extractor, head_proj, head_quan)
	model = torch.nn.DataParallel(model).cuda(device)


	##############################
	# Training and evaluation
	##############################
	criterion = SSCQ_Loss(args, device, batch_size, N_books, L_word, tau_ic, tau_cc, tau_pn, args.fusion_type, args.N_top_part_neighbor, args.pn_use_pos)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

	# Cosine decay schedule
	eta_decay = 0.1 if args.dataset=='cifar10' else 0.01
	eta_min = args.lr*eta_decay
	warmup_to = eta_min + (args.lr - eta_min) * (1+math.cos(math.pi * args.epoch_warm_lr / args.epoch_max)) / 2
	total_batches = len(trainloader)
	mAP, mAP_max = 0.0, 0.0

	# Inference without training
	if args.evaluate_only:
		ckpt_name = args.ckpt_name
		print(ckpt_name)
		try:
			with open(ckpt_name, 'rb') as f:
				model.load_state_dict(torch.load(f), strict=False)
			print(f'Loaded {ckpt_name}')
		except:
			print('Cannot find the pre-trained model.')
			return None
		if args.get_intra_inter_loss:
			inter_dist, intra_dist = perform_intra_inter_loss(device, args, model, model.module[-1].codebooks, databaseloader, queryloader, databaseset, testset)
			print(inter_dist, intra_dist)
			return
		if args.draw_tsne_figure:
			print('drawing t-sne figure')
			draw_tsne_fig(device, args, model, model.module[-1].codebooks, databaseloader, queryloader, databaseset, testset)
			return 

		# evaluate single book 
		mAP, PR_curve, P_at_topk_curve = perform_retrieval(device, args, model, model.module[-1].codebooks, databaseloader, queryloader, databaseset, testset, one_book=True)

		if args.save_pr_curve_data:
			np.savetxt(os.path.join(output_path, "{:d}-bits_PR_curve.txt".format(N_bits)), PR_curve)
			np.savetxt(os.path.join(output_path, "{:d}-bits_P_at_topk_curve.txt".format(N_bits)), P_at_topk_curve)

		return mAP

	# Training and evaluation
	for epoch in range(args.epoch_max):
		print('Epoch: %d' % (epoch))
		running_loss = 0.0
		running_loss_icz = 0.0
		running_loss_icf = 0.0
		running_loss_cc = 0.0
		running_loss_pn = 0.0
		running_loss_cd = 0.0
		running_loss_cl = 0.0
		running_loss_cqc = 0.0

		# Decay learning rate with cosine annealing schedule
		lr = eta_min + (args.lr-eta_min) * (1 + math.cos(math.pi * (epoch+1) / args.epoch_max)) / 2
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		model.train()

		for i, data in enumerate(trainloader, 0):
			# Warm up learning rate
			if epoch < args.epoch_warm_lr:
				p = (i + epoch * total_batches) / (args.epoch_warm_lr * total_batches)
				lr = args.lr_warmup_from + p * (warmup_to - args.lr_warmup_from)

				for param_group in optimizer.param_groups:
					param_group['lr'] = lr

			# Load data with two augmented views
			#import ipdb; ipdb.set_trace()
			Xa, Xb = data[0][0].to(device), data[0][1].to(device)

			optimizer.zero_grad()

			Fa, Za, Pa, Ca = model(Xa)
			Fb, Zb, Pb, Cb = model(Xb)

			loss_icz, loss_icf, loss_cc, loss_pn, loss_cd, loss_cl, loss_cqc = criterion(Fa, Fb, Za, Zb, Pa, Pb, Ca, Cb)
			loss_icz = loss_icz * lambda_icz
			loss_icf = loss_icf * lambda_icf
			loss_cqc = loss_cqc * lambda_cqc
			loss_cc = lambda_cc*loss_cc
			loss_pn = lambda_pn*loss_pn
			loss_cd = lambda_cd*loss_cd
			loss_cl = lambda_cl*loss_cl 
			loss = loss_icz + loss_icf + loss_cc + loss_pn + loss_cd + loss_cl + loss_cqc

			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			running_loss_icz += loss_icz.item()
			running_loss_icf += loss_icf.item()
			running_loss_cc += loss_cc.item()
			running_loss_pn += loss_pn.item()
			running_loss_cd += loss_cd.item()
			running_loss_cl += loss_cl.item()
			running_loss_cqc += loss_cqc.item()

			# print every 10 mini-batches
			if (i+1) % args.print_iter == 0:
				print("[{:3d}] loss: {:.4f}, loss_icf: {:.4f},  loss_icz: {:.4f}, loss_cc: {:.4f}, loss_pn: {:.4f}, loss_cd: {:.4f},  loss_cl: {:.4f}, mAP: {:.4f}, MAX mAP: {:.4f}, lr: {:.6f}".format(
					i+1, running_loss/args.print_iter, running_loss_icf/args.print_iter, running_loss_icz/args.print_iter,
					running_loss_cc/args.print_iter, running_loss_pn/args.print_iter, running_loss_cd/args.print_iter, 
					running_loss_cl/args.print_iter, running_loss_cqc/args.print_iter,
					mAP, mAP_max, lr))
				running_loss = 0.0
				running_loss_cc = 0.0
				running_loss_icz = 0.0
				running_loss_icf = 0.0
				running_loss_pn = 0.0
				running_loss_cd = 0.0
				running_loss_cl = 0.0
				running_loss_cqc = 0.0
				
		if args.get_timing:
			total_time = 0
			for key, val in criterion.time_dict.items():
				mean_time = np.mean(val)
				total_time += mean_time
				print(f'{key}: abs time: {mean_time:.4f}')
			for key, val in criterion.time_dict.items():
				mean_time = np.mean(val)
				rel_ratio = mean_time / total_time
				print(f'{key}: rel ratio: {rel_ratio:.4f}')
				
			print('Only run 1 epoch to get timing, ')
			break
		
		# Perform evaluation
		if (epoch+1) % args.eval_epoch == 0:
			mAP, PR_curve, P_at_topk_curve = perform_retrieval(device, args, model, model.module[-1].codebooks, databaseloader, queryloader, databaseset, testset)
			if mAP > mAP_max:
				Result_path = os.path.join(output_path, "{:d}-bits_checkpoint.pth".format(N_bits))
				torch.save(model.state_dict(), Result_path)
				mAP_max = mAP
				print('Find the best mAP: {:.4f}'.format(mAP_max))
				if args.save_pr_curve_data:
					np.savetxt(os.path.join(output_path, "{:d}-bits_PR_curve.txt".format(N_bits)), PR_curve)
					np.savetxt(os.path.join(output_path, "{:d}-bits_P_at_topk_curve.txt".format(N_bits)), P_at_topk_curve)

	return mAP_max


if __name__ == '__main__': 
	mAP_max = main()
	if mAP_max:
		print('Finally, the best mAP is: {:.4f}'.format(mAP_max))
