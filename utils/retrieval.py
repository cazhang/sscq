"""
Perform retrieval (mAP, PR curve data, P at topK data).
"""

import torch
import numpy as np
from tqdm import tqdm
import imageio
from utils.utils import indexing, pqDist_one
import torchshow as ts

def Evaluate_mAP(codebooks, N_books, database_codes, query_codes, database_labels, query_labels, device, TOP_K, book_idx=None):
	"""Compute mAP and precision at top-K curve data"""
	num_query = query_labels.shape[0]
	mean_AP = 0.0
	position_list = torch.arange(1, TOP_K+1, dtype=torch.float, device=device)
	pre = []

	with tqdm(total=num_query, desc="Computing mAP.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
		for i in range(num_query):
			# Retrieve images from database
			retrieval = (query_labels[i, :] @ database_labels.t() > 0).float().to(device)

			# Arrange position according to hamming distance
			retrieval = retrieval[torch.argsort(pqDist_one(codebooks, N_books, database_codes, query_codes[i], book_idx=book_idx))][:TOP_K]

			# Compute precision at top-K
			score_p = retrieval.cumsum(dim=-1)
			index_p = position_list
			pre.append((score_p/index_p).unsqueeze(0))

			# Retrieval count
			retrieval_cnt = retrieval.sum().int().item()

			# Can not retrieve images
			if retrieval_cnt == 0:
				continue

			# Generate score for every position
			score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

			# Acquire index
			index = (torch.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float().to(device)

			mean_AP += (score / index).mean()
			pbar.update(1)

		mean_AP = mean_AP / num_query
		P_at_topk = torch.cat(pre).mean(dim=0)
	return mean_AP, P_at_topk

def get_topK_retrieval(codebooks, N_books, database_codes, query_codes, database_labels, query_labels, device, TOP_K, book_idx):
	"""Compute mAP and precision at top-K curve data"""
	num_query = query_labels.shape[0]
	mean_AP = 0.0
	position_list = torch.arange(1, TOP_K+1, dtype=torch.float, device=device)
	ret_ind = []
	ret_flag = []

	with tqdm(total=num_query, desc="Getting retrieval.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
		for i in range(num_query):
			if i > 10: break 
			# Retrieve images from database
			retrieval = (query_labels[i, :] @ database_labels.t() > 0).float().to(device)

			# Arrange position
			topK_indices = torch.argsort(pqDist_one(codebooks, N_books, database_codes, query_codes[i], device=device, book_idx=book_idx))[:TOP_K]
			retrieval = retrieval[topK_indices]

			ret_ind.append(topK_indices)
			ret_flag.append(retrieval)

	return ret_ind, ret_flag




def pr_curve(codebooks, N_books, database_codes, query_codes, database_labels, query_labels, device=torch.device('cuda')):
	"""Computer precision-recall curve data"""
	num_query = query_labels.shape[0]
	position_list = torch.arange(1, len(database_labels)+1, dtype=torch.float, device=device)
	p, r = [], []

	with tqdm(total=num_query, desc="Computing PR data.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
		for i in range(num_query):
			# Retrieve images from database
			retrieval = (query_labels[i, :] @ database_labels.t() > 0).float().to(device)

			# Arrange position according to hamming distance
			retrieval = retrieval[torch.argsort(pqDist_one(codebooks, N_books, database_codes, query_codes[i]))]

			# Retrieval count
			retrieval_cnt = retrieval.sum(dim=-1)
			retrieval_cnt[retrieval_cnt<=10e-6] = 1.0

			# Can not retrieve images
			if retrieval_cnt == 0:
				continue
			score = retrieval.cumsum(dim=-1)
			index = position_list
			p.append((score/index).unsqueeze(0))
			r.append((score/retrieval_cnt.unsqueeze(dim=-1)).unsqueeze(0))
			pbar.update(1)
		Pdata = torch.cat(p).mean(dim=0)
		Rdata = torch.cat(r).mean(dim=0)

	return Pdata, Rdata




def perform_retrieval(device, args, model, codebooks, databaseloader, queryloader, databaseset, testset, one_book=False):
	"""Perform retireval"""
	print("Start performing retrieval.")

	model.eval()
	with torch.no_grad():
		with tqdm(total=len(databaseloader), desc="Creating retrieval database.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
			for i, data in enumerate(databaseloader, 0):
				database_x_batch, database_y_batch = data[0].to(device), data[1].to(device)
				outputs = model(database_x_batch)
				database_c_batch = indexing(codebooks, args.N_books, outputs[0])
				# single label
				if args.dataset == 'cifar10':
					database_y_batch = torch.eye(10)[database_y_batch]
				# multiple labels
				else:
					database_y_batch = database_y_batch.float()

				if i == 0:
					database_c = database_c_batch.to(device)
					database_y = database_y_batch.to(device)
				else:
					database_c = torch.cat([database_c, database_c_batch.to(device)], 0)
					database_y = torch.cat([database_y, database_y_batch.to(device)], 0)
				pbar.update(1)

		with tqdm(total=len(queryloader), desc="Computing query embeddings.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
			for i, data in enumerate(queryloader, 0):
				query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
				outputs = model(query_x_batch)
				# single label
				if args.dataset == 'cifar10':
					query_y_batch = torch.eye(10)[query_y_batch]
				else:
					query_y_batch = query_y_batch.float()

				if i == 0:
					query_c = outputs[0].to(device)
					query_y = query_y_batch.to(device)
				else:
					query_c = torch.cat([query_c, outputs[0].to(device)], 0)
					query_y = torch.cat([query_y, query_y_batch.to(device)], 0)
				pbar.update(1)

	if one_book:
		for book_idx in range(args.N_books):
			ind_topk, flag_topk = get_topK_retrieval(codebooks, args.N_books, database_c.type(torch.int), query_c, database_y, query_y, device, args.Top_N, book_idx=book_idx)
			# visualise retrieval 
			img_list = []
			n_query = len(ind_topk)
			for i in range(n_query):
				query_img, query_label = testset[i]
				img_list.append(query_img)
				for j in range(10):
					db_img, db_label = databaseset[ind_topk[i][j]]
					img_list.append(db_img)

			img_list = torch.stack(img_list)
			save_path = f'query_top10_book{book_idx}'
			ts.save(img_list, path=save_path, nrows=n_query, dpi=300, tight_layout=True)

			mAP, P_at_topk = Evaluate_mAP(codebooks, args.N_books, database_c.type(torch.int), query_c, database_y, query_y, device, args.Top_N, book_idx=book_idx)
			print(f'book{book_idx}, map:{mAP}')

			if args.save_pr_curve_data:
				Pdata, Rdata = pr_curve(codebooks, args.N_books, database_c.type(torch.int), query_c, database_y, query_y)
				return mAP, np.stack([Pdata.cpu().numpy(), Rdata.cpu().numpy()]), np.stack([np.arange(1, args.Top_N+1, dtype=np.float), P_at_topk.cpu().numpy()])
			else:
				return mAP, None, None
	
	else:

		ind_topk, flag_topk = get_topK_retrieval(codebooks, args.N_books, database_c.type(torch.int), query_c, database_y, query_y, device, args.Top_N, book_idx=None)
		# visualise retrieval 
		img_list = []
		n_query = len(ind_topk)
		for i in range(n_query):
			query_img, query_label = testset[i]
			img_list.append(query_img)
			for j in range(10):
				db_img, db_label = databaseset[ind_topk[i][j]]
				img_list.append(db_img)

		img_list = torch.stack(img_list)
		save_path = f'query_top10.png'
		ts.save(img_list, path=save_path, nrows=n_query, dpi=300, tight_layout=True)

		mAP, P_at_topk = Evaluate_mAP(codebooks, args.N_books, database_c.type(torch.int), query_c, database_y, query_y, device, args.Top_N, book_idx=None)
		print(f'map:{mAP}')

		if args.save_pr_curve_data:
			Pdata, Rdata = pr_curve(codebooks, args.N_books, database_c.type(torch.int), query_c, database_y, query_y)
			return mAP, np.stack([Pdata.cpu().numpy(), Rdata.cpu().numpy()]), np.stack([np.arange(1, args.Top_N+1, dtype=np.float), P_at_topk.cpu().numpy()])
		else:
			return mAP, None, None



def perform_intra_inter_loss(device, args, model, codebooks, databaseloader, queryloader, databaseset, testset):
	"""Perform intra-inter class loss using GT labels"""
	print("Start performing distance computation.")
	import os
	import torchmetrics
	save_dir = os.path.dirname(args.ckpt_name)
	save_name = os.path.join(save_dir, 'eval.txt')

	model.eval()
	with torch.no_grad():
		with tqdm(total=len(databaseloader), desc="Creating retrieval database.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
			for i, data in enumerate(databaseloader, 0):
				database_x_batch, database_y_batch = data[0].to(device), data[1].to(device)
				outputs = model(database_x_batch)
				#database_c_batch = indexing(codebooks, args.N_books, outputs[0])
				
				#database_y_batch = database_y_batch.float()

				if i == 0:
					database_c = outputs[0].to(device)
					database_y = database_y_batch.to(device)
				else:
					database_c = torch.cat([database_c, outputs[0].to(device)], 0)
					database_y = torch.cat([database_y, database_y_batch.to(device)], 0)
				pbar.update(1)

		with tqdm(total=len(queryloader), desc="Computing query embeddings.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
			for i, data in enumerate(queryloader, 0):
				query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
				outputs = model(query_x_batch)
				# single label
				#query_y_batch = query_y_batch.float()

				if i == 0:
					query_c = outputs[0].to(device)
					query_y = query_y_batch.to(device)
				else:
					query_c = torch.cat([query_c, outputs[0].to(device)], 0)
					query_y = torch.cat([query_y, query_y_batch.to(device)], 0)
				pbar.update(1)

	
	#query_base_dist = torch.cdist(query_c, database_c, p=2.0) # Euclidean distance
	query_base_dist = torchmetrics.functional.pairwise_cosine_similarity(x=query_c, y=database_c, reduction=None) # cosine sim

	same_list = []
	diff_list = []

	n_query, n_database = query_y.shape[0], database_y.shape[0]
	with tqdm(total=n_query, desc="Computing distance.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
		for i in range(n_query):
			#if i>10: break
			same_idx = torch.where(database_y == query_y[i])[0]
			n_same = len(same_idx)
			n_diff = n_database - n_same
			same_dist = torch.sum(query_base_dist[i, same_idx])
			diff_dist = torch.sum(query_base_dist[i]) - same_dist
			same_list.append(same_dist / n_same)
			diff_list.append(diff_dist / n_diff)
			pbar.update(1)
	
	avg_same_list = torch.mean(torch.FloatTensor(same_list))
	avg_diff_list = torch.mean(torch.FloatTensor(diff_list))

	with open(save_name, 'w') as f:
		f.write('Intra-inter analysis:\n')
		f.write(f'Intra dist: {avg_same_list:.4f}')
		f.write(f'Inter dist: {avg_diff_list:.4f}')

	return avg_same_list, avg_diff_list


# draw t_SNE figure using embeddings 
def draw_tsne_fig(device, args, model, codebooks, databaseloader, queryloader, databaseset, testset):
	"""Perform intra-inter class loss using GT labels"""
	print("Start performing distance computation.")

	from sklearn.manifold import TSNE
	from sklearn.datasets import load_iris
	from numpy import reshape
	import seaborn as sns
	import pandas as pd
	import matplotlib.pyplot as plt
	'''
	iris = load_iris()
	x = iris.data
	y = iris.target

	tsne = TSNE(n_components=2, verbose=1, random_state=123)
	z = tsne.fit_transform(x)
	df = pd.DataFrame()
	df["y"] = y
	df["comp-1"] = z[:,0]
	df["comp-2"] = z[:,1]

	sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
					palette=sns.color_palette("hls", 3),
					data=df).set(title="Iris data T-SNE projection")
	plt.savefig('save_as_a_png.png')
	return; 
	'''

	import os
	import torchmetrics
	from datetime import datetime
	now = datetime.now()
	dt_string = now.strftime("%m%d_%H_%M_%S")

	save_dir = os.path.dirname(args.ckpt_name)
	save_name = os.path.join(save_dir, f'{dt_string}_tsne.png')

	model.eval()
	with torch.no_grad():
		with tqdm(total=len(databaseloader), desc="Creating retrieval database.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
			for i, data in enumerate(databaseloader, 0):
				database_x_batch, database_y_batch = data[0].to(device), data[1].to(device)
				outputs = model(database_x_batch)
				#database_c_batch = indexing(codebooks, args.N_books, outputs[0])
				
				#database_y_batch = database_y_batch.float()

				if i == 0:
					database_c = outputs[0].to(device)
					database_y = database_y_batch.to(device)
				else:
					database_c = torch.cat([database_c, outputs[0].to(device)], 0)
					database_y = torch.cat([database_y, database_y_batch.to(device)], 0)
				pbar.update(1)

		with tqdm(total=len(queryloader), desc="Computing query embeddings.", bar_format='{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
			for i, data in enumerate(queryloader, 0):
				query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
				outputs = model(query_x_batch)
				# single label
				#query_y_batch = query_y_batch.float()

				if i == 0:
					query_c = outputs[0].to(device)
					query_y = query_y_batch.to(device)
				else:
					query_c = torch.cat([query_c, outputs[0].to(device)], 0)
					query_y = torch.cat([query_y, query_y_batch.to(device)], 0)
				pbar.update(1)

	x = query_c.cpu().numpy()[:10000]
	y = query_y.cpu().numpy()[:10000]
	print(x.shape, y.shape)
	ind3 = np.where(y==3)
	ind5 = np.where(y==5)
	ind0 = np.where(y==0)
	ind8 = np.where(y==8)
	#y[ind3]=0
	#y[ind0]=3
	#y[ind5]=8
	#y[ind8]=5
	# swap y: 3 -> 0, 5 -> 8
	tsne = TSNE(n_components=2, verbose=1, random_state=123)
	z = tsne.fit_transform(x)
	df = pd.DataFrame()
	df["y"] = y
	df["tsne-1"] = z[:,0]
	df["tsne-2"] = z[:,1]

	#df = pd.DataFrame(z, y, ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
	#plt.legend(fontsize=20) 
	gfg=sns.scatterplot(x="tsne-1", y="tsne-2", hue=df.y.tolist(),
					palette=sns.color_palette("hls", 10),
					data=df)
	gfg.legend(fontsize=15)
	gfg.set_xlabel("tsne-1",fontsize=15)
	gfg.set_ylabel("tsne-2",fontsize=15)


	#plt.legend(labels=["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
	plt.savefig(save_name)
	

	return

