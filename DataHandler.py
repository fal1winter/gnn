import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import torch_sparse
import scipy

import random as rd


t.manual_seed(args.seed)
t.cuda.manual_seed_all(args.seed)
t.backends.cudnn.deterministic = True
np.random.seed(args.seed)
rd.seed(args.seed)

class DataHandler:
	def __init__(self):
		if args.dataset == 'yelp':
			predir = './Datasets/sparse_yelp/'
		elif args.dataset == 'gowalla':
			predir = './Datasets/sparse_gowalla/'
		elif args.dataset == 'amazon':
			predir = './Datasets/sparse_amazon/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		self.adj_aug_layer = args.adj_aug_layer
		self.adj_aug = args.adj_aug
		self.adj_aug_sample = args.adj_aug_sample
		self.adj_aug_sample_ratio = args.adj_aug_sample_ratio
		self.train_middle_model = args.train_middle_model
		self.adj_aug_sample_random = args.adj_aug_sample_random

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		flag = False
		if isinstance(mat, scipy.sparse._csr.csr_matrix):
			pass
		
		elif isinstance(mat, torch_sparse.tensor.SparseTensor):
			mat = mat.to_scipy(layout='csr')
			flag = 	True
		else:
			pass

		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		res =  (dInvSqrtMat.dot(mat.dot(dInvSqrtMat))).tocoo()

		if flag:
			res = torch_sparse.SparseTensor.from_scipy(res).cuda()

		return res



	def makeTorchAdj(self, mat):
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0  # type: scipy.sparse._csr.csr_matrix
		
		ori_Adj = torch_sparse.SparseTensor.from_scipy(mat).cuda()		
		I = torch_sparse.SparseTensor.eye(mat.shape[0]).cuda()		
		
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		rows = t.from_numpy(mat.row.astype(np.int64))
		cols = t.from_numpy(mat.col.astype(np.int64))		
		shape = t.Size(mat.shape)

		adj_tea =  t.sparse.FloatTensor(idxs, vals, shape).cuda()
		adj_stu =  torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(adj_tea).cuda()
	
		# print("#####################adj_stu requires_grad##########################")
		# print(adj_stu.requires_grad())
		
		if self.adj_aug:
			last_An = ori_Adj
			closure_Adj = I + ori_Adj
			
			if self.adj_aug_sample:
				for i in range(self.adj_aug_layer):
					next_An = last_An.matmul( ori_Adj)
					row, col, vals= next_An.coo()
					length = row.shape[0]
					
					if self.adj_aug_sample_random:
						idx = rd.sample(range(length), int(self.adj_aug_sample_ratio *length))
						idx.sort()
					
					else:				
						val_l = vals.tolist()
						def fn(x):
							return val_l[x]
						idx = range(length)
						idx = sorted(idx,  reverse=True, key = fn)
						idx = idx[:int(self.adj_aug_sample_ratio * length)]
						idx.sort()						

					new_row = row[idx]
					new_col = col[idx]
					vals = vals[idx]
					next_An = torch_sparse.SparseTensor(row=new_row,col=new_col,value=vals, sparse_sizes=last_An.sizes())
					closure_Adj = closure_Adj + next_An
					last_An = next_An								
			else:
				for i in range(self.adj_aug_layer):
					next_An = last_An.matmul(ori_Adj)
					closure_Adj = closure_Adj + next_An
					last_An = next_An

			closure_Adj_norm = self.normalizeAdj(closure_Adj)

		else:
			closure_Adj = None
			closure_Adj_norm = None
		#return adj_tea,adj_stu, closure_Adj_norm, (clo_rows, clo_cols) #(rows, cols)

		if self.train_middle_model:
			clo_rows, clo_cols,  _ = closure_Adj_norm.coo()
			return adj_tea, adj_stu, closure_Adj_norm, (clo_rows, clo_cols)
		else:
			return adj_tea, adj_stu, closure_Adj_norm, (rows, cols)

		
	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape		

		self.adj_tea, self.adj_stu, self.adj_closure, self.indices = self.makeTorchAdj(trnMat)
		
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row 
		self.cols = coomat.col 
		self.dokmat = coomat.todok() 
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
