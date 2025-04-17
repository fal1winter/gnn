import torch as t
import torch
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import infoNCE, KLDiverge, pairPredict, calcRegLoss
import torch_sparse
import random as rd

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


def matmul_sparse_dense( adj, embeds, torch_sparse_flag=True):
	#return t.spmm(adj, embeds)
	if torch_sparse_flag:
		row, col, val = adj.coo()

		index = torch.stack([row,col],dim=0)
		value = val
		m = adj.sizes()[0]
		n = adj.sizes()[1]
		matrix = embeds
		return torch_sparse.spmm(index, value, m, n, matrix)

	else:
		return t.sparse.mm(adj, embeds)



def subspace_contrast_loss(mask, embs, ancs , weight, threshold):
	mask = mask[ancs]
	embs = embs[ancs]
		
	target = (mask @ mask.T) >= (threshold-0.5)
	target = target.float()	
	
	logits = embs @ embs.T
	
	loss_fn = nn.CrossEntropyLoss()

	loss =  loss_fn(logits  , target) * weight
	return loss


def subspace_hyperedge_contrast_loss(usr_mask, item_mask,  usr_embs, item_embs, Uancs, Ipos,  weight, threshold, temp,prn):
	if args.hyper_contr_resample:
		# print("###################Uancs size#############################3")
		# print(len(Uancs))
		# print(len(Ipos))
		# print(Uancs)
		Uancs = rd.sample(Uancs.tolist(), int(args.hyper_contr_resample * len(Uancs.tolist())))
		Ipos = rd.sample(Ipos.tolist(), int(args.hyper_contr_resample * len(Ipos.tolist())))

	u_mask = usr_mask[Uancs] 
	i_mask = item_mask[Ipos]

	u_embs = usr_embs[Uancs]
	i_embs = item_embs[Ipos]
		
	u_threM_l = t.unsqueeze(u_mask.sum(axis=-1), -1)
	u_threM_l = u_threM_l.expand(u_mask.shape[0],  u_mask.shape[0])

	u_threM_r = t.unsqueeze(u_mask.sum(axis=-1), 0)
	u_threM_r = u_threM_r.expand(u_mask.shape[0],  u_mask.shape[0])


	i_threM_l = t.unsqueeze(i_mask.sum(axis=-1), -1)
	i_threM_l = i_threM_l.expand(i_mask.shape[0],  i_mask.shape[0])

	i_threM_r = t.unsqueeze(i_mask.sum(axis=-1), 0)
	i_threM_r = i_threM_r.expand(i_mask.shape[0],  i_mask.shape[0])

	
	loosen_factor =  args.hyper_contr_loosen_factor

	u_target_l = (u_mask @ u_mask.T) >= (u_threM_l - loosen_factor[prn]  )
	u_target_r = (u_mask @ u_mask.T) >= (u_threM_r - loosen_factor[prn]  )

	u_target = (u_target_l * u_target_r).float()
		

	i_target_l = (i_mask @ i_mask.T) >= (i_threM_l - loosen_factor[prn]  )  
	i_target_r = (i_mask @ i_mask.T) >= (i_threM_r - loosen_factor[prn]  )


	i_target =  (i_target_l * i_target_r).float()
		
	u_logits = (u_embs @ u_embs.T)/ temp

	i_logits = (i_embs @ i_embs.T) / temp
	
	u_loss_fn = nn.CrossEntropyLoss()
	i_loss_fn = nn.CrossEntropyLoss()

	u_loss = u_loss_fn(u_logits  , u_target)  
	i_loss = i_loss_fn(i_logits  , i_target) 
	loss = (u_loss + i_loss) * weight

	return loss


def subspace_hyperedge_contrast_loss_v2(usr_mask, item_mask,  usr_embs, item_embs, Uancs, Ipos,  weight, threshold, temp,prn):
	if args.hyper_contr_resample:
		# print("###################Uancs size#############################3")
		# print(len(Uancs))
		# print(len(Ipos))
		# print(Uancs)
		Uancs = rd.sample(Uancs.tolist(), int(args.hyper_contr_resample * len(Uancs.tolist())))
		Ipos = rd.sample(Ipos.tolist(), int(args.hyper_contr_resample * len(Ipos.tolist())))

	u_mask = usr_mask[Uancs] 
	i_mask = item_mask[Ipos]
	ui_mask =  t.concat([u_mask, i_mask],  dim=0)


	u_embs = usr_embs[Uancs]
	i_embs = item_embs[Ipos]
	ui_embs = t.concat([u_embs, i_embs],  dim=0)

	ui_threM_l = t.unsqueeze(ui_mask.sum(axis=-1), -1)
	ui_threM_l = ui_threM_l.expand(ui_mask.shape[0],  ui_mask.shape[0])
	ui_threM_r = ui_threM_l.T
	
	loosen_factor =  args.hyper_contr_loosen_factor

	ui_mask_square = (ui_mask @ ui_mask.T)

	ui_target_l = ui_mask_square >= (ui_threM_l - loosen_factor[prn]  )
	ui_target_r = ui_mask_square >= (ui_threM_r - loosen_factor[prn]  )


	ui_target = (ui_target_l * ui_target_r).float()	

	ui_logits = (ui_embs @ ui_embs.T)/ temp
		
	ui_loss_fn = nn.CrossEntropyLoss()

	ui_loss = ui_loss_fn(ui_logits  , ui_target)  
	loss = (ui_loss) * weight

	return loss


def subspace_kernel(embeds1_hid, embeds2_hid, mask1, mask2 ,nodes1,prn, temp=1.0):

	pckEmbeds1 = embeds1_hid[nodes1]
	pckMask1 = mask1[nodes1]
	preds_logits = pckEmbeds1 @ embeds2_hid.T / temp	

	threM_l = t.unsqueeze(pckMask1.sum(axis=-1), -1)
	threM_l = threM_l.expand(pckMask1.shape[0],  mask2.shape[0])

	threM_r = t.unsqueeze(mask2.sum(axis=-1), 0)
	threM_r = threM_r.expand(pckMask1.shape[0],  mask2.shape[0])
	
	loosen_factor =  args.hyper_contr_loosen_factor

	masks_cross = pckMask1 @ mask2.T
	
	target_l = masks_cross >= (threM_l - loosen_factor[prn]  )
	target_r = masks_cross >= (threM_r - loosen_factor[prn]  )

	target = (target_l * target_r).float()
		
	loss_fn = nn.CrossEntropyLoss()

	loss = loss_fn(preds_logits  , target)  	
	return loss



def subspace_hyperedge_contrast_loss_v3(usr_mask, item_mask,  usr_embs, item_embs, Uancs, Ipos,  weight, threshold, temp,prn):
	if args.hyper_contr_resample:
		# print("###################Uancs size#############################")
		# print(len(Uancs))
		# print(len(Ipos))
		# print(Uancs)
		Uancs = rd.sample(Uancs.tolist(), int(args.hyper_contr_resample * len(Uancs.tolist())))
		Ipos = rd.sample(Ipos.tolist(), int(args.hyper_contr_resample * len(Ipos.tolist())))

		u_len = len(usr_mask)
		i_len = len(item_mask)

		u_idx = rd.sample(range(u_len), int(args.hyper_contr_resample * u_len))
		i_idx = rd.sample(range(i_len), int(args.hyper_contr_resample * i_len))

		usr_mask = usr_mask[u_idx]
		item_mask = item_mask[i_idx]

		usr_embs = usr_embs[u_idx]
		item_embs = item_embs[i_idx]

	loss = 0
	loss += subspace_kernel(usr_embs,  item_embs, usr_mask,  item_mask ,Uancs, prn, temp)
	loss += subspace_kernel(item_embs, usr_embs,  item_mask, usr_mask,  Ipos,  prn, temp)
	loss += subspace_kernel(usr_embs,  usr_embs,  usr_mask,  usr_mask,  Uancs, prn, temp)
	loss += subspace_kernel(item_embs, item_embs, item_mask, item_mask, Ipos,  prn, temp)
	loss *= weight

	return loss

class Model(nn.Module):
	def __init__(self, teacher, student, middle_teacher=None):
		super(Model, self).__init__()

		self.teacher = teacher
		self.student = student
		self.middle_teacher = middle_teacher
	
	def forward(self):
		pass

	def calcLoss(self, adj_tea, adj_stu, adj_closure,  ancs, poss, negs, opt,  prn_run = 0, train_flag=False, emb_spar=1.):
		uniqAncs = t.unique(ancs)
		uniqPoss = t.unique(poss)
		if args.train_middle_model:
			suEmbeds, siEmbeds = self.middle_teacher(adj_closure)
		else:
			suEmbeds, siEmbeds = self.student(adj_stu)
		
		if args.distill_from_middle_model:		
			tuEmbeds, tiEmbeds = self.middle_teacher(adj_closure)
		else:
			tuEmbeds, tiEmbeds = self.teacher(adj_tea)

		tuEmbeds = tuEmbeds.detach()
		tiEmbeds = tiEmbeds.detach()

		rdmUsrs = t.randint(args.user, [args.topRange])#ancs
		rdmItms1 = t.randint_like(rdmUsrs, args.item)
		rdmItms2 = t.randint_like(rdmUsrs, args.item)

		if args.distill_from_middle_model:
			tEmbedsLst = self.middle_teacher(adj_closure, getMultOrder=True)
		else:
			tEmbedsLst = self.teacher(adj_tea, getMultOrder=True)

		highEmbeds = sum(tEmbedsLst[2:])
		highuEmbeds = highEmbeds[:args.user].detach()
		highiEmbeds = highEmbeds[args.user:].detach()

		if args.use_emb_level_kd:
			contrastDistill = (infoNCE(highuEmbeds, suEmbeds, uniqAncs, args.tempcd) + infoNCE(highiEmbeds, siEmbeds, uniqPoss, args.tempcd)) * args.cdreg
		else:
			contrastDistill = 0.

		# soft-target-based distillation
		
		if args.distill_from_middle_model:		
			tpairPreds = self.middle_teacher.pairPredictwEmbeds(tuEmbeds, tiEmbeds, rdmUsrs, rdmItms1, rdmItms2)
		else:
			tpairPreds = self.teacher.pairPredictwEmbeds(tuEmbeds, tiEmbeds, rdmUsrs, rdmItms1, rdmItms2)


		if args.train_middle_model:
			spairPreds = self.middle_teacher.pairPredictwEmbeds(suEmbeds, siEmbeds, rdmUsrs, rdmItms1, rdmItms2)
		else:
			spairPreds = self.student.pairPredictwEmbeds(suEmbeds, siEmbeds, rdmUsrs, rdmItms1, rdmItms2)
		
		if args.use_pre_level_kd:
			softTargetDistill = KLDiverge(tpairPreds, spairPreds, args.tempsoft) * args.softreg
		else:
			softTargetDistill = 0
		#softTargetDistill = -1

		if args.train_middle_model:
			mainLoss = -self.middle_teacher.pairPredictwEmbeds(suEmbeds, siEmbeds, ancs, poss, negs).sigmoid().log().mean()
		else:
			mainLoss = -self.student.pairPredictwEmbeds(suEmbeds, siEmbeds, ancs, poss, negs).sigmoid().log().mean()

		if args.use_contr_deno:
			if args.train_middle_model:	
				selfContrast = 0
				selfContrast += (t.log(self.middle_teacher.pointNegPredictwEmbeds(suEmbeds, siEmbeds, uniqAncs, args.tempsc) + 1e-5) ).mean()
				selfContrast += (t.log(self.middle_teacher.pointNegPredictwEmbeds(siEmbeds, suEmbeds, uniqPoss, args.tempsc) + 1e-5) ).mean()
				selfContrast += (t.log(self.middle_teacher.pointNegPredictwEmbeds(suEmbeds, suEmbeds, uniqAncs, args.tempsc) + 1e-5) ).mean()
				selfContrast += (t.log(self.middle_teacher.pointNegPredictwEmbeds(siEmbeds, siEmbeds, uniqPoss, args.tempsc) + 1e-5) ).mean()
				selfContrast *= args.screg

			else:
				selfContrast = 0
				selfContrast += (t.log(self.student.pointNegPredictwEmbeds(suEmbeds, siEmbeds, uniqAncs, args.tempsc) + 1e-5) ).mean()
				selfContrast += (t.log(self.student.pointNegPredictwEmbeds(siEmbeds, suEmbeds, uniqPoss, args.tempsc) + 1e-5) ).mean()
				selfContrast += (t.log(self.student.pointNegPredictwEmbeds(suEmbeds, suEmbeds, uniqAncs, args.tempsc) + 1e-5) ).mean()
				selfContrast += (t.log(self.student.pointNegPredictwEmbeds(siEmbeds, siEmbeds, uniqPoss, args.tempsc) + 1e-5) ).mean()
				selfContrast *= args.screg
		else:
			selfContrast = 0


		# weight-decay reg
		if args.train_middle_model:	
			regParams = [self.middle_teacher.uEmbeds, self.middle_teacher.iEmbeds]
			edgeWeights = [self.middle_teacher.adj_mask1_train]
		else:	
			regParams = [self.student.uEmbeds, self.student.iEmbeds]
			edgeWeights = [self.student.adj_mask1_train]
		if args.use_ereg:
			regLoss = calcRegLoss(params=regParams) * args.reg + calcRegLoss(params=edgeWeights) * args.ereg
		else:
			regLoss = calcRegLoss(params=regParams) * args.reg


		if (args.use_subspcreg) and (prn_run >= args.hyper_contr_start_prn_run) and (prn_run <= args.hyper_contr_end_prn_run) and (not args.train_middle_model):			
			threshold = args.latdim * (emb_spar / 100. )						

			embs = t.concat([self.student.uEmbeds, self.student.iEmbeds],  dim=0)

			masked_embs = torch.mul(self.student.emb_mask2_fixed , embs )

			if args.use_ly0emb4subspcreg:
				suEmbeds_sub = masked_embs[:args.user]
				siEmbeds_sub = masked_embs[args.user:]
			else:
				suEmbeds_sub = suEmbeds
				siEmbeds_sub = siEmbeds
			
			mask = self.student.emb_mask2_fixed.detach()
			usr_mask = mask[:args.user]
			item_mask = mask[args.user:]

			if args.subspcreg_version == 'v3':
				subsp_loss = subspace_hyperedge_contrast_loss_v3(usr_mask, item_mask,  suEmbeds_sub, siEmbeds_sub, uniqAncs, uniqPoss,  args.subspcreg, threshold, args.tempsubsp, prn_run)

			elif args.subspcreg_version == 'v2':
				subsp_loss = subspace_hyperedge_contrast_loss_v2(usr_mask, item_mask,  suEmbeds_sub, siEmbeds_sub, uniqAncs, uniqPoss,  args.subspcreg, threshold, args.tempsubsp, prn_run)
			else:
				#subsp_loss = subspace_hyperedge_contrast_loss(usr_mask, item_mask,  suEmbeds, siEmbeds, uniqAncs, uniqPoss,  args.subspcreg, threshold, args.tempsubsp, prn_run)
				subsp_loss = subspace_hyperedge_contrast_loss(usr_mask, item_mask,  suEmbeds_sub, siEmbeds_sub, uniqAncs, uniqPoss,  args.subspcreg, threshold, args.tempsubsp, prn_run)
		else:
			#print("################Do not use subspcreg#######################")
			subsp_loss = 0.
		
		loss = mainLoss + softTargetDistill + regLoss + selfContrast + contrastDistill + subsp_loss		
		
		losses = {'mainLoss': mainLoss, 'contrastDistill': contrastDistill, 'softTargetDistill': softTargetDistill, 'regLoss': regLoss}
		return loss, losses



class LightGCN_sp(nn.Module):
	def __init__(self, adj):
		super(LightGCN_sp, self).__init__()

		self.layer_num = args.gnn_layer
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.net_layer = nn.Sequential(*[GCNLayer_sp() for i in range(args.gnn_layer)])

		self.adj_mask1_train = nn.Parameter(torch.ones(adj.nnz()))
		self.adj_mask2_fixed = nn.Parameter(torch.ones(adj.nnz()), requires_grad=False)	
		
		self.adj_nonzero = adj.nnz()
		self.emb_mask2_fixed = nn.Parameter(torch.ones(args.item+args.user, args.latdim), requires_grad=False)


	def forward(self, adj, getMultOrder=False, getTogether=False):

		adj = adj.mul_nnz(self.adj_mask1_train, layout='coo')
		adj = adj.mul_nnz(self.adj_mask2_fixed, layout='coo')

		embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)		
		embeds = torch.mul( self.emb_mask2_fixed ,embeds )

		embedsLst = [embeds]
		for gcn in self.net_layer:
			embeds = gcn(adj, embedsLst[-1])			
			embedsLst.append(embeds)
		
		embeds = sum(embedsLst)
		if not getMultOrder:			
			if getTogether:
				return embeds
			else:
				return embeds[:args.user], embeds[args.user:]
		else:
			return embedsLst
	
	def pairPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss, negs):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		return pairPredict(ancEmbeds, posEmbeds, negEmbeds)


	def pointPosPredictwEmbeds(self, uEmbeds_hid, iEmbeds_hid, ancs, poss):
		ancEmbeds = uEmbeds_hid[ancs]
		posEmbeds = iEmbeds_hid[poss]
		nume = self.pairPred(ancEmbeds, posEmbeds)
		return nume
	
	def pointNegPredictwEmbeds(self, embeds1_hid, embeds2_hid, nodes1, temp=1.0):
		pckEmbeds1 = embeds1_hid[nodes1]
		preds = self.crossPred(pckEmbeds1, embeds2_hid)
		return t.exp(preds / temp).sum(-1)


	def predAll(self, pckUEmbeds, iEmbeds):
		return pckUEmbeds @ iEmbeds.T
	
	def testPred(self, usr, trnMask, adj):
		uEmbeds, iEmbeds = self.forward(adj)
		allPreds = self.predAll(uEmbeds[usr], iEmbeds) * (1 - trnMask) - trnMask * 1e8
		return allPreds

	def pairPred(self, embeds1, embeds2):
		return (embeds1 * embeds2).sum(dim=-1)	

	def crossPred(self, embeds1, embeds2):
		return embeds1 @ embeds2.T

class GCNLayer_sp(nn.Module):
	def __init__(self):
		super(GCNLayer_sp, self).__init__()

	def forward(self, adj, embeds, torch_sparse_flag=False):
		if torch_sparse_flag:
			row, col, val = adj.coo()

			index = torch.stack([row,col],dim=0)
			value = val
			m = adj.sizes()[0]
			n = adj.sizes()[1]
			matrix = embeds
			res = torch_sparse.spmm(index, value, m, n, matrix)
			
			return res

		else:
			return adj.matmul(embeds)




class LightGCN(nn.Module):
	def __init__(self):
		super(LightGCN, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

	def forward(self, adj, getMultOrder=False):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)
		if not getMultOrder:
			return embeds[:args.user], embeds[args.user:]
		else:
			return embedsLst
	
	def pairPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss, negs):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		return pairPredict(ancEmbeds, posEmbeds, negEmbeds)
	
	def predAll(self, pckUEmbeds, iEmbeds):
		return pckUEmbeds @ iEmbeds.T
	
	def testPred(self, usr, trnMask, adj):
		uEmbeds, iEmbeds = self.forward(adj)
		allPreds = self.predAll(uEmbeds[usr], iEmbeds) * (1 - trnMask) - trnMask * 1e8
		return allPreds


class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		#return t.spmm(adj, embeds)
		return adj.matmul(embeds)