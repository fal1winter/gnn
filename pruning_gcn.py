import torch
import numpy as np
import random
from Utils.Utils import innerProduct
import copy
from Params import args

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
        
def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def get_mask_distribution(Models, handler, args,   if_numpy=False, use_tea2drop_edges=False, use_mTea2drop_edges=True, use_adj_mask_aug=False):
    if args.train_middle_model:
        model = Models.middle_teacher
    else:    
        model = Models.student

    adj_mask_tensor = model.adj_mask1_train.detach().flatten() # Flattens input by reshaping it into a one-dimensional tensor
    ori_adj_mask_tensor = adj_mask_tensor
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor_nnz = adj_mask_tensor[nonzero] # 13264 - 2708


    if use_adj_mask_aug:
        if use_mTea2drop_edges:
            Uembs, Iembs = Models.middle_teacher(handler.adj_closure)
            embs = torch.concat([Uembs.detach(), Iembs.detach()], dim=0).detach()

        elif  use_tea2drop_edges:
            Uembs, Iembs = Models.teacher(handler.adj_tea)
            embs = torch.concat([Uembs.detach(), Iembs.detach()], dim=0).detach()

        else:
            if args.train_middle_model:
                embs = Models.middle_teacher(handler.adj_closure, getTogether=True).detach()                
            else:
                embs = Models.student(handler.adj_stu, getTogether=True).detach()
    
        rows = handler.indices[0]
        cols = handler.indices[1]

        usrs_embs = embs[rows]
        items_embs = embs[cols]


        if args.use_tanh2aug:
            adj_mask_aug1 =  args.adj_mask_aug1 * innerProduct(usrs_embs, items_embs).tanh()
        else:
            adj_mask_aug1 =  args.adj_mask_aug1 * innerProduct(usrs_embs, items_embs).sigmoid()


        if args.use_SM_edgeW2aug:
            smTea_adj =  handler.adj_closure.set_value(Models.middle_teacher.adj_mask1_train).to_torch_sparse_coo_tensor()
            stu_adj =  handler.adj_stu.set_value(torch.ones(handler.adj_stu.nnz()).cuda()).to_torch_sparse_coo_tensor()
            adj_mask_aug2_smat = smTea_adj * stu_adj
            adj_mask_aug2 = adj_mask_aug2_smat.coalesce().values()
            # print("Debug ############args.use_SM_edgeW2aug#############, ", args.use_SM_edgeW2aug, type(args.use_SM_edgeW2aug))
            adj_mask_aug = adj_mask_aug1 + args.adj_mask_aug2 * adj_mask_aug2
        else:
            adj_mask_aug = adj_mask_aug1

        if args.adj_weights_abs:
            adj_mask_tensor_aug = (torch.abs(adj_mask_tensor) + adj_mask_aug) * model.adj_mask2_fixed.detach()
        else:
            adj_mask_tensor_aug = (adj_mask_tensor + adj_mask_aug) * model.adj_mask2_fixed.detach()

        ori_adj_mask_tensor_aug = adj_mask_tensor_aug

        nonzero_aug = torch.abs(adj_mask_tensor_aug) > 0
        adj_mask_tensor_nnz_aug = adj_mask_tensor_aug[nonzero_aug] # 13264 - 2708
    else:
        ori_adj_mask_tensor_aug = adj_mask_tensor                
        adj_mask_tensor_nnz_aug = adj_mask_tensor_nnz
    

    embeds = torch.concat([model.uEmbeds, model.iEmbeds], axis=0)
    embeds = embeds.flatten()
    nonzero = torch.abs(embeds) > 0
    emb_mask_tensor_nnz = embeds[nonzero]

    
    if if_numpy:
        return adj_mask_tensor_nnz.detach().numpy(), ori_adj_mask_tensor.detach().numpy(),  adj_mask_tensor_nnz_aug.detach().numpy() ,ori_adj_mask_tensor_aug.detach().numpy(),  emb_mask_tensor_nnz.detach().numpy()
    else:
        return adj_mask_tensor_nnz.detach(), ori_adj_mask_tensor.detach(), adj_mask_tensor_nnz_aug.detach(), ori_adj_mask_tensor_aug.detach() , emb_mask_tensor_nnz.detach()
    


def get_each_mask(mask_weight_tensor, threshold, torch_sparse=False):
    
    if torch_sparse:
        row, col, mask_weight_tensor = mask_weight_tensor.coo()


    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)

    if torch_sparse:
        res = torch_sparse.SparseTensor(row=row,col=col,value=mask)
    
    else:
        res = mask
    return res

def drop_dims(mask_weight_tensor, threshold, torch_sparse=False):
    
    if torch_sparse:
        row, col, mask_weight_tensor = mask_weight_tensor.coo()
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, mask_weight_tensor, zeros)

    if torch_sparse:
        res = torch_sparse.SparseTensor(row=row,col=col,value=mask)    
    else:
        res = mask
    return res


def get_final_mask_epoch(Models, rewind_weight, args, handler,  keep_adj_weights2next_prn=False, use_adj_mask_aug=True, use_adj_random_pruning=False, use_emb_random_pruning=False, epoch=None, prn=None):
    if args.train_middle_model:
        model = Models.middle_teacher
    else:
        model = Models.student

    adj_percent = args.pruning_percent_adj

    emb_percent = args.pruning_percent_emb

    adj_mask_tensor_nnz, ori_adj_mask_tensor,  adj_mask_tensor_nnz_aug, ori_adj_mask_tensor_aug, emb_mask_tensor_nnz =\
        get_mask_distribution(Models, handler, args, if_numpy=False, use_tea2drop_edges=args.use_tea2drop_edges, use_mTea2drop_edges=args.use_mTea2drop_edges, use_adj_mask_aug=use_adj_mask_aug)
                                                 

    if not use_adj_random_pruning:
        if use_adj_mask_aug:
            adj_nnz_total_aug = adj_mask_tensor_nnz_aug.shape[0]    
            # sort
            adj_y_aug, adj_i_aug = torch.sort(adj_mask_tensor_nnz_aug.abs())
                ### get threshold
            adj_thre_index_aug = int(adj_nnz_total_aug * adj_percent)
            adj_thre_aug = adj_y_aug[adj_thre_index_aug]
        else:
            adj_nnz_total = adj_mask_tensor_nnz.shape[0]
            adj_y, adj_i = torch.sort(adj_mask_tensor_nnz.abs())
            adj_thre_index = int(adj_nnz_total * adj_percent)
            adj_thre = adj_y[adj_thre_index]

    if not use_emb_random_pruning:
        emb_nnz_total = emb_mask_tensor_nnz.shape[0]
        emb_y,emb_i = torch.sort(emb_mask_tensor_nnz.abs())
        emb_thre_index = int(emb_nnz_total * emb_percent)
        emb_thre = emb_y[emb_thre_index]



    mask_dict = {}
    ori_adj_mask = model.adj_mask1_train.detach()
    ori_uEmbeds = model.uEmbeds.detach()
    ori_iEmbeds = model.iEmbeds.detach()
    ori_embeds = torch.concat([ori_uEmbeds, ori_iEmbeds], axis=0)



    if keep_adj_weights2next_prn:
        rewind_weight['adj_mask1_train'] = drop_dims(ori_adj_mask, adj_thre, torch_sparse=False)
        rewind_weight['adj_mask2_fixed'] = get_each_mask(ori_adj_mask, adj_thre, torch_sparse=False)#    rewind_weight['adj_mask1_train']
    else:
        if use_adj_random_pruning:        
                ori_adj_fixed_mask_tensor = copy.deepcopy(model.adj_mask2_fixed.detach())
                rewind_weight['adj_mask2_fixed'] = get_each_mask_random(ori_adj_fixed_mask_tensor, ori_adj_fixed_mask_tensor.nonzero(), adj_percent)
                rewind_weight['adj_mask1_train'] = rewind_weight['adj_mask2_fixed']            

        else:
            if use_adj_mask_aug:
                rewind_weight['adj_mask1_train'] = get_each_mask(ori_adj_mask_tensor_aug, adj_thre_aug, torch_sparse=False)
                rewind_weight['adj_mask2_fixed'] = rewind_weight['adj_mask1_train']
            else:
                rewind_weight['adj_mask1_train'] = get_each_mask(ori_adj_mask_tensor, adj_thre, torch_sparse=False)
                rewind_weight['adj_mask2_fixed'] = rewind_weight['adj_mask1_train']


    if use_emb_random_pruning :
        emb_fixed_mask = copy.deepcopy(model.emb_mask2_fixed.detach())
        
        pruned_embs, pruned_fixed_mask = drop_dims_random(ori_embeds, emb_fixed_mask, emb_percent)
        rewind_weight['uEmbeds'] = pruned_embs[:args.user]
        rewind_weight['iEmbeds'] = pruned_embs[args.user:]
        rewind_weight['emb_mask2_fixed'] = pruned_fixed_mask # get_each_mask(ori_embeds, emb_thre, torch_sparse=False) # rewind_weight['emb_mask1_train']

    else:
        rewind_weight['uEmbeds'] = drop_dims(ori_uEmbeds, emb_thre, torch_sparse=False)
        rewind_weight['iEmbeds'] = drop_dims(ori_iEmbeds, emb_thre, torch_sparse=False)        
        #rewind_weight['emb_mask1_train'] = get_each_mask(ori_emb_mask, emb_thre, torch_sparse=False)
        rewind_weight['emb_mask2_fixed'] = get_each_mask(ori_embeds, emb_thre, torch_sparse=False) # rewind_weight['emb_mask1_train']

    
   

    adj_nonzero =  rewind_weight['adj_mask2_fixed'].sum()
    edge_spar = adj_nonzero * 100 / model.adj_mask2_fixed.numel()

    # wei_spar = -1.

    emb_nonzero = rewind_weight['emb_mask2_fixed'].sum()
    emb_all = rewind_weight['emb_mask2_fixed'].numel()

    emb_spar = emb_nonzero * 100 / emb_all

    # return rewind_weight, edge_spar, wei_spar, emb_spar
    return rewind_weight, edge_spar, emb_spar

    
def get_each_mask_random(ori_fixed_mask_tensor, tensor_nonzero, prune_percent):
    
    nnz_elem_total = tensor_nonzero.shape[0]
    pruned_num = int(nnz_elem_total * prune_percent)

    index = random.sample( list(range(nnz_elem_total)) , pruned_num)
    index.sort()
    pruned_tensor_idx = tensor_nonzero[index].tolist() #tensor_nonzero is a 1-D idx tensor of nnz elements

    for i in pruned_tensor_idx:
        ori_fixed_mask_tensor[i] = 0      

    return ori_fixed_mask_tensor

def drop_dims_random(ori_tensor, ori_fixed_mask, prune_percent):
    fixed_mask_nonzero_idx = ori_fixed_mask.nonzero()

    nnz_elem_total = fixed_mask_nonzero_idx.shape[0]
    pruned_num = int(prune_percent * nnz_elem_total)

    index = random.sample(list(range(nnz_elem_total)), pruned_num)
    index.sort()

    pruned_tensor_idx = fixed_mask_nonzero_idx[index].tolist() #tensor_nonzero is a 2-D idx tensor of nnz elements            

    for i, j in pruned_tensor_idx:
        ori_fixed_mask[i][j] = 0  

    masked_tensor = ori_fixed_mask * ori_tensor

    return masked_tensor, ori_fixed_mask


def drop_dims(mask_weight_tensor, threshold, torch_sparse=False):
    
    if torch_sparse:
        row, col, mask_weight_tensor = mask_weight_tensor.coo()

    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, mask_weight_tensor, zeros)

    if torch_sparse:
        res = torch_sparse.SparseTensor(row=row,col=col,value=mask)
    
    else:
        res = mask
    return res


def print_sparsity(model, torch_sparse=False, have_weight=False):

    if torch_sparse:
        adj_nonzero = model.adj_nonzero
        adj_mask_nonzero = model.adj_mask2_fixed.nnz()
        edge_spar = adj_mask_nonzero * 100 / adj_nonzero

    else:
        adj_nonzero = model.adj_nonzero
        adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
        edge_spar = adj_mask_nonzero * 100 / adj_nonzero        
        print(int(adj_mask_nonzero))
        print(torch.nonzero(model.adj_mask2_fixed).shape[0])

    emb_mask_total = model.emb_mask2_fixed.numel()
    emb_mask_nonzero = model.emb_mask2_fixed.sum().item()

    emb_spar = emb_mask_nonzero * 100. / emb_mask_total

    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] | Emb:[{:.2f}%]".format(edge_spar,emb_spar ))    
    print("-" * 100)

    # wei_spar = -1.

    # return edge_spar, wei_spar, emb_spar
    return edge_spar, emb_spar


        
