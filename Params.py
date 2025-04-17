import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parser_loader():
    parser = argparse.ArgumentParser(description='Options')    
    parser.add_argument('--mask_epoch', type=int, default=600, help='epoch number for the training phase with edge weights')
    parser.add_argument('--fix_epoch', type=int, default=600, help='epoch number for the training phase without edge weights, i.e., fixed edge weights')
    parser.add_argument('--epoch_tea', default=600, type=int, help='number of epochs for teacher model')
    parser.add_argument('--use_dynamic_epoch', default=False, type=str2bool,help='whether to train for different training epochs in different pruning steps')
    parser.add_argument('--epoch_list', type=list, default=[0, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600], help="list of epoch numbers for different pruning steps")
    parser.add_argument('--esc_train_prn_run', default=-1, type=int, help= "escape the training for the pruning run 'esc_train_prn_run' ")

    parser.add_argument('--pruning_percent_adj', type=float, default=0.05, help="the pruning ratio for the interaction matrix (adjacent matrix), i.e., the edges for one pruning step")
    parser.add_argument('--pruning_percent_emb', type=float, default=0.2, help="the pruning ratio for the embedding entries for one pruning step")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')

    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--tstBat', default=64, type=int, help='number of users in a testing batch')


    parser.add_argument('--model_save_path',type=str, default='./outModels/model0/teacher_gowalla', help='path to save the trained model')
    parser.add_argument('--his_save_path',type=str, default='./History/teacher_gowalla', help='path to save the training record')
    parser.add_argument('--train_stu_from_break_point', default=False, type=str2bool, help='to train the student model from a certain pruning epoch (breakpoint)')    
    parser.add_argument('--break_point', type=str,default='./outModels/Fmodel5/sm_teacher64ly6Aaug_student64_ly2-reg1e-8-ereg1e-8_lr1e-3_IMP18-ep600-DYsubspcreg1e-4-IMP467-1e-7-loosen3-1adj_mask_aug5e-2-DY2adj_mask_aug-IMP7895e-2-IMP1011-8e-1-NOabs-adjPr0-gowalla_2nd_best_train_98_5.mod',  help='the file name of the breakpoint model')    

    parser.add_argument('--load_model_tea', type=str,default=None, help='model\'s file name to load when training the teacher model from a breakpoint')
    parser.add_argument('--load_his_tea', type=str,default=None, help='training record to load when training the teacher model from a breakpoint')
    
    parser.add_argument('--teacher_model', type=str,default='./inModels/teacher_dim64_layer8_lr4e-4_ep1000_gowalla_reg1.3e-7.mod', help='the file name of the trained teacher model to load when training the intermediate KD model (or student model)')
    parser.add_argument('--middle_teacher_model',type=str, default='./inModels/teacher64-student64aug-gowalla_best_global_new.mod', help='the file name of the trained intermediate KD model to load when training the student model')
    
    
    parser.add_argument('--distill_from_middle_model',type=str2bool, default=True, help='whether to distill from the intermediate KD layer model')
    parser.add_argument('--distill_from_teacher_model', type=str2bool,default=False, help='whether to distill directly from the teacher model')
    parser.add_argument('--train_middle_model', type=str2bool,default=False, help='train the intermediate KD model')

    
    # Decay regularizer
    parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')
    parser.add_argument('--use_ereg', default=True, type=str2bool, help='whether to use weight decay regularizer for edge weights')
    parser.add_argument('--ereg', default=1e-8, type=float, help='weight-decay regularizer weight for edge weights')

    # Bilevel alignment.
    parser.add_argument('--use_emb_level_kd', type=str2bool,default=True,  help='whether to use the embedding level KD')    
    parser.add_argument('--use_pre_level_kd', type=str2bool,default=True,  help='whether to use the prediction level KD')    
    parser.add_argument('--cdreg', default=1e-2, type=float, help='embedding-level contrastive distillation reg weight')
    parser.add_argument('--softreg', default=1., type=float, help='prediction-level distillation reg weight')
    

    # Importance distillation    
    parser.add_argument('--use_adj_mask_aug', type=str2bool, default=False,   help='whether to use adj_mask_aug1')    
    parser.add_argument('--use_SM_edgeW2aug', type=str2bool, default=False,   help='whether to use adj_mask_aug2')   
    parser.add_argument('--adj_mask_aug1', default=5e-2, type=float, help='the significance of the intermediate teacher\'s edge weights for the importance distillation')
    parser.add_argument('--adj_mask_aug2', default=1, type=float, help='the significance of the intermediate teacher\'s predictions for the importance distillation')
    parser.add_argument('--use_dynamic_adj_mask_aug1', default=False, type=str2bool, help='whether to use adaptive adj_mask_aug1 weights in different pruning steps')
    parser.add_argument('--use_dynamic_adj_mask_aug2', default=False, type=str2bool, help='whether to use adaptive adj_mask_aug2 weights in different pruning steps')                                                    
    parser.add_argument('--adj_mask_aug_list1', default=[5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2,5e-2, 5e-2   ], type=list, help='adaptive weights\' list of the adj_mask_aug1')                                                        
    parser.add_argument('--adj_mask_aug_list2', default=[1,  1 ,  1,  1,  1,  1, 1.,  1.,  1., 1., 1.,  1., 1., 1.,  1., 1., 1., 1., 1., 1., 1., 1., 1.   ], type=list, help='adaptive weights\' list of the adj_mask_aug2')
    parser.add_argument('--adj_weights_abs', default=False, type=str2bool, help='whether to use the absolute value of edge weights for adj_mask_aug1 term, False by default')
    parser.add_argument('--use_tanh2aug', default=False, type=str2bool, help='whether to use tanh instead of sigmoid for importance distillation, False by default') 
    
    # GCN 
    parser.add_argument('--latdim', default=64, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    

    # Interaction adjacent matrix augmentation
    parser.add_argument('--adj_aug', default=True,type=str2bool, help='whether to augment the original interaction adjacent matrix')
    parser.add_argument('--adj_aug_layer', default=1, type=int,help='use (adj_aug_layer+1)-hop edges to augment the original interaction adjacent matrix')    
    parser.add_argument('--adj_aug_sample', default=False,type=str2bool, help='whether to sample some edges from the augmented edges to use')
    parser.add_argument('--adj_aug_sample_random', default=False, type=str2bool,help='whether to sample the the augmented edges randomly')
    parser.add_argument('--adj_aug_sample_ratio', default=0.9, type=float, help='the retention ratio of the sampling process')


    # Uniformity Constraint
    parser.add_argument('--use_subspcreg', default=True, type=str2bool, help='whether to use subspace self-contrastive uniformity constraint for embeddings')
    parser.add_argument('--subspcreg', default=1e-4, type=float, help='subspace self-contrastive uniformity constraint weight')
    parser.add_argument('--subspcreg_version', default='v3', type=str, help='the version of subspace uniformity constraint, v1,v2,or,v3')    
    parser.add_argument('--use_ly0emb4subspcreg', default=False, type=str2bool, help='whether to use layer=0 embeddings for subspace self-contrastive uniformity constraint, False by default')        
    parser.add_argument('--use_dynamic_subspcreg', default=False, type=str2bool, help='use adaptive subspace self-contrastive reg weights for embeddings in different pruning steps')    
    parser.add_argument('--subspcreg_list', default=[1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4 ], type=list, help="weights\' list of the subspace self-contrastive uniformity constraint for different pruning steps")
    parser.add_argument('--use_contr_deno', default=True, type=str2bool, help='whether to use auxiliary denominator for the uniformity constraint, True by default')
    parser.add_argument('--screg', default=1., type=float, help='auxiliary denominator weight')
    

    # The applicability scope of uniformity constraint
    parser.add_argument('--hyper_contr_start_prn_run', default=2, type=int, help='the pruning step to start using subspace self-contrastive uniformity constraint')
    parser.add_argument('--hyper_contr_end_prn_run', default=13, type=int, help='the pruning step to stop using subspace self-contrastive uniformity constraint')    
    parser.add_argument('--hyper_contr_resample', default=False, type=str2bool, help='whether to sample some nodes to use subspace self-contrastive uniformity constraint')    
    parser.add_argument('--hyper_contr_resample_ratio', default=0.5, type=float, help='the retention ratio of the sampling process')        
    parser.add_argument('--hyper_contr_loosen_factor', default=[0,0,0,2,3,4,4,3,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0], type=list, help='adaptive threshold hyperparameters for similarity relaxation in different pruning steps ')    


    # Pruning
    parser.add_argument('--keep_adj_weights2next_prn', default=False, type=str2bool, help='whether to keep the learned edge weights to the next pruning step, False by default')                
    parser.add_argument('--use_mTea2drop_edges', default=True, type=str2bool, help='whether to use the intermediate KD model to augment edge dropping, True by default')    
    parser.add_argument('--use_tea2drop_edges', default=False, type=str2bool, help='whether to use the teacher model to augment edge dropping, False by default')    
    parser.add_argument('--use_adj_random_pruning', default=False, type=str2bool, help='whether to drop edges randomly')    
    parser.add_argument('--use_emb_random_pruning', default=False, type=str2bool, help='whether to drop embedding dimensions randomly')        
    
    
    # Get the model without edge weights
    parser.add_argument('--run_no_adj_weights', default=False, type=str2bool, help='whether to run the model without the learnable edge weights')    


    
    parser.add_argument('--topk', default=20, type=int, help='K of top K when testing')
    parser.add_argument('--topRange', default=100000, type=int, help='adaptive pick range')
    parser.add_argument('--tempsoft', default=0.03, type=float, help='temperature for prediction-level KD')
    parser.add_argument('--tempcd', default=0.1, type=float, help='temperature for embedding-level KD')
    parser.add_argument('--tempsc', default=1, type=float, help='temperature for uniformity constraint\'s auxiliary denominator')
    parser.add_argument('--tempsubsp', default=1, type=float, help='temperature for subspace self-contrastive uniformity constraint')
    parser.add_argument('--use_dynamic_tempsubsp', default=False, type=str2bool, help='whether to use adaptive temperature for subspace self-contrastive uniformity constraint in different pruning steps')
    parser.add_argument('--tempsubsp_list', default=[1,   1,    1,  1,    1,  1  , 1,  1, 1, 1,  1, 1,   1 ,  1 ,  1,   1, 1,  1,  1,  1,  1,   1,  1, 1 ], type=list, help='adaptive temperatures for the subspace self-contrastive uniformity constraint in different pruning steps')    
   

    parser.add_argument('--dataset', default='gowalla', type=str, help='name of dataset')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='2', type=str, help='indicates which gpu to use')
    
    
    parser.add_argument('--train_2nd_best_gap', default=0.99, type=float, help='the allowable gap between the best result and the 2nd best')
    parser.add_argument('--fixed_2nd_best_gap', default=0.99, type=float, help='the allowable gap between the best result and the 2nd best without edge weights')    
    parser.add_argument('--PRUNING_START', default=1, type=int, help='the starting pruning epoch')
    parser.add_argument('--PRUNING_END', default=13, type=int, help='the ending pruning epoch')


    args = parser.parse_args()
    return args

args = parser_loader()
#assert args.distill_from_middle_model != args.train_middle_model, "train_middle_model and distill_from_middle_model shoud be different"
#assert args.adj_aug == args.train_middle_model, "train_middle_model and adj_aug shoud be identical"

assert (args.train_middle_model and args.adj_aug) or (not args.train_middle_model), "adj_aug must be True if train_middle_model is True"

assert (args.distill_from_middle_model != args.distill_from_teacher_model) , "distill_from_teacher_model and distill_from_middle_model shoud be different"