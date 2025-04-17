import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import pruning_gcn
import copy
import warnings

import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import LightGCN, LightGCN_sp
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
from Model import Model

import setproctitle
import  random


from DataHandler import DataHandler

warnings.filterwarnings('ignore')
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


def run_fix_mask(args, prn_run, rewind_weight_mask, coach, global_best_acc, global_best_acc_fixed, global_2nd_best_acc_fixed,  global_best_acc_train, global_2nd_best_acc_train):

    pruning_gcn.setup_seed(args.seed)
    
    coach.prepareModel()
    log('Model Prepared')
    
    if args.train_middle_model:
        to_train_model = coach.model.middle_teacher
    else:
        to_train_model = coach.model.student
    
    to_train_model.load_state_dict(rewind_weight_mask)

    edge_spar, emb_spar = pruning_gcn.print_sparsity(to_train_model)

    for name, param in to_train_model.named_parameters():
        '''
        print("###########################parameters name################################################")
        print(name)
        '''
        if 'mask' in name:
            '''
            if name == 'adj_mask1_train':
                continue
            '''
            param.requires_grad = False


    to_train_model.cuda()
    coach.run(global_best_acc, global_best_acc_fixed, global_2nd_best_acc_fixed, global_best_acc_train, global_2nd_best_acc_train ,prn_run=prn_run ,edge_spar=edge_spar,  emb_spar=emb_spar ,rewind_weight=rewind_weight, train_mask=False)


def run_get_mask(args, prn_run, rewind_weight_mask, coach, global_best_acc, global_best_acc_fixed, global_2nd_best_acc_fixed, global_best_acc_train, global_2nd_best_acc_train):

    pruning_gcn.setup_seed(args.seed)
    coach.prepareModel()
    log('Model Prepared')
   
    if args.train_middle_model:
        to_train_model = coach.model.middle_teacher
    else:
        to_train_model = coach.model.student

    if rewind_weight_mask is not None:
        to_train_model.load_state_dict(rewind_weight_mask)
    
    edge_spar, emb_spar = pruning_gcn.print_sparsity(to_train_model)
    
    to_train_model.cuda()
    
    rewind_weight = copy.deepcopy(to_train_model.state_dict())
    rewind_weight =  coach.run(global_best_acc, global_best_acc_fixed, global_2nd_best_acc_fixed, global_best_acc_train, global_2nd_best_acc_train,   prn_run=prn_run ,edge_spar=edge_spar, emb_spar=emb_spar ,rewind_weight=rewind_weight)

    return rewind_weight

class Coach:
    def __init__(self, handler):
        self.handler = handler
        

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save, epoch=None):
        ret = 'Epoch %d/%d, %s: ' % (ep, epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self, global_best_acc,  global_best_acc_fixed=None, global_2nd_best_acc_fixed=None, global_best_acc_train=None, global_2nd_best_acc_train=None    ,prn_run=None, edge_spar=None, emb_spar=None ,rewind_weight=None, train_mask=True, only_save_fixed_mask=True):

        best_acc = { 'train or fixed': 'train', 'PRN': -1, 'Recall': 0, 'NDCG' : 0, 'Epoch':0, 'edge_spar':1.0,'emb_spar':1.0}            

        if train_mask:
            epoch = args.mask_epoch
        else:
            epoch = args.fix_epoch
        stloc = 0
        for ep in range(stloc, epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            if args.esc_train_prn_run != prn_run:
                reses_tr = self.trainEpoch(train_mask=train_mask, prn_run = prn_run, emb_spar=emb_spar)
                log(self.makePrint('Train', ep, reses_tr, tstFlag, epoch=epoch))

            '''
            if args.train_middle_model:
                to_train_model = self.model.middle_teacher
            else:
                to_train_model = self.model.student            
            print("######################to_train_model.adj_mask1_train################################")
            print(to_train_model.adj_mask1_train)            
            print("######################to_train_model.emb_mask2_fixed################################")
            print(to_train_model.emb_mask2_fixed.flatten())
            '''

            if tstFlag:
                with torch.no_grad():
                    reses_tst = self.testEpoch()             
                    if reses_tst['Recall']  >= best_acc['Recall']:
                        best_acc['Recall'] = reses_tst['Recall']
                        best_acc['NDCG'] = reses_tst['NDCG']
                        best_acc['Epoch'] = ep
                        if train_mask and (args.distill_from_middle_model or args.distill_from_teacher_model ) and (not args.train_middle_model):
                        # if train_mask and (args.distill_from_middle_model or args.distill_from_teacher_model ) :
                            rewind_weight, _, _ =  pruning_gcn.get_final_mask_epoch(self.model, rewind_weight, args, self.handler,  keep_adj_weights2next_prn=args.keep_adj_weights2next_prn, use_adj_mask_aug=args.use_adj_mask_aug, use_adj_random_pruning=args.use_adj_random_pruning, use_emb_random_pruning=args.use_emb_random_pruning, epoch=ep, prn=prn_run )                        
                        
                        if (reses_tst['Recall']  >= global_best_acc['Recall']):
                            global_best_acc['Recall'] = reses_tst['Recall']
                            global_best_acc['NDCG'] = reses_tst['NDCG']
                            global_best_acc['Epoch'] = ep
                            global_best_acc['PRN'] = prn_run
                            global_best_acc['edge_spar'] = edge_spar
                            global_best_acc['emb_spar'] = emb_spar
                            if train_mask:
                                global_best_acc['train or fixed'] = "Train Mask"
                            else:
                                global_best_acc['train or fixed'] = "Fixed Mask"
                            self.saveHistory(epoch, '_best_global_' +  str(int(edge_spar)) + '_' + str(int(emb_spar)))

                        if (reses_tst['Recall']  >= global_best_acc_fixed['Recall']) and  (not train_mask):
                            global_best_acc_fixed['Recall'] = reses_tst['Recall']
                            global_best_acc_fixed['NDCG'] = reses_tst['NDCG']
                            global_best_acc_fixed['Epoch'] = ep
                            global_best_acc_fixed['PRN'] = prn_run
                            global_best_acc_fixed['edge_spar'] = edge_spar
                            global_best_acc_fixed['emb_spar'] = emb_spar                        
                            global_best_acc_fixed['train or fixed'] = "Fixed Mask"                        
                            self.saveHistory(epoch,'_best_wo_edge_wei_' +  str(int(edge_spar)) + '_' + str(int(emb_spar)))

                        if (reses_tst['Recall']>= (global_best_acc_fixed['Recall'] * (1.-args.fixed_2nd_best_gap) ) ) and (not train_mask) :
                            global_2nd_best_acc_fixed['Recall'] = reses_tst['Recall']
                            global_2nd_best_acc_fixed['NDCG'] = reses_tst['NDCG']
                            global_2nd_best_acc_fixed['Epoch'] = ep
                            global_2nd_best_acc_fixed['PRN'] = prn_run
                            global_2nd_best_acc_fixed['edge_spar'] = edge_spar
                            global_2nd_best_acc_fixed['emb_spar'] = emb_spar                        
                            global_2nd_best_acc_fixed['train or fixed'] = "Fixed Mask"                        
                            self.saveHistory(epoch,'_2nd_best_wo_edge_wei_' +  str(int(edge_spar)) + '_' + str(int(emb_spar)))



                        if (reses_tst['Recall']  >= global_best_acc_train['Recall']) and  ( train_mask):
                            global_best_acc_train['Recall'] = reses_tst['Recall']
                            global_best_acc_train['NDCG'] = reses_tst['NDCG']
                            global_best_acc_train['Epoch'] = ep
                            global_best_acc_train['PRN'] = prn_run
                            global_best_acc_train['edge_spar'] = edge_spar
                            global_best_acc_train['emb_spar'] = emb_spar                        
                            global_best_acc_train['train or fixed'] = "Train Mask"                        
                            self.saveHistory(epoch,'_best_w_edge_wei_' +  str(int(edge_spar)) + '_' + str(int(emb_spar)))

                        if (reses_tst['Recall']>= (global_best_acc_train['Recall'] * (1. - args.train_2nd_best_gap) ) ) and ( train_mask) :
                            global_2nd_best_acc_train['Recall'] = reses_tst['Recall']
                            global_2nd_best_acc_train['NDCG'] = reses_tst['NDCG']
                            global_2nd_best_acc_train['Epoch'] = ep
                            global_2nd_best_acc_train['PRN'] = prn_run
                            global_2nd_best_acc_train['edge_spar'] = edge_spar
                            global_2nd_best_acc_train['emb_spar'] = emb_spar                        
                            global_2nd_best_acc_train['train or fixed'] = "Train Mask"                        
                            self.saveHistory(epoch, '_2nd_best_w_edge_wei_' +  str(int(edge_spar)) + '_' + str(int(emb_spar)))


                    log(self.makePrint('Test', ep, reses_tst, tstFlag, epoch=epoch))

            if train_mask:
                print(f"Train Edge Weights & Emb. before Pruning Run PRN:[{prn_run}] (Dataset {args.dataset} Get Mask) Epoch:[{ep}/{epoch}], Recall:[{reses_tst['Recall']:.5f}] NDCG:[{reses_tst['NDCG']:.5f}] | Best Recall:[{best_acc['Recall']:.5f}] Best NDCG:[{best_acc['NDCG']:.5f}] at Epoch:[{best_acc['Epoch']}] | Adj:[{edge_spar:.5f}%] Emb:[{emb_spar:.5f}%]", flush=True)
            else:
                print(f"Train Emb. with Fixed Edge Weights before Pruning Run PRN:[{prn_run}] (Dataset {args.dataset} Fixed Mask) Epoch:[{ep}/{epoch}], Recall:[{reses_tst['Recall']:.5f}] NDCG:[{reses_tst['NDCG']:.5f}] | Best for now Recall:[{best_acc['Recall']:.5f}] Best NDCG:[{best_acc['NDCG']:.5f}] at Epoch:[{best_acc['Epoch']}] | Adj:[{edge_spar:.5f}%] Emb:[{emb_spar:.5f}%]",flush=True)
            
                print(f"Train without Edge Weights before Pruning Run PRN:[{prn_run}] (Dataset {args.dataset} Fixed Mask) Epoch:[{ep}/{epoch}], Best for Global Recall:[{global_best_acc['Recall']:.5f}] Best NDCG:[{global_best_acc['NDCG']:.5f}] at Epoch:[{global_best_acc['Epoch']}] in PRN:[{global_best_acc['PRN']}] with [{global_best_acc['train or fixed']}] | Best Adj:[{global_best_acc['edge_spar']:.5f}%] Best Emb:[{global_best_acc['emb_spar']:.5f}%]",flush=True)

            print(f"Best for Global Recall:[{global_best_acc['Recall']:.5f}] Best NDCG:[{global_best_acc['NDCG']:.5f}] at Epoch:[{global_best_acc['Epoch']}] in PRN:[{global_best_acc['PRN']}] with [{global_best_acc['train or fixed']}] | Best Adj:[{global_best_acc['edge_spar']:.5f}%] Best Emb:[{global_best_acc['emb_spar']:.5f}%]",flush=True)
            
            print(f"Best with Edge Weights, Recall:[{global_best_acc_train['Recall']:.5f}] Best NDCG:[{global_best_acc_train['NDCG']:.5f}] at Epoch:[{global_best_acc_train['Epoch']}] in PRN:[{global_best_acc_train['PRN']}] with [{global_best_acc_train['train or fixed']}] | Best Adj:[{global_best_acc_train['edge_spar']:.5f}%] Best Emb:[{global_best_acc_train['emb_spar']:.5f}%]",flush=True)
            print(f"2nd Best with Edge Weights, Recall:[{global_2nd_best_acc_train['Recall']:.5f}] Best NDCG:[{global_2nd_best_acc_train['NDCG']:.5f}] at Epoch:[{global_2nd_best_acc_train['Epoch']}] in PRN:[{global_2nd_best_acc_train['PRN']}] with [{global_2nd_best_acc_train['train or fixed']}] | Best Adj:[{global_2nd_best_acc_train['edge_spar']:.5f}%] Best Emb:[{global_2nd_best_acc_train['emb_spar']:.5f}%]",flush=True)
            
            print(f"Best without Edge Weights,  Recall:[{global_best_acc_fixed['Recall']:.5f}] Best NDCG:[{global_best_acc_fixed['NDCG']:.5f}] at Epoch:[{global_best_acc_fixed['Epoch']}] in PRN:[{global_best_acc_fixed['PRN']}] with [{global_best_acc_fixed['train or fixed']}] | Best Adj:[{global_best_acc_fixed['edge_spar']:.5f}%] Best Emb:[{global_best_acc_fixed['emb_spar']:.5f}%]",flush=True)
            print(f"2nd Best without Edge Weights,[{global_2nd_best_acc_fixed['Recall']:.5f}] Best NDCG:[{global_2nd_best_acc_fixed['NDCG']:.5f}] at Epoch:[{global_2nd_best_acc_fixed['Epoch']}] in PRN:[{global_2nd_best_acc_fixed['PRN']}] with [{global_2nd_best_acc_fixed['train or fixed']}] | Best Adj:[{global_2nd_best_acc_fixed['edge_spar']:.5f}%] Best Emb:[{global_2nd_best_acc_fixed['emb_spar']:.5f}%]",flush=True)
                
            print()

        reses = self.testEpoch()
        log(self.makePrint('Test', epoch, reses, True, epoch=epoch))
        if not train_mask:
            print(f"syd final: PRN[{prn_run}] (GCN {args.dataset} FIX Mask) | Best Recall:[{best_acc['Recall']:.5f}] Best NDCG:[{best_acc['NDCG']:.5f}] at Epoch:[{best_acc['Epoch']:.5f}] | Adj:[{edge_spar:.5f}%] Emb:[{emb_spar:.5f}%]", flush=True)
        #self.saveHistory()

        print(f"Best for Global Recall:[{global_best_acc['Recall']:.5f}] Best NDCG:[{global_best_acc['NDCG']:.5f}] at Epoch:[{global_best_acc['Epoch']}] in PRN:[{global_best_acc['PRN']}] with [{global_best_acc['train or fixed']}] | Best Adj:[{global_best_acc['edge_spar']:.5f}%] Best Emb:[{global_best_acc['emb_spar']:.5f}%]",flush=True)
        print(f"Best without Edge Weights,  Recall:[{global_best_acc_fixed['Recall']:.5f}] Best NDCG:[{global_best_acc_fixed['NDCG']:.5f}] at Epoch:[{global_best_acc_fixed['Epoch']}] in PRN:[{global_best_acc_fixed['PRN']}] with [{global_best_acc_fixed['train or fixed']}] | Best Adj:[{global_best_acc_fixed['edge_spar']:.5f}%] Best Emb:[{global_best_acc_fixed['emb_spar']:.5f}%]",flush=True)
        print(f"2nd Best without Edge Weights,[{global_2nd_best_acc_fixed['Recall']:.5f}] Best NDCG:[{global_2nd_best_acc_fixed['NDCG']:.5f}] at Epoch:[{global_2nd_best_acc_fixed['Epoch']}] in PRN:[{global_2nd_best_acc_fixed['PRN']}] with [{global_2nd_best_acc_fixed['train or fixed']}] | Best Adj:[{global_2nd_best_acc_fixed['edge_spar']:.5f}%] Best Emb:[{global_2nd_best_acc_fixed['emb_spar']:.5f}%]",flush=True)
        return rewind_weight    
    def prepareModel(self):        
        if args.train_middle_model:
            teacher = t.load( args.teacher_model )['model'].cuda()            
            middle_teacher = LightGCN_sp(self.handler.adj_closure).cuda()            
            student = None

        elif args.distill_from_middle_model:
            teacher = None
            tmp = t.load(args.middle_teacher_model)['model']
            if hasattr(tmp, 'middle_teacher'):
                middle_teacher = tmp.middle_teacher.cuda()       
                print("Use the middle teacher (Intermediate KD model) to direct the training of student")   
            elif hasattr(tmp, 'student'):
                middle_teacher = tmp.student.cuda() 
                print("A trained student model works as the middle teacher (Intermediate KD model)")                   
            else:
                middle_teacher = None
                assert middle_teacher!=None, "No middle teacher (Intermediate KD model) is given"
            del tmp

            if args.train_stu_from_break_point:                                        
                tmp = t.load(args.break_point)['model']
                student = tmp.student.cuda() 
                del tmp
            else:
                student = LightGCN_sp(self.handler.adj_stu).cuda()

        else:
            teacher = t.load( args.teacher_model )['model'].cuda()
            middle_teacher = None
            if args.train_stu_from_break_point:                                        
                tmp = t.load(args.break_point)['model']
                student = tmp.student.cuda() 
                del tmp
            else:
                student = LightGCN_sp(self.handler.adj_stu).cuda()
                

        self.model = Model(teacher, student, middle_teacher).cuda()
        
        if args.train_middle_model:
            self.opt = t.optim.Adam(self.model.middle_teacher.parameters(), lr=args.lr, weight_decay=0)    
        else:
            self.opt = t.optim.Adam(self.model.student.parameters(), lr=args.lr, weight_decay=0)    
    
    
    def trainEpoch(self, train_mask=True, prn_run=0, emb_spar=1.):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        for i, tem in enumerate(trnLoader):
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda() 
            negs = negs.long().cuda()
            loss, losses = self.model.calcLoss(self.handler.adj_tea, self.handler.adj_stu, self.handler.adj_closure ,ancs, poss, negs, self.opt,  prn_run = prn_run, train_flag=train_mask, emb_spar=emb_spar)
            epLoss += loss.item()
            epPreLoss += losses['mainLoss'].item()
            regLoss = losses['regLoss'].item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()            
            log('Step %d/%d: loss = %.3f, regLoss = %.3f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

  
    def testEpoch(self):
            tstLoader = self.handler.tstLoader
            epRecall, epNdcg = [0] * 2
            i = 0
            num = tstLoader.dataset.__len__()
            steps = num // args.tstBat
            for usr, trnMask in tstLoader:
                i += 1
                usr = usr.long().cuda()
                trnMask = trnMask.cuda()
                if args.train_middle_model:
                    allPreds = self.model.middle_teacher.testPred(usr, trnMask, self.handler.adj_closure)
                else:
                    allPreds = self.model.student.testPred(usr, trnMask, self.handler.adj_stu)
                _, topLocs = t.topk(allPreds, args.topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
                epRecall += recall
                epNdcg += ndcg
                log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
            ret = dict()
            ret['Recall'] = epRecall / num
            ret['NDCG'] = epNdcg / num
            return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig =0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg


    def saveHistory(self, epoch , postfix=None):
        if epoch == 0:
            return
        with open( args.his_save_path + postfix +'.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content,  args.model_save_path + postfix + '.mod')
        log('Model Saved: %s' % args.model_save_path)


    def loadTeacher(self):
        ckp = t.load('./inModels/teacher_' + args.teacher_model + '.mod')
        teacher = ckp['model']
        return teacher


def mask_multiple_embeddings(rewind_weight):
    rewind_weight['uEmbeds'] = rewind_weight['uEmbeds'] * rewind_weight['ori_emb_mask'][:args.user].cuda()
    rewind_weight['iEmbeds'] = rewind_weight['iEmbeds'] * rewind_weight['ori_emb_mask'][args.user:].cuda()
    del rewind_weight['ori_emb_mask']
    del rewind_weight['ori_adj_mask']

    return rewind_weight

def exit_and_print_results_for_now(signum=None, frame=None):

    print("##################################Best results for now##############################################")
    print(f"Best for Global Recall:[{global_best_acc['Recall']:.5f}] Best NDCG:[{global_best_acc['NDCG']:.5f}] at Epoch:[{global_best_acc['Epoch']}] in PRN:[{global_best_acc['PRN']}] with [{global_best_acc['train or fixed']}] | Best Adj:[{global_best_acc['edge_spar']:.5f}%] Best Emb:[{global_best_acc['emb_spar']:.5f}%]",flush=True)
   
    print(f"Best with Edge Weights, Recall:[{global_best_acc_train['Recall']:.5f}] Best NDCG:[{global_best_acc_train['NDCG']:.5f}] at Epoch:[{global_best_acc_train['Epoch']}] in PRN:[{global_best_acc_train['PRN']}] with [{global_best_acc_train['train or fixed']}] | Best Adj:[{global_best_acc_train['edge_spar']:.5f}%] Best Emb:[{global_best_acc_train['emb_spar']:.5f}%]",flush=True)
    print(f"2nd Best with Edge Weights, Recall:[{global_2nd_best_acc_train['Recall']:.5f}] Best NDCG:[{global_2nd_best_acc_train['NDCG']:.5f}] at Epoch:[{global_2nd_best_acc_train['Epoch']}] in PRN:[{global_2nd_best_acc_train['PRN']}] with [{global_2nd_best_acc_train['train or fixed']}] | Best Adj:[{global_2nd_best_acc_train['edge_spar']:.5f}%] Best Emb:[{global_2nd_best_acc_train['emb_spar']:.5f}%]",flush=True)
   
    print(f"Best without Edge Weights,  Recall:[{global_best_acc_fixed['Recall']:.5f}] Best NDCG:[{global_best_acc_fixed['NDCG']:.5f}] at Epoch:[{global_best_acc_fixed['Epoch']}] in PRN:[{global_best_acc_fixed['PRN']}] with [{global_best_acc_fixed['train or fixed']}] | Best Adj:[{global_best_acc_fixed['edge_spar']:.5f}%] Best Emb:[{global_best_acc_fixed['emb_spar']:.5f}%]",flush=True)
    print(f"2nd Best without Edge Weights,[{global_2nd_best_acc_fixed['Recall']:.5f}] Best NDCG:[{global_2nd_best_acc_fixed['NDCG']:.5f}] at Epoch:[{global_2nd_best_acc_fixed['Epoch']}] in PRN:[{global_2nd_best_acc_fixed['PRN']}] with [{global_2nd_best_acc_fixed['train or fixed']}] | Best Adj:[{global_2nd_best_acc_fixed['edge_spar']:.5f}%] Best Emb:[{global_2nd_best_acc_fixed['emb_spar']:.5f}%]",flush=True)    
    #exit()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu    
    logger.saveDefault = True

    pruning_gcn.setup_seed(args.seed)
    t.manual_seed(args.seed)
    t.cuda.manual_seed_all(args.seed)
    t.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    pruning_gcn.print_args(args)

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.prepareModel()

    if args.train_middle_model:
        init_parameters = copy.deepcopy(coach.model.middle_teacher.state_dict())
    else:
        init_parameters = copy.deepcopy(coach.model.student.state_dict())
    rewind_weight = init_parameters
    ori_init_parameters = copy.deepcopy(init_parameters)
    layer_num = args.gnn_layer
    
    '''
    print("###############rewind_weight####################")
    print(rewind_weight.keys())
    '''

    global_best_acc = {'Recall': 0, 'NDCG' : 0, 'Epoch':0, 'PRN':0 , 'train or fixed': 'default','edge_spar':1.0, 'emb_spar':1.0}
    global_best_acc_fixed = {'Recall': 0, 'NDCG' : 0, 'Epoch':0, 'PRN':0 , 'train or fixed': 'default','edge_spar':1.0, 'emb_spar':1.0}
    global_2nd_best_acc_fixed = {'Recall': 0, 'NDCG' : 0, 'Epoch':0, 'PRN':0 , 'train or fixed': 'default','edge_spar':1.0,  'emb_spar':1.0}

    global_best_acc_train = {'Recall': 0, 'NDCG' : 0, 'Epoch':0, 'PRN':0 , 'train or fixed': 'default','edge_spar':1.0,  'emb_spar':1.0}
    global_2nd_best_acc_train = {'Recall': 0, 'NDCG' : 0, 'Epoch':0, 'PRN':0 , 'train or fixed': 'default','edge_spar':1.0, 'emb_spar':1.0}

    try:
        for prn in range(args.PRUNING_START, args.PRUNING_END+1):
            if args.use_dynamic_adj_mask_aug1:  
                args.adj_mask_aug1 = args.adj_mask_aug_list1[prn]

            if args.use_dynamic_adj_mask_aug2:  
                args.adj_mask_aug2 = args.adj_mask_aug_list2[prn]                

            if args.use_dynamic_subspcreg:  
                args.subspcreg = args.subspcreg_list[prn]

            if args.use_dynamic_epoch:
                 args.mask_epoch = args.epoch_list[prn]
                 args.fix_epoch = args.epoch_list[prn]

            if args.use_dynamic_tempsubsp:
                 args.tempsubsp = args.tempsubsp_list[prn]
                 
            if prn == 1 and args.run_no_adj_weights:
                run_fix_mask(args, prn, rewind_weight, coach, global_best_acc, global_best_acc_fixed, global_2nd_best_acc_fixed, global_best_acc_train, global_2nd_best_acc_train)                                            

            rewind_weight = run_get_mask(args, prn, rewind_weight, coach, global_best_acc, global_best_acc_fixed, global_2nd_best_acc_fixed, global_best_acc_train, global_2nd_best_acc_train)
            #rewind_weight = mask_multiple_embeddings(rewind_weight)

            if args.run_no_adj_weights:
                run_fix_mask(args, prn, rewind_weight, coach, global_best_acc, global_best_acc_fixed, global_2nd_best_acc_fixed, global_best_acc_train, global_2nd_best_acc_train)
                        
    except KeyboardInterrupt as e:
        exit_and_print_results_for_now()
    
    exit_and_print_results_for_now()
