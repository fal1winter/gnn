CUDA_VISIBLE_DEVICES=0 python Main.py \
    --dataset='yelp' \
    --mask_epoch=600 `#300 is an example; increase it for better recommendation performance` \
    --lr=0.0001 \
    --reg=5e-7 \
    --ereg=1e-9 \
    --latdim=64 \
    --gnn_layer=2 \
    --adj_aug=True `# Necessary for obtaining the final student` \
    --adj_aug_layer=1  `# Necessary for obtaining the final student`  \
    --use_adj_mask_aug=True `# Necessary for obtaining the final student` \
    --use_SM_edgeW2aug=True  `# Necessary for obtaining the final student` \
    --adj_mask_aug1=1.0 \
    --adj_mask_aug2=0.05 \
    --use_mTea2drop_edges=True `# Necessary for obtaining the final student` \
    --use_tea2drop_edges=False `# Necessary for obtaining the final student` \
    --PRUNING_START=1 \
    --PRUNING_END=13 \
    --model_save_path=./outModels/yelp/example1/checkpoints/fnl_student_ckpts \
    --his_save_path=./outModels/yelp/example1/history/fnl_student_his \
    --middle_teacher_model=./inModels/yelp/example_intermediate_KD_model_dim64_yelp.mod \
    --distill_from_middle_model=True `# Necessary for obtaining the final student` \
    --distill_from_teacher_model=False  `# Necessary for obtaining the final student` \
    --train_middle_model=False `# Necessary for obtaining the final student` \
    --pruning_percent_adj=0.05 \
    --pruning_percent_emb=0.2 | tee ./logs/fnl_student_yelp_example1_log
