CUDA_VISIBLE_DEVICES=2 python pretrainTeacher.py \
    --dataset='gowalla' \
    --epoch_tea=1000 \
    --lr=0.0004 \
    --reg=1.3e-7 \
    --latdim=64 \
    --gnn_layer=8 \
    --model_save_path=./outModels/gowalla/example3/checkpoints/tea_ckpts  \
    --his_save_path=./outModels/gowalla/example3/history/tea_his  | tee ./logs/teacher_gowalla_example3_log