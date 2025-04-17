## [WSDM'25] LightGNN: Simple Graph Neural Network for Recommendation
<br/>
<center class="half">
    <img src=https://github.com/user-attachments/assets/9ddc22ff-96f3-4fc2-874f-9af87a91a8a3 width=57.5% /><img src=https://github.com/user-attachments/assets/8e6220a5-23c0-435e-a508-78e2045b5ac0 width=42.5% />
</center>
<br/>
<br/>

![image](https://github.com/user-attachments/assets/86c731fc-c44e-4b6b-ac4b-3f87748c476c)
<br/>
<!-- <center>
<font face="华文琥珀" size="5">
If you find this code useful, please consider giving us a star🌟 Your support is greatly appreciated😊        
</font>    
</center> -->

<br/>

![image](https://github.com/user-attachments/assets/54677ee5-85ad-4020-b39f-f3d8b34a7243)

<!--   华文琥珀
  Centrale Sans Rounded Light 
  Bradley Hand 
  Comic Sans MS -->
## News
- :star2: [2025/01] More examples have been updated, come and give them a try! :rocket::rocket::rocket:
- :star: [2025/01] LightGNN's code has been released. 
- :star: [2024/10] LightGNN has been accepted by _WSDM 2025_ conference.



## 1. Abstract
Graph neural networks (GNNs) have demonstrated superior performance in collaborative recommendation through their ability to conduct high-order representation smoothing, effectively capturing structural information within users' interaction patterns. However, existing GNN paradigms face significant challenges in scalability and robustness when handling large-scale, noisy, and real-world datasets. To address these challenges, we present LightGNN, a lightweight and distillation-based GNN pruning framework designed to substantially reduce model complexity while preserving essential collaboration modeling capabilities. Our LightGNN framework introduces a computationally efficient pruning module that adaptively identifies and removes redundant edges and embedding entries for model compression. The framework is guided by a resource-friendly hierarchical knowledge distillation objective, whose intermediate layer augments the observed graph to maintain performance, particularly in high-rate compression scenarios. Extensive experiments on public datasets demonstrate LightGNN's effectiveness, significantly improving both computational efficiency and recommendation accuracy. Notably, LightGNN achieves an 80% reduction in edge count and 90% reduction in embedding entries while maintaining performance comparable to more complex state-of-the-art baselines. The implementation of our LightGNN framework is available at the github repository: https://github.com/HKUDS/LightGNN.



## 2. Requirements

```
python == 3.9

pytorch == 1.12.1

torch-sparse == 0.6.15

torch-scatter == 2.0.9

scipy == 1.9.3
```
Please refer to *develop-environment.md* for more details on the development environment.

## 3. Quick Start

After preparing the development environment, we provide some demo trained checkpoints of the teacher models and intermediate models (middle teacher models) based on Gowalla, Yelp, and Amazon datasets as examples to facilitate your quick start. Once the development environment is set up, you can run the following commands to quickly start the training process to get familiar with the code.

But before that, you need to download the demo checkpoint files for these examples first. Just run the *download_demo_ckpts.sh* script in the *./inModels* directory to complete the download.
```Bash
git clone https://github.com/HKUDS/LightGNN.git
cd ./LightGNN/inModels
bash download_demo_ckpts.sh
cd ..
```

To get help information about the parameters, you can run the following command.
```Bash
python Main.py --help
```
*Note: All example commands in 3.1 to 3.4 are executed in the root directory of the LightGNN project, and the parameters involved are just examples, not guaranteed to be optimal settings.*

### 3.1 Train the final student model (supervised by the demo intermediate model provided by us)

You can just start training to get the final pruned student model by running the following example commands (based on Gowalla dataset).
```Bash
CUDA_VISIBLE_DEVICES=0 python Main.py \
    --dataset='gowalla' \
    --mask_epoch=300 `#300 is an example; increase it for better recommendation performance` \
    --lr=0.001 \
    --reg=1e-8 \
    --latdim=64 \
    --gnn_layer=2 \
    --adj_aug=True `# Necessary for obtaining the final student` \
    --adj_aug_layer=1  `# Necessary for obtaining the final student`  \
    --use_adj_mask_aug=True `# Necessary for obtaining the final student` \
    --use_SM_edgeW2aug=True  `# Necessary for obtaining the final student` \
    --adj_mask_aug1=0.05 \
    --adj_mask_aug2=1.0 \
    --use_mTea2drop_edges=True `# Necessary for obtaining the final student` \
    --use_tea2drop_edges=False `# Necessary for obtaining the final student` \
    --PRUNING_START=1 \
    --PRUNING_END=13 \
    --model_save_path=./outModels/gowalla/example1/checkpoints/fnl_student_ckpts \
    --his_save_path=./outModels/gowalla/example1/history/fnl_student_his \
    --middle_teacher_model=./inModels/gowalla/example_intermediate_KD_model_dim64_gowalla_ckpt.mod \
    --distill_from_middle_model=True `# Necessary for obtaining the final student` \
    --distill_from_teacher_model=False  `# Necessary for obtaining the final student` \
    --train_middle_model=False `# Necessary for obtaining the final student` \
    --pruning_percent_adj=0.15 \
    --pruning_percent_emb=0.2 | tee ./logs/fnl_student_gowalla_example1_log
```
*"pruning_percent_adj" refers to the proportion of edges discarded in the adjacency matrix of the interaction graph for each pruning run. "pruning_percent_emb" represents the proportion of embedding entries discarded in the embedding matrix for each pruning run.*

### 3.2 Train the intermediate KD model (supervised by the demo original teacher model) as the final teacher model
You can also directly train the intermediate KD model (i.e., the middle teacher model), supervised by the demo original teacher model provided by us. After completion, you can supervise the training of your final student model using this intermediate model trained by yourself (refer to section 3.1). Also take Gowalla dataset as an example.
```Bash
 CUDA_VISIBLE_DEVICES=0 python Main.py \
    --dataset='gowalla' \
    --mask_epoch=500 \
    --lr=0.0004 \
    --reg=1.3e-07 \
    --latdim=64 \
    --gnn_layer=6 \
    --adj_aug=True `#Necessary for training intermediate model` \
    --adj_aug_layer=1 `#Necessary for training intermediate model` \
    --use_adj_mask_aug=False  \
    --use_SM_edgeW2aug=False  `#Necessary for training intermediate model`\
    --PRUNING_START=1 `#Necessary for training intermediate model`\
    --PRUNING_END=1  `#Necessary for training intermediate model` \
    --model_save_path=./outModels/gowalla/example2/checkpoints/mid_tea_ckpts \
    --his_save_path=./outModels/gowalla/example2/history/mid_tea_his \
    --teacher_model=./inModels/gowalla/teacher_model_dim64_gowalla.mod `#Necessary for training intermediate model` \
    --distill_from_middle_model=False `#Necessary for training intermediate model`\
    --distill_from_teacher_model=True `#Necessary for training intermediate model`\
    --train_middle_model=True | tee ./logs/intermediate_gowalla_example2_log `#Necessary for training intermediate model`
```

### 3.3 Train from scratch (train the original teacher model)
Of course, you can also train from scratch, i.e.,  training your own original teacher model. Then, use this trained original teacher model to guide the training of your own intermediate model (see 3.2). Finally, utilize the trained intermediate model to supervise the training of your own final student model (see 3.1). Throughout this process, you can adjust the corresponding parameters to achieve the desired trade-off between sparsity and performance. Also take Gowalla dataset as an example.
```Bash
CUDA_VISIBLE_DEVICES=0 python pretrainTeacher.py \
    --dataset='gowalla' \
    --epoch_tea=1000 \
    --lr=0.0004 \
    --reg=1.3e-7 \
    --latdim=64 \
    --gnn_layer=8 \
    --model_save_path=./outModels/gowalla/example3/checkpoints/tea_ckpts  \
    --his_save_path=./outModels/gowalla/example3/history/tea_his  | tee ./logs/teacher_gowalla_example3_log
```

### 3.4 More Quick-Start Examples
We offer more quick-start examples to help you get started quickly (based on the Gowalla, Yelp, Amazon datasets), and the scripts for these examples are placed in the *./example_scripts/* directory. You can follow the steps below to run these examples. For instance, to run the example in section 3.1, we can alternatively run the following commands in the LightGNN's root directory:

```Bash
cp ./example_scripts/example_train_final_student_lightgnn_gowalla.sh ./
bash example_train_final_student_lightgnn_gowalla.sh
```


# Supplementary Material
See "**Supplementary Material.pdf**" for __more details__ of our LightGNN framework, including detailed *Algorithm*, *Robustness against Noise*, etc.

# Citation
If you find LightGNN useful in your research, applications, or teaching, please kindly cite:
```
@article{chen2025lightgnn,
  title={LightGNN: Simple Graph Neural Network for Recommendation},
  author={Chen, Guoxuan and Xia, Lianghao and Huang, Chao},
  journal={arXiv preprint arXiv:2501.03228},
  year={2025}
}
```
