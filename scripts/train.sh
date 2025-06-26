#!/bin/bash -l
#SBATCH --gres=gpu:a40:4
#SBATCH --time=24:00:00

# for web access
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

cd $HOME/tabicl-survival
module load cuda/12.6.2
module load python/3.12-conda
source $WORK/venvs/tabicl-survival/bin/activate

# from sample scripts
torchrun --standalone --nproc_per_node=1 /path/to/tabicl/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name Stage1 \
            --wandb_dir /my/wandb/dir \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 100000 \
            --batch_size 512 \
            --micro_batch_size 4 \
            --lr 1e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --prior_device cpu \
            --batch_size_per_gp 4 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir /my/stage1/checkpoint/dir \
            --save_temp_every 50 \
            --save_perm_every 5000