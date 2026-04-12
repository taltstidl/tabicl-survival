#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00

# for web access
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

cd $HOME/tabicl-survival
module load cuda/12.6.1
module load python/3.12-conda
source $WORK/venvs/tabicl-survival/bin/activate

# from sample scripts
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --standalone --nproc_per_node=1 src/tabicl/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name Stage2 \
            --wandb_dir wandb \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 2000 \
            --batch_size 512 \
            --micro_batch_size 1 \
            --lr 4e-5 \
            --scheduler polynomial_decay_warmup \
            --warmup_proportion 0 \
            --poly_decay_lr_end 5e-6 \
            --poly_decay_power 2.0 \
            --gradient_clipping 1.0 \
            --target_type surv \
            --loss_func cox \
            --prior_type mix_scm \
            --prior_device cpu \
            --batch_size_per_gp 2 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --min_seq_len 1000 \
            --max_seq_len 40000 \
            --log_seq_len True \
            --seq_len_per_gp True \
            --min_train_size 0.5 \
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
            --checkpoint_dir stage2-surv-cox \
            --checkpoint_path stage1-surv-cox/step-{latest}.ckpt \
            --save_temp_every 5 \
            --save_perm_every 100 \
            --only_load_model True