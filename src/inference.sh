# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash
[ -z "${arch}" ] && arch="--ffn_dim 1024 --hidden_dim 256 --dropout_rate 0.1 --intput_dropout_rate 0.1 --attention_dropout_rate 0.1 --n_layer 6 --peak_lr 2.5e-4 --end_lr 1e-6 --head_size 12 --weight_decay 0.00 --edge_type one_hop --warmup_updates 1000 --tot_updates 500000"
[ -z "$ckpt_name" ] && ckpt_name=last.ckpt

export CUDA_VISIBLE_DEVICES=2,4,6
# export NCCL_SOCKET_IFNAME=lo
export CUDA_LAUNCH_BLOCKING=1
default_root_dir=$PWD
n_gpu=1

python entry.py --num_workers 6 --seed 0 --batch_size 1 \
      --dataset_name typed_uspto50k_split2 \
      --gpus $n_gpu --accelerator ddp  $arch \
      --default_root_dir $default_root_dir \
      --checkpoint_path $default_root_dir/lightning_logs/checkpoints/last.ckpt --test --progress_bar_refresh_rate 1 \
      --inference_path $default_root_dir/results/typed_uspto50k_split2/ \
      --weak_ensemble 1 \
      --limit_test_batches=0.0002 

