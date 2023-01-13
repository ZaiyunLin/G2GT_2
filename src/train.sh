# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash
cd g2gt_github/src/
[ -z "${exp_name}" ] && exp_name="uspto50k"
[ -z "${seed}" ] && seed="0"
[ -z "${arch}" ] && arch="--ffn_dim 1024 --hidden_dim 256 --dropout_rate 0.1 --intput_dropout_rate 0.05 --attention_dropout_rate 0.1 --n_layer 6 --peak_lr 2.5e-4 --end_lr 1e-6 --head_size 12 --weight_decay 0.00 --edge_type one_hop --warmup_updates 3000 --tot_updates 700000"
[ -z "${batch_size}" ] && batch_size="5"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"      # all_fold
echo "arch: ${arch}"              # --ffn_dim 4096 --hidden_dim 4096 --attention_dropout_rate 0.1 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20 --weight_decay 0.0 --intput_dropout_rate 0.0 --warmup_updates 10000 --tot_updates 15000000
echo "seed: ${seed}"              # 0
echo "batch_size: ${batch_size}"  # 256 x 4
echo "==============================================================================="

default_root_dir=$PWD
# export NCCL_SOCKET_IFNAME=lo
mkdir -p $default_root_dir
n_gpu=1
export CUDA_VISIBLE_DEVICES=0

python entry.py --num_workers 10 --num_nodes 1 --seed $seed --batch_size $batch_size  --min_epochs 100 --accumulate_grad_batches 2  --sync_batchnorm True  --val_check_interval 1000 \
      --dataset_name uspto --gradient_clip_val 4 \
      --gpus $n_gpu  --accelerator ddp \
      $arch \
      --default_root_dir $default_root_dir --progress_bar_refresh_rate 10\ 
      #>> log.txt
      #--accumulate_grad_batches=1  >> text.log\
      #--val_check_interval 5 --check_val_every_n_epoch=1 \
      #limit_train_batches=0.1, limit_val_batches=0.2, limit_test_batches=0.3
      #CUDA_LAUNCH_BLOCKING=1 
      