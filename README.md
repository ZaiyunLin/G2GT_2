Requirements and Installation

# install requirements
create a new environment \
conda create --name G2GT python=3.7 \
conda activate G2GT 

pip install pytorch-lightning==1.4.5  \
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
pip install networkx \
pip install tensorboardX==2.4.1 \
pip install rdkit-pypi==2021.9.3 \
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
pip install pympler \
pip install --upgrade easy-kubeflow \
pip install Cython
pip install joblib


# Example Usage
```
#!/usr/bin/env bash
[ -z "${exp_name}" ] && exp_name="uspto"
[ -z "${seed}" ] && seed="0"
[ -z "${arch}" ] && arch="--ffn_dim 1024 --hidden_dim 256 --dropout_rate 0.1 --intput_dropout_rate 0.05 --attention_dropout_rate 0.1 --n_layer 6 --peak_lr 2.5e-4 --end_lr 1e-6 --head_size 12 --weight_decay 0.00 --edge_type one_hop --warmup_updates 3000 --tot_updates 700000"
[ -z "${batch_size}" ] && batch_size="5"
[ -z "${dataset_name}" ] && dataset_name="uspto-50k-split2-split"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"      
echo "arch: ${arch}"             
echo "seed: ${seed}"             
echo "batch_size: ${batch_size}" 
echo "==============================================================================="

default_root_dir=$PWD
mkdir -p $default_root_dir
n_gpu=1
export CUDA_VISIBLE_DEVICES=0

python entry.py --num_workers 1 --num_nodes 1 --seed $seed --batch_size $batch_size  --min_epochs 100 --accumulate_grad_batches 1  --sync_batchnorm True  --val_check_interval 1000 \
      --dataset_name $dataset_name --gradient_clip_val 4 \
      --gpus $n_gpu  --accelerator ddp \
      $arch \
      --default_root_dir $default_root_dir --progress_bar_refresh_rate 10  \


```
# Example Inference  
``` bash inference.sh```

``` # Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash
[ -z "${arch}" ] && arch="--ffn_dim 1024 --hidden_dim 256 --dropout_rate 0.1 --intput_dropout_rate 0.1 --attention_dropout_rate 0.1 --n_layer 6 --peak_lr 2.5e-4 --end_lr 1e-6 --head_size 12 --weight_decay 0.00 --edge_type one_hop --warmup_updates 1000 --tot_updates 500000"
[ -z "$ckpt_name" ] && ckpt_name=last.ckpt

export CUDA_VISIBLE_DEVICES=2,4,6
default_root_dir=$PWD
n_gpu=1

python entry.py --num_workers 6 --seed 0 --batch_size 1 \
      --dataset_name uspto-full-distilled-weaklabel \
      --gpus $n_gpu --accelerator ddp  $arch \
      --default_root_dir $default_root_dir \
      --checkpoint_path $default_root_dir/lightning_logs/checkpoints/last.ckpt --test --progress_bar_refresh_rate 1 \
      --inference_path $default_root_dir/results/uspto-full-distilled-weaklabel-sample \
      --beam_size 50 \ #beam size for sampling and beam search
      #--sampling \ # use sampling or beam search
```


# Calculate accuracy
Add multiple foldername if using more than one results (e.g. when combining sampling results with beam search results).  
The raw inference outputs are under results folder
``` python score.py --file uspto-50k-sampling uspto-50k-beam --beam 100``` 

# Notice 
Currently the weak-ensemble tags are hard coded to be 50.
Our final results are calculated using a joint set of beam search and weak_enemble sampling
