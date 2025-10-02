#!/bin/bash

dataset="iu_xray"
annotation="data/iu_xray/annotation.json"
base_dir="./data/iu_xray/images"

version="v1_deep"
savepath="./save/$dataset/$version"

python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 1 \
    --val_batch_size 1 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 50 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --precision 16-mixed \
    --length_penalty 2.0 \
    --num_workers 2 \
    --accumulate_grad_batches=8 \
    --devices 1 \
    --max_epochs 1 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 0 \
    2>&1 |tee -a ${savepath}/log.txt
