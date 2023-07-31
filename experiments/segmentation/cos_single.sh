#!/bin/bash

python run_seg.py \
    --model "cosformer" --model_id "cos_single" \
    --data_path "./data" --data "binary" --num_workers 4 \
    --seq_len 512 --n_layers 6 --n_heads 8 --d_model 512 --d_ff 1024 \
    --gpu --testing --epochs 1 \
    --batch_size 64 --test_batch_size 64