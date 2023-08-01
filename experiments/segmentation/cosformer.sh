#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "cosformer" --model_id "cos3" \
    --data_path "./data" --data "binary" --num_workers 4 --max_samples 500000 \
    --seq_len 512 --d_model 768 --d_ff 2048 --n_heads 16 \
    --gpu --testing --epochs 5 \
    --batch_size 64 --test_batch_size 48