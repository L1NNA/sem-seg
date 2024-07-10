#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=2 main.py \
    --model "cosformer" --model_id "long" \
    --data_path "./data" --data "labeling" --seq_len 1024 --num_workers 4 \
    --batch_size 64 --skip_label 2 \
    --n_layers 12 --n_heads 16 --d_model 1024 --d_ff 2048 \
    --gpu --testing --epochs 5 \
    --lr 0.00007 --dropout 0.1 # low lr, dropout
