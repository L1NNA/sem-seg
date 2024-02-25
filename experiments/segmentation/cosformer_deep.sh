#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=2 main.py \
    --model "cosformer" --model_id "cos_deep2" \
    --data_path "./data" --data "segmentation" --seq_len 512 --num_workers 4 \
    --batch_size 64 \
    --n_layers 12 --n_heads 16 --d_model 1024 --d_ff 2048 \
    --gpu --testing --epochs 26 \
    --lr 0.00007 --dropout 0.1 # low lr, dropout
