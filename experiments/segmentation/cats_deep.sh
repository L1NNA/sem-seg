#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "cats" --model_id "cats_deep" \
    --data_path "./data" --data "binary" --batch_size 64 --num_workers 4 \
    --seq_len 1024 --n_layers 6 --n_heads 8 --d_model 512 --d_ff 1024 \
    --w_layers 6 --n_windows 16 \
    --gpu --training --testing --epochs 20