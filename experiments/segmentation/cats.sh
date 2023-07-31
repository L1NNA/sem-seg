#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "cats" --model_id "cat2" \
    --data_path "./data" --data "binary" --num_workers 4 --seq_len 512 \
    --n_layers 6 --n_heads 8 --d_model 512 --d_ff 1024 \
    --gpu --testing --training \
    --epochs 20 --batch_size 64 --test_batch_size 64