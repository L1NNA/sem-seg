#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "cosformer" --model_id "cos1" \
    --data_path "./data" --data "binary" --num_workers 4 \
    --seq_len 512 \
    --gpu --training --testing --epochs 5 \
    --batch_size 192 --test_batch_size 64