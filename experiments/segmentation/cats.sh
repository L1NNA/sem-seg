#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "cats" --model_id "cats1" \
    --data_path "./data2" --data "binary" --num_workers 4 --seq_len 512 \
    --gpu --training --testing --epochs 5 \
    --batch_size 128 --test_batch_size 32