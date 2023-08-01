#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "cats" --model_id "cat1" \
    --data_path "./data" --data "binary" --num_workers 4 --seq_len 512 \
    --n_layers 4 --w_layers 2 \
    --gpu --testing --training \
    --epochs 5 --batch_size 256 --test_batch_size 64