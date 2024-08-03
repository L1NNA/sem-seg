#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=2 main.py \
    --model "transformer" --model_id "trans2" \
    --data_path "./data" --data "segmentation" --num_workers 4 --seq_len 512 \
    --gpu --epochs 5 --batch_size 64 \
    --d_model 768 --n_layers 12 --d_ff 3072 --n_heads 12 --dropout 0.1 \
    --testing 