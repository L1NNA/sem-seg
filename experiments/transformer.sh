#!/bin/bash

python -m torch.distributed.run --nproc_per_node=4 main.py \
    --training --model_name "transformer" \
    --model "transformer" --data "seq" \
    --data_path "./dest" \
    --batch_size 32 --seq_len 512 --num_workers 4 \
    --n_layers 4 --d_model 256 \
    --gpu