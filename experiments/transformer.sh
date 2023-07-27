#!/bin/bash


# -m torch.distributed.run --nproc_per_node=8
#  --local_rank 0

python -m torch.distributed.run --nproc_per_node=4 main.py \
    --training --model_name "transformer" \
    --model "transformer" --data "binary" \
    --data_path "./data" --batch_size 64 --num_workers 4 \
    --seq_len 512 --n_layers 6 --n_heads 8 --d_model 512 --d_ff 1024\
    --gpu