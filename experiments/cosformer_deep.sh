#!/bin/bash

# OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
#     --model "cosformer" --model_id "cos_deep2" \
#     --data_path "./data" --data "binary" --seq_len 512 --num_workers 4 \
#     --batch_size 48 --test_batch_size 16 \
#     --n_layers 12 --n_heads 16 --d_model 1024 --d_ff 2048 \
#     --gpu --training --epochs 20 \
#     --lr_warmup 8000 --lr 0.0007 --dropout 0.3 # a bit high lr

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "cosformer" --model_id "cos_deep2" \
    --data_path "./data" --data "binary" --seq_len 512 --num_workers 4 \
    --batch_size 48 --test_batch_size 16 \
    --n_layers 12 --n_heads 16 --d_model 1024 --d_ff 2048 \
    --gpu --training --testing --epochs 25 \
    --lr_warmup 8000 --lr 0.00007 --dropout 0.1 # low lr, dropout
