#!/bin/bash

# OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
#     --model "graphcodebert" --model_id "graph2plus" \
#     --data_path "./data2" --data "binary" --batch_size 30 --num_workers 4 \
#     --seq_len 512 --test_batch_size 20 \
#     --gpu --training --testing --epochs 1

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
    --model "cosformer" --model_id "cos2" \
    --data_path "./data2" --data "binary" --num_workers 4 --seq_len 512 \
    --d_model 768 --d_ff 3072 --n_layers 12 --n_heads 12 --dropout 0.1 \
    --gpu --training --testing --epochs 3 \
    --batch_size 40 --test_batch_size 20

# OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
#     --model "transformer" --model_id "trans2" \
#     --data_path "./data2" --data "binary" --num_workers 4 --seq_len 512 \
#     --d_model 768 --d_ff 3072 --n_layers 12 --n_heads 12 --dropout 0.1 \
#     --gpu --training --testing --epochs 5 \
#     --batch_size 40 --test_batch_size 20

# OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
#     --model "cats" --model_id "cats2" \
#     --data_path "./data2" --data "binary" --num_workers 4 --seq_len 512 \
#     --d_model 768 --d_ff 3072 --n_layers 6 --n_heads 12 --dropout 0.1 \
#     --w_layers 6 --n_windows 8 \
#     --gpu --training --testing --epochs 5 \
#     --batch_size 40 --test_batch_size 20