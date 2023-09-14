#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
    --model "cosformer" --model_id "cos" \
    --data_path "./data2" --data "binary" --num_workers 4 --seq_len 512 \
    --d_model 768 --n_layers 4 --n_heads 6 --dropout 0.3 \
    --gpu --training --testing --epochs 3 \
    --batch_size 200 --test_batch_size 200

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
    --model "transformer" --model_id "trans" \
    --data_path "./data2" --data "binary" --num_workers 4 --seq_len 512 \
    --d_model 768 --n_layers 4 --n_heads 6 --dropout 0.3 \
    --gpu --training --testing --epochs 5 \
    --batch_size 200 --test_batch_size 200

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
    --model "cats" --model_id "cats" \
    --data_path "./data2" --data "binary" --num_workers 4 --seq_len 512 \
    --d_model 768 --n_layers 2 --n_heads 6 --dropout 0.3 \
    --w_layers 2 --n_windows 8 \
    --gpu --training --testing --epochs 5 \
    --batch_size 200 --test_batch_size 200