#!/bin/bash

# OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
python main.py --local_rank 3 \
    --model "cosformer" --model_id "deep" \
    --data_path "./data" --data "segmentation" --num_workers 4 --seq_len 511 \
    --n_layers 12 --n_heads 12 --d_model 768 --d_ff 3072 --dropout 0.1 \
    --gpu --training --validation --testing --epochs 2 \
    --batch_size 28 --lr 0.00001