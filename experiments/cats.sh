#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "cats" --model_id "siamese_long" \
    --data_path "./data" --data "siamese_clone" --num_workers 4 --seq_len 768 \
    --n_windows 3 --w_layers 6 \
    --d_model 768 --n_layers 6 --d_ff 3072 --n_heads 12 --dropout 0.1 \
    --gpu --training --validation --testing --epochs 5 \
    --batch_size 17 --lr 0.00001