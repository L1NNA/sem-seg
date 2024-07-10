#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "cosformer" --model_id "siamese_clone_cosformer" \
    --data_path "./data" --data "siamese_clone" --num_workers 4 \
    --seq_len 768 --n_windows 3  --batch_size 17 \
    --gpu --training --validation --testing --epochs 3