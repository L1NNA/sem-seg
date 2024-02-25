#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "autobert" --model_id "siamese_clone_longformer" --bert_name "longformer" \
    --data_path "./data" --data "siamese_clone" --batch_size 2 --num_workers 4 \
    --seq_len 511 --n_windows 4 --optim "sgd" \
    --gpu --training --validation --testing --epochs 1