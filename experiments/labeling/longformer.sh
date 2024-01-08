#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=2 main.py \
    --model "longformer" --model_id "longformer_labeling" \
    --data_path "./data" --data "labeling" --batch_size 2 --num_workers 4 \
    --seq_len 4096 \
    --gpu --training --validation --testing --epochs 2