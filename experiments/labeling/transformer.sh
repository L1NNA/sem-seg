#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
    --model "transformer" --model_id "transformer_labeling" \
    --data_path "./data" --data "labeling" \
    --batch_size 96 --num_workers 4 --seq_len 511 \
    --gpu --testing --epochs 4