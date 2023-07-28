#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "graphcodebert" --model_id "graph1" \
    --data_path "./data" --data "binary" --batch_size 16 --num_workers 4 \
    --seq_len 512 \
    --gpu --training --testing --epochs 20