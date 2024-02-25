#!/bin/bash

# python main.py --local_rank 0 \

# OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
# python main.py --local_rank 1 \
#     --model "autobert" --model_id "graph2plus" --bert_name "graphcodebert" \
#     --data_path "./data" --data "segmentation" --batch_size 28 --num_workers 4 \
#     --seq_len 511 \
#     --gpu --training --testing --epochs 1

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "autobert" --model_id "short" --bert_name "graphcodebert" \
    --data_path "./data" --data "segmentation" --batch_size 28 --num_workers 4 \
    --seq_len 256 \
    --gpu --training --testing --epochs 1