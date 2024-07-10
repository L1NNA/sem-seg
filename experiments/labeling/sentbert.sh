#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=4 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "autobert" --bert_name "sentbert" \
    --model_id "try4" \
    --data_path "./data" --data "labeling" \
    --batch_size 36 --num_workers 3 --seq_len 511 \
    --gpu --training --testing --epochs 1 --lr 0.00001