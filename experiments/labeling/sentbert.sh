#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=2 main.py \
    --model "autobert" --bert_name "sentbert" \
    --model_id "sentbert_labeling" \
    --data_path "./data" --data "labeling" \
    --batch_size 64 --num_workers 4 --seq_len 511 \
    --gpu --testing --epochs 2