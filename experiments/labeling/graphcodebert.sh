#!/bin/bash

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
    --model "autobert" --bert_name "graphcodebert" \
    --model_id "graphcodebert_labeling" \
    --data_path "./data" --data "labeling" --batch_size 16 --num_workers 4 \
    --seq_len 511 \
    --gpu --training --validation --testing --epochs 2