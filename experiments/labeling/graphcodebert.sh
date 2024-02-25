#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=4 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "autobert" --bert_name "graphcodebert" \
    --model_id "graphcodebert_labeling" \
    --data_path "./data" --data "labeling" --batch_size 16 --num_workers 3 \
    --seq_len 511 \
    --gpu --validation --testing --epochs 2