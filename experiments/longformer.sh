#!/bin/bash

# python main.py --local_rank 0 \

python main.py --local_rank 0 \
    --model "longformer" --model_id "ccr_long" --bert_name "longformer" \
    --data_path "./data" --batch_size 5 --num_workers 4 \
    --seq_len 2048 --n_windows 4 \
    --gpu --training --validation --testing --epochs 1