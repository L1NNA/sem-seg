#!/bin/bash

python main.py --local_rank 0 \
    --do_ccr --num_workers 4 \
    --model "graphcodebert" --model_id "ccr_graph" --bert_name "graphcodebert" \
    --data_path "./data" --batch_size 48 --num_workers 4 \
    --seq_len 512 --n_windows 1 \
    --gpu --training --testing --epochs 1