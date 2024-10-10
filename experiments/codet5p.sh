#!/bin/bash

# --do_ccr

python main.py --local_rank 0 \
    --do_seg --num_workers 4 \
    --model "codet5p" --model_id "seg+_t5p" --bert_name "codet5p" \
    --data_path "./data" --batch_size 32 --num_workers 4 \
    --seq_len 512 --n_windows 1 \
    --gpu --training --testing --epochs 10