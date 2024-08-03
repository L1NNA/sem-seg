#!/bin/bash

# --do_seg --do_cls

python main.py --local_rank 0 \
    --do_ccr --num_workers 4 \
    --model "sentbert" --model_id "ccr_sentbert" --bert_name "sentbert" \
    --data_path "./data" --batch_size 80 --test_batch_size 160 --num_workers 4 \
    --seq_len 512 --n_windows 1 \
    --gpu --testing --epochs 5