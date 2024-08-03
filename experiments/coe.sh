#!/bin/bash

python main.py --local_rank 0 \
    --do_seg --do_cls --do_ccr \
    --model "coe" --model_id "new_8_128" --bert_name "sentbert" \
    --data_path "./data" --num_workers 4 \
    --seq_len 1024 --n_windows 8 --w_layers 6 \
    --n_heads 12 --d_model 364 --dropout 0.1 --d_ff 2048 \
    --gpu --training --valid --testing --epochs 1 \
    --batch_size 32 --lr 0.00001
