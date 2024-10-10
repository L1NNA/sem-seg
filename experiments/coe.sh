#!/bin/bash

# --training  --batch_size 16 --do_cls --do_ccr

python main.py --local_rank 0 \
    --do_seg --do_ccr --num_workers 4 \
    --model "coe" --model_id "seg_cls_t5p" --bert_name "codet5p" \
    --data_path "./data" --num_workers 4 \
    --test_batch_size 128 --batch_size 48 \
    --seq_len 512 --n_windows 1 \
    --gpu --training --testing --epochs 3

# python main.py --local_rank 0 \
#     --do_seg \
#     --model "coe" --model_id "seg_only_512" --bert_name "sentbert" \
#     --data_path "./data" --num_workers 4 \
#     --seq_len 1024 --n_windows 2 --w_layers 6 \
#     --n_heads 12 --d_model 364 --dropout 0.1 --d_ff 2048 \
#     --gpu --training --valid --testing --epochs 3 \
#     --batch_size 32 --lr 0.00001
