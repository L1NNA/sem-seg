#!/bin/bash

# --do_seg --do_cls
    
python main.py --local_rank 0 \
    --do_seg --num_workers 4 \
    --model "codet5p" --model_id "seg_only_codet5p" --bert_name "codet5p" \
    --data_path "./data" --batch_size 100 --num_workers 4 \
    --seq_len 512 --n_windows 1 \
    --gpu --training --testing --epochs 1

python main.py --local_rank 0 \
    --do_seg --num_workers 4 \
    --model "sentbert" --model_id "seg_only_sentbert" --bert_name "sentbert" \
    --data_path "./data" --batch_size 100 --num_workers 4 \
    --seq_len 512 --n_windows 1 \
    --gpu --training --testing --epochs 1