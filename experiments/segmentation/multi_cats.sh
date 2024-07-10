#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "multi_cats" --model_id "long" --gpu \
    --data_path "./data" --data "multi_window_seg" --num_workers 4 \
    --bert_name "sentbert" \
    --seq_len 1536 --n_windows 4 \
    --n_heads 12 --d_model 768 --dropout 0.1 --d_ff 2048 --w_layers 6 \
    --training --testing --epochs 1 --batch_size 16
