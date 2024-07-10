#!/bin/bash

# python main.py --local_rank 2 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "multi_cats" --model_id "short_4" --bert_name "sentbert" \
    --data_path "./data" --num_workers 4 \
    --w_layers 6 --lr 0.00001 \
    --d_model 768 --n_layers 6 --d_ff 3072 --n_heads 12 --dropout 0.1 \
    --gpu --training --validation --testing --epochs 5 \
    --seq_len 1024 --n_windows 4 --batch_size 16 \
    --skip_label 1  --data "labeling"

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "multi_cats" --model_id "short_4_skip_seg" --bert_name "sentbert" \
    --data_path "./data" --num_workers 4 --seq_len 1024 \
    --w_layers 6 --lr 0.00001 \
    --d_model 768 --n_layers 6 --d_ff 3072 --n_heads 12 --dropout 0.1 \
    --gpu --training --validation --testing --epochs 5 \
    --seq_len 1024 --n_windows 4 --batch_size 16 \
    --skip_label 1  --data "labeling" --skip_seg
