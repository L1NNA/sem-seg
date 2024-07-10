#!/bin/bash

# python main.py --local_rank 2 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "single_cats" --model_id "try1" --bert_name "sentbert" \
    --data_path "./data" --num_workers 4 \
    --w_layers 6 --lr 0.00001 \
    --d_model 768 --n_layers 6 --d_ff 3072 --n_heads 12 --dropout 0.1 \
    --gpu --training --validation --testing --epochs 5 \
    --seq_len 2044 --n_windows 4 --batch_size 8 \
    --skip_label 1  --data "single_labeling"


OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "single_cats" --model_id "try2" --bert_name "sentbert" \
    --data_path "./data" --num_workers 4 \
    --w_layers 6 --lr 0.00001 \
    --d_model 768 --n_layers 6 --d_ff 3072 --n_heads 12 --dropout 0.1 \
    --gpu --training --validation --testing --epochs 5 \
    --seq_len 2048 --n_windows 8 --batch_size 8 \
    --skip_label 1  --data "single_labeling"

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "single_cats" --model_id "try3" --bert_name "sentbert" \
    --data_path "./data" --num_workers 4 \
    --w_layers 6 --lr 0.00001 \
    --d_model 768 --n_layers 6 --d_ff 3072 --n_heads 12 --dropout 0.1 \
    --gpu --training --validation --testing --epochs 5 \
    --seq_len 1024 --n_windows 4 --batch_size 12 \
    --skip_label 1  --data "single_labeling"

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "single_cats" --model_id "try4" --bert_name "sentbert" \
    --data_path "./data" --num_workers 4 \
    --w_layers 6 --lr 0.00001 \
    --d_model 768 --n_layers 6 --d_ff 3072 --n_heads 12 --dropout 0.1 \
    --gpu --training --validation --testing --epochs 5 \
    --seq_len 1022 --n_windows 2 --batch_size 12 \
    --skip_label 1  --data "single_labeling"