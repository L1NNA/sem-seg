#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=4 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "coe" --model_id "try6" --bert_name "sentbert" \
    --data_path "./data" --data "coe" --num_workers 3 \
    --seq_len 511 --n_windows 3 --w_layers 2 \
    --n_heads 12 --d_model 768 --dropout 0.1 --d_ff 2048 \
    --gpu --training --validation --testing --epochs 1 \
    --batch_size 14 --lr 0.00001