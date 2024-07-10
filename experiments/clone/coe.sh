#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
    --model "coe" --model_id "8_128" --bert_name "sentbert" \
    --data_path "./data" --data "coe" --num_workers 4 \
    --seq_len 1024 --n_windows 8 --w_layers 6 \
    --n_heads 12 --d_model 364 --dropout 0.1 --d_ff 2048 \
    --gpu --training --testing --epochs 1 \
    --batch_size 16 --lr 0.00001
