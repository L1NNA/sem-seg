#!/bin/bash

# OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
python main.py --local_rank 2 \
    --model "cats" --model_id "deep" \
    --data_path "./data" --data "segmentation" --num_workers 4 --seq_len 511 \
    --n_windows 7 --w_layers 6 \
    --d_model 768 --n_layers 6 --d_ff 3072 --n_heads 12 --dropout 0.1 \
    --gpu --training --validation --testing --epochs 5 \
    --batch_size 32