#!/bin/bash

# python main.py --local_rank 0 \

OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=5 main.py \
    --model "graphcodebert" --model_id "coe" \
    --data_path "./data" --data "coe" --batch_size 64 --num_workers 4 \
    --seq_len 511 --n_windows 512 \
    --gpu --training --validation --testing --epochs 1