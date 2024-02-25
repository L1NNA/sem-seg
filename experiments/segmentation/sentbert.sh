#!/bin/bash

# python main.py --local_rank 0 \

# OMP_NUM_THREADS=3 python -m torch.distributed.run --nproc_per_node=4 main.py \
python main.py --local_rank 0 \
    --model "autobert" --bert_name "sentbert" \
    --model_id "sentbert_seg" \
    --data_path "./data" --data "segmentation" \
    --batch_size 32 --num_workers 4 --seq_len 511 \
    --gpu --training --testing --epochs 1