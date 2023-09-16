#!/bin/bash

python main.py --local_rank 1 \
    --model "graphcodebert" --model_id "graph2plus" \
    --data_path "./data2" --data "binary" --batch_size 1 --num_workers 4 \
    --seq_len 512 \
    --gpu --epochs 1 \
    --segmentation "./data2/test/angular2-weather-widget_angular2-weather-widget.umd.min.js.pt"
