#!/bin/bash

bash experiments/get_enwik8.sh

base_dir=~/checkpoints

python main.py \
    --nepochs 100 --nbatches 1000 --data data/enwik8 \
    --hid-sz 256 --inner-hid-sz 1024 --mem-sz 256 --batch-sz 64 --nlayers 8 \
    --lr 0.0003 --momentum 0 --dropout 0 --optim adam --lr-warmup 8000 \
    --attn-lim 4096 --nheads 4 --grad-clip 0.3 \
    --checkpoint-freq 25 --expire-span --expire-span-loss 0.000001 \
    --expire-span-ramp 64 --expire-span-pre-div 64 \
    --checkpoint $base_dir/enwik8_small.pt
