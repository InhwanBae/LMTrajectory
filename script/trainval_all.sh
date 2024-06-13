#!/bin/bash
echo "Start training with Accelerate"


accelerate launch trainval.py \
    --cfg ./config/config-pixel.json \
    --dataset eth \
    --tag LMTraj-nt-eth-pixel

accelerate launch trainval.py \
    --cfg ./config/config-pixel.json \
    --dataset hotel \
    --tag LMTraj-nt-hotel-pixel

accelerate launch trainval.py \
    --cfg ./config/config-pixel.json \
    --dataset univ \
    --tag LMTraj-nt-univ-pixel

accelerate launch trainval.py \
    --cfg ./config/config-pixel.json \
    --dataset zara1 \
    --tag LMTraj-nt-zara1-pixel

accelerate launch trainval.py \
    --cfg ./config/config-pixel.json \
    --dataset zara2 \
    --tag LMTraj-nt-zara2-pixel
