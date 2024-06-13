#!/bin/bash
echo "Start evaluation with Accelerate"


accelerate launch trainval.py \
    --cfg ./config/config-pixel-deterministic.json \
    --dataset eth \
    --tag exp3-ct-eth-pixel-multimodal-best \
    --test

accelerate launch trainval.py \
    --cfg ./config/config-pixel-deterministic.json \
    --dataset hotel \
    --tag exp3-ct-hotel-pixel-multimodal-best \
    --test

accelerate launch trainval.py \
    --cfg ./config/config-pixel-deterministic.json \
    --dataset univ \
    --tag exp3-ct-univ-pixel-multimodal-best \
    --test

accelerate launch trainval.py \
    --cfg ./config/config-pixel-deterministic.json \
    --dataset zara1 \
    --tag exp3-ct-zara1-pixel-multimodal-best \
    --test

accelerate launch trainval.py \
    --cfg ./config/config-pixel-deterministic.json \
    --dataset zara2 \
    --tag exp3-ct-zara2-pixel-multimodal-best \
    --test
