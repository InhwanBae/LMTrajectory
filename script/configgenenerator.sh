#!/bin/bash
echo "Generate config file"

obs_len=8
pred_len=12

python utils/config.py \
    --config_name config.json \
    --model_name_or_path t5-small \
    --cache_dir ./.cache/ \
    --dataset_path ./datasets/ \
    --metric pixel \
    --obs_len ${obs_len} \
    --pred_len ${pred_len} \
    --checkpoint_path /data/LLM_model/ \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 1024 \
    --gradient_accumulation_steps 1 \
    --checkpointing_steps epoch \
    --preprocessing_num_workers 64 \
    --num_train_epochs 64 \
    --learning_rate 1e-4 \
    --use_slow_tokenizer \
    # --overwrite_cache
