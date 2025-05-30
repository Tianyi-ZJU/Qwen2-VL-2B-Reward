#!/bin/bash

# 运行奖励模型训练脚本

CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py \
    --base_model_name pretrained/qwen \
    --train_data_path dataset/train_data.jsonl \
    --val_data_path dataset/val_data.jsonl \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --use_flash_attention \
    --pooling_method mean \
    --use_scheduler \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --num_workers 4 \
    --seed 42 \
    --use_wandb \
    --wandb_project qwen-reward \
    --wandb_api_key \
    --gradient_accumulation_steps 1 \
    --log_samples \
    --score_min -5.0 \
    --score_max 5.0 \
    --resume_from_checkpoint checkpoints/checkpoint_epoch_14.pt \
    --load_weights_only