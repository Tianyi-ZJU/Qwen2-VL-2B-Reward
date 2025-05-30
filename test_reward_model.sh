#!/bin/bash

# 确保结果目录存在
mkdir -p results

# 运行奖励模型测试脚本
CUDA_VISIBLE_DEVICES=1 python test_reward_model.py \
    --model_path checkpoints/checkpoint_epoch_3.pt \
    --base_model_name pretrained/qwen \
    --data_path /mnt/lsk_nas/liuhuadai/luotianyi/reward/dataset/datasets--MMInstruction--VL-RewardBench/snapshots/VL-RewardBench/data/test-00000-of-00001.parquet \
    --output_path results/reward_model_results.jsonl \
    --batch_size 4 \
    --use_flash_attention \
    --pooling_method mean 