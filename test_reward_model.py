#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse
import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO
from collections import defaultdict
import logging
import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reward_model_test.log')
    ]
)
logger = logging.getLogger(__name__)

# 导入自定义模型和工具函数
from models import RewardModel
from utils import load_checkpoint, set_seed
from reward_prompt import reward_prompt

def parse_args():
    parser = argparse.ArgumentParser(description="测试奖励模型在VL_RewardBench上的性能")
    parser.add_argument('--model_path', type=str, required=True, help='奖励模型检查点路径')
    parser.add_argument('--base_model_name', type=str, default='pretrained/qwen', help='基础模型路径')
    parser.add_argument('--data_path', type=str, required=True, help='VL_RewardBench测试数据路径(.parquet文件)')
    parser.add_argument('--output_path', type=str, default='reward_model_results.jsonl', help='结果输出路径')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小')
    parser.add_argument('--use_flash_attention', action='store_true', help='是否使用Flash Attention 2加速')
    parser.add_argument('--pooling_method', type=str, default='mean', help='池化方法')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

def load_model(args):
    """加载奖励模型"""
    logger.info(f"从 {args.model_path} 加载模型")
    
    # 初始化模型
    model = RewardModel(
        base_model_name=args.base_model_name,
        use_flash_attention=args.use_flash_attention,
        pooling_method=args.pooling_method
    )
    model = model.to(args.device)
    
    # 加载检查点
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"模型加载成功")
    
    return model

def process_image(image_bytes):
    """处理图像数据"""
    if isinstance(image_bytes, bytes):
        image = Image.open(BytesIO(image_bytes))
        return image
    else:
        # 如果已经是PIL.Image或其他格式，直接返回
        return image_bytes

def evaluate_pair(model, image, query, response1, response2, score_range=(-5, 5)):
    """评估一对回答的质量"""
    text1 = reward_prompt(query, response1, score_range=score_range)
    text2 = reward_prompt(query, response2, score_range=score_range)
    
    # 处理图像
    processed_image = process_image(image)
    
    # 获取分数
    with torch.no_grad():
        score1 = model([processed_image], [text1])[0].item()
        score2 = model([processed_image], [text2])[0].item()
    
    # 判断哪个回答更好
    better_idx = 0 if score1 > score2 else 1
    scores = [float(score1), float(score2)]  # 确保是Python原生类型
    
    return better_idx, scores

def test_model(model, data_path, output_path, batch_size, device):
    """测试模型在VL_RewardBench上的性能"""
    logger.info(f"加载测试数据: {data_path}")
    
    # 加载数据
    df = pd.read_parquet(data_path)
    logger.info(f"加载了 {len(df)} 条测试样本")
    
    results = []
    dataset_results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="测试中"):
        item = row.to_dict()
        
        # 处理图像
        image = process_image(item["image"]['bytes'])
        
        # 获取查询和回答
        query = item["query"]
        responses = item["response"]
        human_ranking = item["human_ranking"]
        
        # 获取模型预测
        better_idx, scores = evaluate_pair(
            model, 
            image, 
            query, 
            responses[0], 
            responses[1]
        )
        
        # 判断是否正确
        correct = (better_idx == 0 and human_ranking[1] > human_ranking[0]) or \
                 (better_idx == 1 and human_ranking[0] > human_ranking[1])
        
        # 获取数据集类型
        dataset_type = get_dataset_from_id(item["id"])
        dataset_results[dataset_type]["total"] += 1
        if correct:
            dataset_results[dataset_type]["correct"] += 1
        
        # 保存结果 - 确保NumPy数组被转换为Python列表
        result = {
            "id": item["id"],
            "query": query,
            "response": responses,
            "human_ranking": human_ranking.tolist() if isinstance(human_ranking, np.ndarray) else human_ranking,
            "model_scores": scores,
            "model_choice": int(better_idx),  # 确保是Python原生类型
            "correct": bool(correct),  # 确保是Python原生类型
            "dataset": dataset_type
        }
        results.append(result)
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            # 确保所有NumPy数组和其他不可序列化的对象被转换为Python原生类型
            json_result = json.dumps(result, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
            f.write(json_result + '\n')
    
    # 计算总体准确率
    total_correct = sum(result["correct"] for result in results)
    total_samples = len(results)
    accuracy = total_correct / total_samples
    
    logger.info(f"总体准确率: {accuracy:.4f} ({total_correct}/{total_samples})")
    
    # 计算各数据集准确率
    logger.info("各数据集准确率:")
    group_mapping = {
        "vlfeedback": "general",
        "povid": "hallucination",
        "reasoning_tasks": "reasoning",
        "rlhf-v": "hallucination",
        "rlaif-v": "hallucination",
        "wildvision-battle": "general"
    }
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    
    for dataset, stats in dataset_results.items():
        acc = stats["correct"] / stats["total"]
        logger.info(f"  {dataset}: {acc:.4f} ({stats['correct']}/{stats['total']})")
        
        group = group_mapping.get(dataset, "other")
        group_correct[group] += stats["correct"]
        group_total[group] += stats["total"]
    
    # 计算分组准确率
    logger.info("分组准确率:")
    task_list = ['reasoning', 'hallucination', 'general']
    for group in task_list:
        if group_total[group] > 0:
            acc = group_correct[group] / group_total[group]
            logger.info(f"  {group}: {acc:.4f} ({group_correct[group]}/{group_total[group]})")
    
    # 计算宏平均和总体准确率
    macro_avg = sum(group_correct[k]/group_total[k] for k in task_list if group_total[k] > 0) / len([k for k in task_list if group_total[k] > 0])
    overall_acc = sum(group_correct.values()) / sum(group_total.values())
    
    logger.info(f"宏平均准确率: {macro_avg:.4f}")
    logger.info(f"总体准确率: {overall_acc:.4f}")
    
    return accuracy, results

def get_dataset_from_id(id_str):
    """根据ID确定数据集类型"""
    def get_id_prefix(id_value):
        split_index = min((id_value.find('_'), id_value.find('-')), key=lambda x: x if x != -1 else float('inf'))
        if split_index != -1:
            id_prefix = id_value[:split_index]
        else:
            id_prefix = id_value
        return id_prefix
    
    id_prefix = get_id_prefix(id_str)
    if id_prefix == "RLAIF":
        return "rlaif-v"
    elif id_prefix == "RLHF":
        return "rlhf-v"
    elif id_prefix == "mathverse" or id_prefix == "mmmu":
        return "reasoning_tasks"
    elif id_prefix == "wildvision":
        return "wildvision-battle"
    elif id_prefix == 'hallucination':
        return "povid"
    else:
        return "vlfeedback"

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 创建结果输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 从模型路径中提取模型名称
    model_name = os.path.basename(args.model_path).replace('.pt', '')
    
    # 添加时间戳到输出文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_path
    if output_path.endswith('.jsonl'):
        output_path = output_path.replace('.jsonl', f'_{model_name}_{timestamp}.jsonl')
    else:
        output_path = f"{output_path}_{model_name}_{timestamp}.jsonl"
    
    # 加载模型
    model = load_model(args)
    model.eval()
    
    # 测试模型
    accuracy, results = test_model(model, args.data_path, output_path, args.batch_size, args.device)
    
    logger.info(f"测试完成，结果已保存到 {output_path}")
    logger.info(f"总体准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main() 