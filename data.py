import torch
from torch.utils.data import Dataset
import json
import logging
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class RewardDataset(Dataset):
    """
    用于奖励模型训练的数据集
    
    参数:
        data_path (str): JSONL格式数据文件的路径
    """
    def __init__(self, data_path, image_size=448):
        logger.info(f"加载数据集: {data_path}")
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        logger.info(f"加载了 {len(self.data)} 个样本")
        
        # 添加图像变换，调整所有图像到统一大小
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # 自动将PIL图像转换为[0,1]范围的张量，并调整通道顺序
        ])
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            # 不再加载和转换图像，只返回图像路径
            # 让模型的forward方法处理图像加载
            return {
                'image': item['image'],  # 直接返回图像路径
                'query': item['query'],
                'response1': item['response'][0],
                'response2': item['response'][1],
                'ranking': torch.tensor(item['human_ranking'], dtype=torch.float)
            }
        except Exception as e:
            logger.error(f"加载样本 {idx} 时出错: {e}")
            # 返回一个默认样本，避免整个训练中断
            if len(self.data) > 0:
                return self.__getitem__(0 if idx != 0 else 1)
            else:
                raise RuntimeError("数据集为空或所有样本均无法加载")

def collate_fn(batch):
    """
    自定义的collate函数，处理图像和文本数据
    
    参数:
        batch: 一批数据
        
    返回:
        dict: 处理后的批次数据
    """
    images = [item['image'] for item in batch]  # 图像路径列表
    queries = [item['query'] for item in batch]
    responses1 = [item['response1'] for item in batch]
    responses2 = [item['response2'] for item in batch]
    rankings = torch.stack([item['ranking'] for item in batch])
    
    return {
        'image': images,
        'query': queries,
        'response1': responses1,
        'response2': responses2,
        'ranking': rankings
    }

def prepare_dataloader(dataset, batch_size, is_training=True, num_workers=4):
    """
    创建数据加载器
    
    参数:
        dataset: 数据集实例
        batch_size (int): 批次大小
        is_training (bool): 是否为训练模式
        num_workers (int): 数据加载线程数
    
    返回:
        torch.utils.data.DataLoader: 数据加载器
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    ) 