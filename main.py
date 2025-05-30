import argparse
import os
import torch
from models import RewardModel
from data import RewardDataset, prepare_dataloader
from trainer import RewardModelTrainer
from utils import setup_logging, set_seed
import gc

logger = setup_logging()

def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="Qwen2-VL奖励模型训练")
    
    # 模型参数
    parser.add_argument('--base_model_name', type=str, default='pretrained/qwen',
                        help='基础模型路径')
    parser.add_argument('--use_flash_attention', action='store_true', 
                        help='是否使用Flash Attention 2加速')
    parser.add_argument('--pooling_method', type=str, default='mean',
                        choices=['cls', 'mean', 'max', 'last', 'cls_mean'],
                        help='序列表示的池化方法')
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='训练数据路径')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='验证数据路径')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='数据加载器的工作进程数')
    parser.add_argument('--image_size', type=int, default=448,
                        help='图像调整大小')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='权重衰减系数')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, 
                        help='梯度累积步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                        help='梯度裁剪阈值')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, 
                        help='学习率预热比例')
    parser.add_argument('--use_scheduler', action='store_true', 
                        help='是否使用学习率调度器')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    parser.add_argument('--use_margin_loss', action='store_true',
                        help='是否使用边界损失(margin loss)')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='边界损失的边界值')
    
    # 检查点参数
    parser.add_argument('--save_interval', type=int, default=1,
                        help='保存检查点的间隔轮数')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--experiment_name', type=str, default='reward_model',
                        help='实验名称，用于区分不同的训练运行，避免检查点覆盖')
    parser.add_argument('--keep_last_n', type=int, default=3,
                        help='保留最近的N个检查点')
    parser.add_argument('--early_stop', type=int, default=5,
                        help='早停轮数')
    parser.add_argument('--resume_from_checkpoint', type=str, default='', 
                        help='从指定检查点恢复训练')
    parser.add_argument('--load_weights_only', action='store_true',
                        help='仅加载模型权重，不继承训练状态（学习率、epoch等）')
    
    # wandb相关参数
    parser.add_argument('--use_wandb', action='store_true', 
                        help='是否使用wandb记录训练过程')
    parser.add_argument('--wandb_project', type=str, default='qwen-reward', 
                        help='wandb项目名称')
    parser.add_argument('--log_interval', type=int, default=10, 
                        help='记录训练信息的间隔步数')
    parser.add_argument('--log_samples', action='store_true', 
                        help='是否记录样本数据到wandb')
    parser.add_argument('--wandb_api_key', type=str, default='', 
                        help='wandb API密钥')
    parser.add_argument('--wandb_optional', action='store_true', 
                        help='wandb登录失败时是否继续训练')
    
    args = parser.parse_args()
    return args

def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 创建保存检查点的目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # 添加显存监控
    def print_gpu_memory():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(f"GPU {i}: 已分配 {allocated:.2f} GB, 缓存 {cached:.2f} GB")
    
    print_gpu_memory()
    logger.info("初始化模型...")
    
    # 初始化模型
    model = RewardModel(
        args.base_model_name, 
        use_flash_attention=args.use_flash_attention, 
        pooling_method=args.pooling_method
    )
    
    print_gpu_memory()
    logger.info("加载数据集...")
    
    # 加载数据集
    train_dataset = RewardDataset(args.train_data_path, image_size=args.image_size)
    val_dataset = RewardDataset(args.val_data_path, image_size=args.image_size) if args.val_data_path else None
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader = prepare_dataloader(
        train_dataset, 
        batch_size=args.batch_size, 
        is_training=True, 
        num_workers=args.num_workers
    )
    
    val_loader = prepare_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        is_training=False,
        num_workers=args.num_workers
    ) if val_dataset else None
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = RewardModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args
    )
    
    print_gpu_memory()
    # 开始训练
    logger.info("开始训练...")
    model, best_accuracy = trainer.train()
    
    print_gpu_memory()
    logger.info(f"训练完成! 最佳准确率: {best_accuracy:.4f}")
    
    return model, best_accuracy

if __name__ == "__main__":
    main() 