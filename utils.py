import torch
import os
import shutil
import random
import numpy as np
import logging
import wandb
from tqdm import tqdm
from PIL import Image
from reward_prompt import reward_prompt

def setup_logging(log_file='reward_training.log'):
    """
    设置日志
    
    参数:
        log_file (str): 日志文件路径
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def set_seed(seed):
    """
    设置随机种子，保证结果可重复
    
    参数:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"设置随机种子: {seed}")

def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, is_best, args):
    """
    保存检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch (int): 当前轮次
        loss (float): 当前损失
        accuracy (float): 当前准确率
        is_best (bool): 是否是最佳模型
        args: 参数
    """
    import datetime
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 尝试从args中获取experiment_name，如果不存在则使用默认值
    experiment_name = getattr(args, 'experiment_name', 'default')
    
    # 如果是第一次保存检查点，则在args中记录时间戳，以便后续使用相同的时间戳
    if not hasattr(args, 'timestamp'):
        args.timestamp = timestamp
    else:
        # 使用已记录的时间戳，确保同一次训练的所有检查点使用相同的时间戳
        timestamp = args.timestamp
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'accuracy': accuracy,
        'args': args
    }
    
    # 创建目录结构：checkpoints/experiment_name_timestamp/
    experiment_dir = os.path.join(args.checkpoint_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存最新的检查点
    checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
    logger.info(f"保存检查点到 {checkpoint_path}")
    
    # 如果是最佳模型，保存一份副本
    if is_best:
        best_path = os.path.join(experiment_dir, f'best_model_{experiment_name}_{timestamp}.pt')
        shutil.copy2(checkpoint_path, best_path)
        logger.info(f"保存最佳模型到 {best_path}")
        
        # 将最佳模型上传到wandb
        if hasattr(args, 'use_wandb') and args.use_wandb:
            wandb.save(best_path)
    
    # 如果启用了保存最新N个检查点
    if hasattr(args, 'keep_last_n') and args.keep_last_n > 0:
        checkpoints = sorted([f for f in os.listdir(experiment_dir) if f.startswith('checkpoint_')])
        if len(checkpoints) > args.keep_last_n:
            for old_checkpoint in checkpoints[:-args.keep_last_n]:
                old_path = os.path.join(experiment_dir, old_checkpoint)
                os.remove(old_path)
                logger.info(f"删除旧检查点 {old_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    加载检查点，用于继续训练
    
    参数:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        checkpoint_path (str): 检查点路径
        
    返回:
        model: 加载后的模型
        optimizer: 加载后的优化器
        scheduler: 加载后的学习率调度器
        epoch (int): 当前轮次
        loss (float): 当前损失
        accuracy (float): 当前准确率
    """
    logger.info(f"从 {checkpoint_path} 加载检查点")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', float('inf'))
    accuracy = checkpoint.get('accuracy', 0.0)
    
    return model, optimizer, scheduler, epoch, loss, accuracy

def load_model_weights_only(model, checkpoint_path):
    """
    仅加载模型权重，不加载优化器、调度器等训练状态
    
    参数:
        model: 模型
        checkpoint_path (str): 检查点路径
        
    返回:
        model: 加载权重后的模型
    """
    logger.info(f"从 {checkpoint_path} 仅加载模型权重")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 仅加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 记录检查点中的一些信息，但不返回
    epoch = checkpoint['epoch']
    accuracy = checkpoint.get('accuracy', 0.0)
    logger.info(f"加载的检查点来自第 {epoch} 轮，准确率: {accuracy:.4f}")
    logger.info(f"注意：训练将从第0轮开始，不继承之前的训练状态")
    
    return model

def evaluate(model, val_loader, device, score_range=(-5, 5), use_margin_loss=False, margin=0.5):
    """
    评估模型
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        score_range: 打分范围，默认为(-5, 5)
        use_margin_loss: 是否使用边界损失
        margin: 边界损失的边界值
        
    返回:
        accuracy (float): 准确率
        avg_loss (float): 平均损失
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    # 获取模型所在设备
    model_device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="评估中"):
            # 不需要将images移动到设备，因为现在是路径列表
            images = batch['image']  # 这是图像路径列表
            queries = batch['query']
            responses1 = batch['response1']
            responses2 = batch['response2']
            rankings = batch['ranking'].to(model_device)
            
            # 使用reward_prompt函数构建完整的提示
            texts1 = [reward_prompt(q, r, score_range=score_range) for q, r in zip(queries, responses1)]
            texts2 = [reward_prompt(q, r, score_range=score_range) for q, r in zip(queries, responses2)]
            
            scores1 = model(images, texts1)
            scores2 = model(images, texts2)
            
            # 计算预测是否正确
            # [0, 1]表示第一个回答更好，[1, 0]表示第二个回答更好
            predictions = (scores1 > scores2).float()
            # 如果rankings[:, 1] > rankings[:, 0]，则第一个回答更好
            correct = (predictions == (rankings[:, 1] > rankings[:, 0]).float()).sum().item()
            
            # 计算损失
            better_scores = torch.where(rankings[:, 1] > rankings[:, 0], scores1, scores2)
            worse_scores = torch.where(rankings[:, 1] > rankings[:, 0], scores2, scores1)
            
            # 计算分数差异
            score_diff = better_scores - worse_scores
            
            
            # 使用放大因子增加梯度信号
            scale_factor = 6.0  # 与训练时相同的放大因子
            scaled_diff = scale_factor * score_diff
            
            # 使用修改后的损失函数
            epsilon = 1e-7
            ranking_loss = -torch.log(torch.sigmoid(scaled_diff) + epsilon).mean().item()
            
            # 根据参数决定是否使用边界损失
            if use_margin_loss:
                # 添加边界损失(margin loss)
                margin_loss = torch.clamp(margin - score_diff, min=0.0).mean().item()
                # 组合两种损失
                loss = ranking_loss + margin_loss
            else:
                # 只使用排序损失
                loss = ranking_loss
            
            total_correct += correct
            total_samples += len(rankings)
            total_loss += loss
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
    return accuracy, avg_loss

def setup_wandb(args):
    """
    设置Weights & Biases
    
    参数:
        args: 参数
        
    返回:
        bool: 是否成功设置
    """
    if not args.use_wandb:
        return False
        
    # 直接在代码中登录wandb
    if args.wandb_api_key:
        try:
            wandb.login(key=args.wandb_api_key)
            logger.info("成功登录到Weights & Biases!")
        except Exception as e:
            logger.error(f"wandb登录失败: {e}")
            if args.wandb_optional:
                args.use_wandb = False
                logger.warning("由于登录失败，wandb功能已禁用。")
                return False
            else:
                raise e
    
    import datetime
    run_name = f"qwen-reward-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        job_type="training"
    )
    logger.info(f"wandb运行已初始化: {run_name}")
    return True

def log_batch_to_wandb(batch, scores1, scores2, rankings, loss, step):
    """
    记录一个批次的数据到wandb
    
    参数:
        batch: 批次数据
        scores1: 第一个回答的分数
        scores2: 第二个回答的分数
        rankings: 排名
        loss: 损失
        step: 步数
    """
    # 选择最多3个样本
    num_samples = min(3, len(batch['query']))
    
    for i in range(num_samples):
        query = batch['query'][i]
        response1 = batch['response1'][i]
        response2 = batch['response2'][i]
        score1 = scores1[i].item()
        score2 = scores2[i].item()
        ranking = rankings[i].tolist()
        
        # 记录文本数据
        wandb.log({
            f"sample_{i}/query": query,
            f"sample_{i}/response1": response1,
            f"sample_{i}/response2": response2,
            f"sample_{i}/score1": score1,
            f"sample_{i}/score2": score2,
            f"sample_{i}/human_ranking": ranking,
            "global_step": step
        })
        
        # 记录图像
        try:
            image = batch['image'][i]
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy().transpose(1, 2, 0)
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            wandb.log({
                f"sample_{i}/image": wandb.Image(
                    image, 
                    caption=f"Query: {query[:50]}..."
                ),
                "global_step": step
            })
        except Exception as e:
            logger.warning(f"记录图像到wandb失败: {e}")

def log_metrics_to_wandb(metrics, global_step, prefix=''):
    """
    记录指标到wandb
    
    参数:
        metrics (dict): 指标字典
        global_step (int): 当前步数
        prefix (str): 指标前缀
    """
    if not wandb.run:
        return
        
    log_dict = {}
    for k, v in metrics.items():
        log_dict[f"{prefix}/{k}" if prefix else k] = v
    log_dict["global_step"] = global_step
    
    wandb.log(log_dict) 