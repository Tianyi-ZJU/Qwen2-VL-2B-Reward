import torch
from tqdm import tqdm
import logging
from transformers import get_linear_schedule_with_warmup
import wandb
import os

from utils import (
    save_checkpoint, load_checkpoint, evaluate, setup_wandb,
    log_batch_to_wandb, log_metrics_to_wandb, load_model_weights_only
)
from reward_prompt import reward_prompt
import random

logger = logging.getLogger(__name__)

class RewardModelTrainer:
    """
    奖励模型训练器
    """
    def __init__(self, model, train_loader, val_loader=None, args=None):
        """
        初始化训练器
        
        参数:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            args: 训练参数
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        
        # 获取设备信息
        self.device = next(model.parameters()).device
        logger.info(f"模型运行在设备: {self.device}")
        
        # 优化器 - 只优化score_head的参数
        trainable_params = [p for p in model.score_head.parameters() if p.requires_grad]
        logger.info(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
        
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-5,  # 增大eps值以提高稳定性
            betas=(0.9, 0.999)  # 默认值
        )
        
        # 设置梯度累积步数，默认为1（不累积）
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        
        # 学习率调度器
        self.total_steps = len(train_loader) * args.num_epochs // self.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.total_steps * args.warmup_ratio),
            num_training_steps=self.total_steps
        ) if args.use_scheduler else None
        
        # 训练状态
        self.start_epoch = 0
        self.best_accuracy = 0
        self.early_stop_counter = 0
        self.global_step = 0
        
        # 如果提供了检查点路径，从检查点恢复
        if args.resume_from_checkpoint:
            if hasattr(args, 'load_weights_only') and args.load_weights_only:
                # 仅加载模型权重，不继承训练状态
                self.model = load_model_weights_only(self.model, args.resume_from_checkpoint)
                logger.info("仅加载模型权重，从第0轮开始训练")
            else:
                # 加载完整检查点，继承训练状态
                self.model, self.optimizer, self.scheduler, self.start_epoch, _, self.best_accuracy = load_checkpoint(
                    self.model, self.optimizer, self.scheduler, args.resume_from_checkpoint
                )
                self.global_step = self.start_epoch * len(train_loader)
                logger.info(f"从检查点恢复训练，起始轮次: {self.start_epoch}, 最佳准确率: {self.best_accuracy:.4f}")
        
        # 初始化wandb
        self.use_wandb = False
        if hasattr(args, 'use_wandb') and args.use_wandb:
            self.use_wandb = setup_wandb(args)
    
    def train(self):
        """
        训练模型
        
        返回:
            model: 训练后的模型
            best_accuracy: 最佳准确率
        """
        args = self.args
        
        # 训练循环
        for epoch in range(self.start_epoch, args.num_epochs):
            # 训练一个epoch
            train_loss = self._train_epoch(epoch)
            
            # 验证
            if self.val_loader:
                score_min = getattr(args, 'score_min', -5.0)
                score_max = getattr(args, 'score_max', 5.0)
                score_range = (score_min, score_max)
                
                # 获取边界损失相关参数
                use_margin_loss = getattr(args, 'use_margin_loss', False)
                margin = getattr(args, 'margin', 0.5)
                
                accuracy, val_loss = evaluate(
                    self.model, 
                    self.val_loader, 
                    args.device, 
                    score_range=score_range,
                    use_margin_loss=use_margin_loss,
                    margin=margin
                )
                
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {accuracy:.4f}")
                
                # 记录验证信息到wandb
                if self.use_wandb:
                    log_metrics_to_wandb({
                        "loss": val_loss,
                        "accuracy": accuracy,
                        "epoch": epoch + 1
                    }, self.global_step, prefix="val")
                    
                    log_metrics_to_wandb({
                        "epoch_loss": train_loss,
                        "epoch": epoch + 1
                    }, self.global_step, prefix="train")
                
                # 检查是否是最佳模型
                is_best = accuracy > self.best_accuracy
                if is_best:
                    self.best_accuracy = accuracy
                    self.early_stop_counter = 0
                    logger.info(f"发现新的最佳模型，准确率: {accuracy:.4f}")
                else:
                    self.early_stop_counter += 1
                    logger.info(f"准确率未提升，早停计数器: {self.early_stop_counter}/{args.early_stop}")
            else:
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}, 训练损失: {train_loss:.4f}")
                
                # 记录训练信息到wandb
                if self.use_wandb:
                    log_metrics_to_wandb({
                        "epoch_loss": train_loss,
                        "epoch": epoch + 1
                    }, self.global_step, prefix="train")
                
                is_best = False
            
            # 保存检查点
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch + 1, train_loss, 
                    accuracy if self.val_loader else 0, is_best, args
                )
            
            # 早停
            if args.early_stop > 0 and self.early_stop_counter >= args.early_stop:
                logger.info(f"早停触发，在 {epoch + 1} 轮后停止训练")
                break
        
        # 结束wandb
        if self.use_wandb:
            wandb.finish()
            logger.info("训练完成，wandb运行已结束")
        else:
            logger.info("训练完成")
            
        return self.model, self.best_accuracy
    
    def _train_epoch(self, epoch):
        """
        训练一个epoch
        
        参数:
            epoch (int): 当前轮次
            
        返回:
            float: 平均损失
        """
        self.model.train()
        total_loss = 0
        epoch_step = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        # 梯度累积相关变量
        accumulated_loss = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # 准备输入
            images = batch['image'] 
            queries = batch['query']
            responses1 = batch['response1']
            responses2 = batch['response2']
            rankings = batch['ranking'].to(self.device)
            
            # 获取分数范围
            score_min = getattr(self.args, 'score_min', -5.0)
            score_max = getattr(self.args, 'score_max', 5.0)
            score_range = (score_min, score_max)
            
            # 使用reward_prompt函数构建完整的提示
            texts1 = [reward_prompt(q, r, score_range=score_range) for q, r in zip(queries, responses1)]
            texts2 = [reward_prompt(q, r, score_range=score_range) for q, r in zip(queries, responses2)]
            
            # 前向传播
            scores1 = self.model(images, texts1)
            scores2 = self.model(images, texts2)
            
            # logger.info(f"scores1: {scores1}")
            # logger.info(f"scores2: {scores2}")
            
            # 计算对比损失
            # [0, 1]表示第一个回答更好，[1, 0]表示第二个回答更好
            # 如果rankings[:, 1] > rankings[:, 0]，则第一个回答更好
            better_scores = torch.where(rankings[:, 1] > rankings[:, 0], scores1, scores2)
            worse_scores = torch.where(rankings[:, 1] > rankings[:, 0], scores2, scores1)

            # 随机选一组对应的better_scores和worse_scores打印出来
            random_index = random.randint(0, len(better_scores) - 1)
            logger.info(f"better_scores: {better_scores[random_index]}")
            logger.info(f"worse_scores: {worse_scores[random_index]}")          
            
            # 计算分数差异
            score_diff = better_scores - worse_scores
            
            # 使用放大因子增加梯度信号
            scale_factor = 4.0
            scaled_diff = scale_factor * score_diff
            
            # 使用修改后的损失函数，增加数值稳定性
            epsilon = 1e-7
            ranking_loss = -torch.log(torch.sigmoid(scaled_diff) + epsilon).mean()
            
            # 根据参数决定是否使用边界损失
            if hasattr(self.args, 'use_margin_loss') and self.args.use_margin_loss:
                # 获取边界值，默认为0.5
                margin = getattr(self.args, 'margin', 0.5)
                # 添加边界损失(margin loss)
                margin_loss = torch.clamp(margin - score_diff, min=0.0).mean()
                # 组合两种损失
                loss = ranking_loss + margin_loss
                
                # 记录日志
                if epoch == 0 and batch_idx < 5:
                    logger.info(f"使用边界损失，边界值: {margin}")
            else:
                # 只使用排序损失
                loss = ranking_loss
                margin_loss = torch.tensor(0.0, device=ranking_loss.device)  # 创建一个零张量用于日志记录
            
            # 打印分数差异的统计信息（仅在第一个epoch的前几个批次）
            if epoch == 0 and batch_idx < 5:
                with torch.no_grad():
                    logger.info(f"分数差异统计: 最小={score_diff.min().item():.6f}, 最大={score_diff.max().item():.6f}, 平均={score_diff.mean().item():.6f}, 标准差={score_diff.std().item():.6f}")
                    logger.info(f"better_scores统计: 最小={better_scores.min().item():.6f}, 最大={better_scores.max().item():.6f}, 平均={better_scores.mean().item():.6f}")
                    logger.info(f"worse_scores统计: 最小={worse_scores.min().item():.6f}, 最大={worse_scores.max().item():.6f}, 平均={worse_scores.mean().item():.6f}")
                    logger.info(f"ranking_loss: {ranking_loss.item():.6f}, margin_loss: {margin_loss.item():.6f}, 总损失: {loss.item():.6f}")
                    
                    # 计算正确预测的比例
                    predictions = (scores1 > scores2).float()
                    correct = (predictions == (rankings[:, 1] > rankings[:, 0]).float()).sum().item()
                    accuracy = correct / len(rankings)
                    logger.info(f"批次准确率: {accuracy:.4f}")
            
            # 根据梯度累积步数缩放损失
            loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 清理不必要的变量，减少内存占用
            
            
            # 累积损失
            accumulated_loss += loss.item()
            
            # 打印梯度信息以进行调试
            if epoch_step == 0 and epoch == 0 and batch_idx % self.gradient_accumulation_steps == 0:
                for name, param in self.model.score_head.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        logger.info(f"参数 {name} 的梯度范数: {grad_norm}")
                        
                        # 如果梯度过小，可能存在问题
                        if grad_norm < 1e-5:
                            logger.warning(f"参数 {name} 的梯度非常小: {grad_norm}")
            
            # 每accumulation_steps步更新一次参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.score_head.parameters(), self.args.max_grad_norm)
                
                # 更新参数
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                # 记录当前批次的损失
                current_loss = accumulated_loss
                accumulated_loss = 0
                
                # 强制进行垃圾回收
                if batch_idx % 8 == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 更新进度条
                total_loss += current_loss
                self.global_step += 1
                epoch_step += 1
                
                progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
                
                # 记录训练信息到wandb
                if self.use_wandb and self.global_step % self.args.log_interval == 0:
                    # 计算当前批次准确率
                    with torch.no_grad():
                        predictions = (scores1 > scores2).float()
                        correct = (predictions == (rankings[:, 1] > rankings[:, 0]).float()).sum().item()
                        batch_accuracy = correct / len(rankings)
                    
                    log_metrics_to_wandb({
                        "loss": current_loss,
                        "batch_accuracy": batch_accuracy,
                        "step": epoch_step,
                        "epoch": epoch + 1,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    }, self.global_step, prefix="train")
                    
                    # 记录样本数据
                    if self.args.log_samples:
                        log_batch_to_wandb(batch, scores1, scores2, rankings, current_loss, self.global_step)

            del scores1, scores2, better_scores, worse_scores, score_diff, scaled_diff, random_index
        
        # 处理最后一个不完整的梯度累积批次
        if accumulated_loss > 0:
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.score_head.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += accumulated_loss
            epoch_step += 1
        
        avg_loss = total_loss / epoch_step if epoch_step > 0 else float('inf')
        logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}, 平均训练损失: {avg_loss:.4f}")
        
        return avg_loss 