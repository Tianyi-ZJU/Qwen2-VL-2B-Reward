import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import logging
from PIL import Image

logger = logging.getLogger(__name__)

from qwen_vl_utils import process_vision_info

class RewardModel(nn.Module):
    def __init__(self, base_model_name, use_flash_attention=True, pooling_method="mean"):
        super().__init__()
        logger.info(f"加载基础模型: {base_model_name}")
        
        # 确定使用的数据类型
        self.dtype = torch.float32  # 使用float32以获得更好的数值稳定性
        
        # 使用与官方代码相同的初始化方式
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="flash_attention_2" if use_flash_attention and torch.cuda.is_available() else "eager",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.processor = AutoProcessor.from_pretrained(base_model_name)
        
        # 设置池化方法
        self.pooling_method = pooling_method
        logger.info(f"使用池化方法: {pooling_method}")
        
        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 添加简单的score head
        hidden_size = self.base_model.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),  # 使用GELU代替ReLU，避免死亡神经元问题
            nn.Linear(hidden_size, 1)
        ).to(dtype=self.dtype)  # 使用float32
        
        # 使用特殊的初始化方法，确保输出有足够的方差
        for m in self.score_head.modules():
            if isinstance(m, nn.Linear):
                # 使用较大的初始值，增加输出的方差
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    # 使用非零偏置，打破对称性
                    nn.init.constant_(m.bias, 0.1)
        
        # 将score head移动到与base_model相同的设备
        if torch.cuda.is_available():
            self.score_head = self.score_head.to(self.base_model.device)
        
        # 确保score_head的参数可训练
        for param in self.score_head.parameters():
            param.requires_grad = True
        
        logger.info(f"模型初始化完成，hidden_size: {hidden_size}，dtype: {self.dtype}")
        logger.info(f"Score head参数数量: {sum(p.numel() for p in self.score_head.parameters() if p.requires_grad)}")
    
    def _get_sequence_representation(self, hidden_states, attention_mask=None):
        """
        根据设置的池化方法提取序列表示
        
        Args:
            hidden_states: 模型最后一层的隐藏状态
            attention_mask: 注意力掩码，用于忽略padding
            
        Returns:
            sequence_representation: 提取的序列表示
        """
        if self.pooling_method == "cls":
            # 使用[CLS]表示（第一个token）
            return hidden_states[:, 0, :]
        
        elif self.pooling_method == "last":
            if attention_mask is not None:
                last_indices = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                sequence_representation = torch.stack([
                    hidden_states[i, last_indices[i], :] 
                    for i in range(batch_size)
                ])
                return sequence_representation
            else:
                # 如果没有mask，使用最后一个token
                return hidden_states[:, -1, :]
        
        elif self.pooling_method == "mean":
            # 使用所有token的平均值
            if attention_mask is not None:
                sum_hidden = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
                seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True)
                seq_lengths = torch.clamp(seq_lengths, min=1e-9)
                return sum_hidden / seq_lengths
            else:
                # 如果没有mask，计算所有token的平均值
                return torch.mean(hidden_states, dim=1)
        
        elif self.pooling_method == "max":
            # 使用所有token的最大值
            if attention_mask is not None:
                masked_hidden = hidden_states + (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
                return torch.max(masked_hidden, dim=1)[0]
            else:
                return torch.max(hidden_states, dim=1)[0]
        
        elif self.pooling_method == "cls_mean":
            cls_repr = hidden_states[:, 0, :]
            
            if attention_mask is not None:
                sum_hidden = torch.sum(hidden_states[:, 1:, :] * attention_mask[:, 1:].unsqueeze(-1), dim=1)
                seq_lengths = torch.sum(attention_mask[:, 1:], dim=1, keepdim=True)
                seq_lengths = torch.clamp(seq_lengths, min=1e-9)
                mean_repr = sum_hidden / seq_lengths
            else:
                mean_repr = torch.mean(hidden_states[:, 1:, :], dim=1)
            
            return (cls_repr + mean_repr) / 2.0
        
        else:
            logger.warning(f"未知的池化方法: {self.pooling_method}，使用默认的CLS表示")
            return hidden_states[:, 0, :]
        
    def forward(self, images, texts):
        batch_size = len(texts)
        
        # 准备输入，按照官方处理方式
        all_inputs = []
        for i in range(batch_size):
            # 处理不同类型的图像输入
            # TODO: 历史遗留问题，这里应该只能是str
            if isinstance(images[i], str):
                # 如果是路径字符串，直接使用
                image_input = images[i]
            elif isinstance(images[i], torch.Tensor):
                # 将tensor转换为PIL图像
                image_tensor = images[i].cpu().numpy()
                image_tensor = (image_tensor * 255).astype('uint8')
                image_tensor = image_tensor.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                image_input = Image.fromarray(image_tensor)
            else:
                image_input = images[i]
                
            # 构建消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_input},
                        {"type": "text", "text": texts[i]}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            
            # 处理输入
            image_inputs, _ = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # 将输入移动到正确的设备 - 修复字典处理
            inputs = inputs.to(self.base_model.device)
            all_inputs.append(inputs)
        
        # 逐个处理输入并获取表示
        all_representations = []
        for inputs in all_inputs:
            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                sequence_representation = self._get_sequence_representation(
                    last_hidden_states, 
                    attention_mask=inputs.get('attention_mask')
                )
                all_representations.append(sequence_representation)
        
        # 将所有表示拼接起来，并分离以确保梯度流向score_head
        sequence_representations = torch.cat(all_representations, dim=0).detach()
        
        sequence_representations = sequence_representations.to(self.dtype)
        
        # 通过score head得到分数
        scores = self.score_head(sequence_representations)
        return scores 