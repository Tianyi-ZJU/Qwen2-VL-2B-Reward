# Qwen2-VL 奖励模型训练

基于Qwen2-VL模型的奖励模型训练与推理代码。本项目通过在Qwen2-VL后端添加score head，将其训练成为一个视觉-语言奖励模型，可用于评估VL对话质量。

## 项目结构

```
.
├── models.py           # 模型定义
├── data.py             # 数据加载
├── utils.py            # 工具函数
├── trainer.py          # 训练逻辑
├── main.py             # 训练入口
├── test_reward_model.py # 测试脚本
├── reward_prompt.py    # 奖励提示模板
├── train.sh            # 训练脚本
├── README.md           # 项目说明
├── pretrained/         # 预训练模型
│   └── qwen/           # Qwen2-VL模型
├── dataset/            # 数据集
├── checkpoints/        # 模型检查点
└── results/            # 测试结果
```

## 环境要求

参照qwen2-vl的要求

## 数据准备

数据格式为JSONL，每行包含以下字段：

```json
{
  "id": "样本唯一标识符",
  "query": "问题文本",
  "image": "图像文件路径",
  "response": ["回答1", "回答2"],
  "human_ranking": [0, 1]  // [0, 1]表示第一个回答更好，[1, 0]表示第二个回答更好
}
```

## 训练模型

### 基本用法

```bash
# 使用脚本运行
bash train.sh

# 或直接使用Python
python main.py \
    --base_model_name pretrained/qwen \
    --train_data_path dataset/train_data.jsonl \
    --val_data_path dataset/val_data.jsonl \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 10 \
    --experiment_name my_reward_model
```

### 高级选项

```bash
python main.py \
    --base_model_name pretrained/qwen \
    --train_data_path dataset/train_data.jsonl \
    --val_data_path dataset/val_data.jsonl \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --use_flash_attention \
    --use_scheduler \
    --warmup_ratio 0.1 \
    --num_workers 4 \
    --seed 42 \
    --experiment_name margin_loss_test \
    --use_margin_loss \
    --margin 0.5 \
    --score_min -5.0 \
    --score_max 5.0 \
    --use_wandb \
    --wandb_project qwen_reward \
    --wandb_api_key YOUR_API_KEY \
    --log_samples
```

### 恢复训练

```bash
# 继承所有训练状态（学习率、epoch等）
python main.py \
    --resume_from_checkpoint checkpoints/experiment_name_timestamp/checkpoint_epoch_5.pt \
    --train_data_path dataset/train_data.jsonl \
    --val_data_path dataset/val_data.jsonl

# 仅加载模型权重，不继承训练状态
python main.py \
    --resume_from_checkpoint checkpoints/experiment_name_timestamp/checkpoint_epoch_5.pt \
    --load_weights_only \
    --train_data_path dataset/train_data.jsonl \
    --val_data_path dataset/val_data.jsonl
```

## 测试模型

```bash
# 使用脚本运行
bash test_reward_model.sh

# 或直接使用Python
python test_reward_model.py \
    --model_path checkpoints/experiment_name_timestamp/best_model_experiment_name_timestamp.pt \
    --base_model_name pretrained/qwen \
    --data_path test_data.parquet \
    --output_path results/reward_results.jsonl
```

## 主要特性

- **模型优化**：使用Flash Attention 2加速
- **训练优化**：
  - 梯度裁剪、学习率预热、早停机制
  - 支持边界损失(margin loss)，可通过`--use_margin_loss`和`--margin`参数控制
  - 可配置奖励分数范围，默认为-5到5
- **训练管理**：
  - 实验名称与时间戳确保检查点不会被覆盖
  - 可选择仅加载模型权重而不继承训练状态
  - 保存最佳模型、自动清理旧检查点
- **数据处理**：支持多进程数据加载、错误处理
- **可视化监控**：与Weights & Biases集成
- **模块化设计**：易于扩展和定制

## 重要参数说明

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--experiment_name` | 实验名称，用于区分不同训练运行 | reward_model |
| `--use_margin_loss` | 是否使用边界损失 | False |
| `--margin` | 边界损失的边界值 | 0.5 |
| `--score_min` | 奖励分数最小值 | -5.0 |
| `--score_max` | 奖励分数最大值 | 5.0 |
| `--load_weights_only` | 仅加载模型权重，不继承训练状态 | False |

### 检查点管理

每次训练会在`checkpoints`目录下创建一个唯一的子目录`experiment_name_timestamp`，所有检查点都保存在其中，确保不会被覆盖。

### 测试结果

测试结果会保存为包含模型名称和时间戳的唯一文件名，避免覆盖之前的结果。

## 模型架构

该项目使用了以下模型架构：
- 基础模型：Qwen2-VL-2B-Instruct
- Score Head：两层MLP，隐藏层维度与基础模型相同
- 输出：单一标量值，表示回答质量的分数