# Optimized TabTransformer v2.0

## 📋 优化总结

| 优化项 | 原版 | 优化版 | 预期收益 |
|--------|------|--------|----------|
| **架构** |
| LayerNorm 位置 | Post-LN | Pre-LN | 更稳定训练，避免梯度爆炸 |
| 激活函数 | ReLU | GELU | 更平滑梯度，加速收敛 |
| 注意力机制 | 分离 QKV | 合并 QKV | 减少内存占用 10-15% |
| 列嵌入 | ❌ | ✅ | 区分不同特征列 |
| 数值特征处理 | LayerNorm | Tokenization | 更好的特征交互 (FT-Transformer) |
| 特征聚合 | Flatten | CLS Token | 参数更少，更高效 |
| MLP | 普通线性层 | 残差连接 | 更深网络更稳定 |
| **训练** |
| 损失函数 | MSELoss | SmoothL1Loss | 对异常值更鲁棒 |
| 优化器 | AdamW | AdamW + Lookahead | 更好泛化 |
| 学习率调度 | OneCycleLR | CosineAnnealingWarmRestarts | 多次重启，跳出局部最优 |
| SWA | ❌ | ✅ | 更好的泛化性能 |
| 数据增强 | ❌ | MixUp | 防止过拟合 |
| 梯度裁剪 | ✅ | ✅ | 防止梯度爆炸 |
| PyTorch 2.0 编译 | ❌ | ✅ | 加速 10-30% |

## 🚀 快速开始

### 基本使用

```python
from optimized_tab_transformer import (
    OptimizedTabTransformerTrainer,
    ModelConfig,
    TrainingConfig,
    LossType
)

# 1. 配置模型
model_config = ModelConfig(
    embedding_dim=32,
    num_heads=4,
    num_transformer_layers=3,
    d_ff=128,
    dropout=0.1,
    use_numerical_tokenization=True,  # FT-Transformer 风格
    use_cls_token=True
)

# 2. 配置训练
training_config = TrainingConfig(
    batch_size=512,
    epochs=300,
    learning_rate=1e-3,
    use_mixup=True,
    use_lookahead=True,
    use_swa=True
)

# 3. 初始化训练器
trainer = OptimizedTabTransformerTrainer(
    model_config=model_config,
    training_config=training_config
)

# 4. 准备数据
train_data, test_data = trainer.prepare_data('path/to/data.csv')

# 5. 创建数据加载器
train_loader, test_loader = trainer.create_data_loaders(train_data, test_data)

# 6. 构建模型
trainer.build_model()

# 7. 训练
history = trainer.train(train_loader, test_loader, loss_type=LossType.SMOOTH_L1)

# 8. 评估
actuals, preds, metrics = trainer.evaluate(test_loader)
```

## 🔧 详细配置说明

### ModelConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_dim` | int | 32 | 嵌入维度 |
| `num_heads` | int | 4 | 注意力头数 (建议: embedding_dim / num_heads >= 8) |
| `num_transformer_layers` | int | 3 | Transformer 层数 |
| `d_ff` | int | 128 | FFN 隐藏层维度 |
| `dropout` | float | 0.1 | Dropout 比例 |
| `hidden_dims` | List[int] | [128, 64] | MLP 隐藏层维度 |
| `use_numerical_tokenization` | bool | True | 是否使用数值特征 tokenization |
| `use_cls_token` | bool | True | 是否使用 CLS token 聚合 |

### TrainingConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | int | 512 | 批次大小 |
| `epochs` | int | 300 | 最大训练轮数 |
| `learning_rate` | float | 1e-3 | 初始学习率 |
| `weight_decay` | float | 1e-4 | 权重衰减 |
| `patience` | int | 20 | 早停耐心值 |
| `use_mixup` | bool | True | 是否使用 MixUp 数据增强 |
| `mixup_alpha` | float | 0.2 | MixUp alpha 参数 |
| `use_lookahead` | bool | True | 是否使用 Lookahead 优化器 |
| `lookahead_k` | int | 5 | Lookahead 更新步数 |
| `lookahead_alpha` | float | 0.5 | Lookahead 插值系数 |
| `use_swa` | bool | True | 是否使用 SWA |
| `swa_start_epoch` | int | 50 | SWA 开始轮数 |
| `swa_lr` | float | 5e-4 | SWA 学习率 |
| `use_weighted_sampling` | bool | True | 是否使用加权采样 |
| `sampling_alpha` | float | 0.5 | 采样权重平滑系数 |
| `gradient_clip` | float | 1.0 | 梯度裁剪阈值 |
| `use_compile` | bool | True | 是否使用 PyTorch 2.0 编译 |

## 📊 损失函数选择

```python
from optimized_tab_transformer import LossType

# 可用的损失函数
LossType.MSE         # 均方误差 (对异常值敏感)
LossType.SMOOTH_L1   # 平滑 L1 损失 (推荐，对异常值鲁棒)
LossType.HUBER       # Huber 损失
LossType.COMBINED    # MSE + L1 组合损失
```

## 🎯 超参数调优建议

### 小数据集 (< 10K 样本)
```python
model_config = ModelConfig(
    embedding_dim=16,
    num_heads=2,
    num_transformer_layers=2,
    d_ff=64,
    dropout=0.2,  # 增加 dropout
    hidden_dims=[64, 32]
)

training_config = TrainingConfig(
    batch_size=128,
    use_mixup=True,
    mixup_alpha=0.3,  # 增加 mixup 强度
    patience=30  # 更长的耐心
)
```

### 中等数据集 (10K - 100K 样本)
```python
model_config = ModelConfig(
    embedding_dim=32,
    num_heads=4,
    num_transformer_layers=3,
    d_ff=128,
    dropout=0.1,
    hidden_dims=[128, 64]
)

training_config = TrainingConfig(
    batch_size=512,
    use_mixup=True,
    use_swa=True
)
```

### 大数据集 (> 100K 样本)
```python
model_config = ModelConfig(
    embedding_dim=64,
    num_heads=8,
    num_transformer_layers=4,
    d_ff=256,
    dropout=0.1,
    hidden_dims=[256, 128, 64]
)

training_config = TrainingConfig(
    batch_size=1024,
    use_mixup=False,  # 数据足够，可以不用
    use_swa=True
)
```

## 🔍 关键优化详解

### 1. Pre-LayerNorm vs Post-LayerNorm

```
Post-LN (原版):
x -> Attention -> Add -> LayerNorm -> FFN -> Add -> LayerNorm

Pre-LN (优化版):
x -> LayerNorm -> Attention -> Add -> LayerNorm -> FFN -> Add

优点: 梯度流更稳定，支持更深的网络
```

### 2. 数值特征 Tokenization (FT-Transformer)

```python
# 原版: 仅 LayerNorm
numerical_normalized = self.layer_norm(numerical_inputs)

# 优化版: 每个数值特征变成一个 token
# [batch, num_features] -> [batch, num_features, embedding_dim]
num_tokens = self.numerical_tokenizer(numerical_inputs)
```

### 3. Lookahead 优化器

```
Fast weights: 正常的 Adam 更新
Slow weights: 每 k 步，用 fast weights 更新一次

slow = slow + alpha * (fast - slow)

效果: 更平滑的优化轨迹，更好的泛化
```

### 4. MixUp 数据增强

```python
# 数值特征: 线性插值
mixed_num = lambda * num_a + (1 - lambda) * num_b

# 类别特征: 随机选择
mixed_cat = random_choice(cat_a, cat_b)

# 目标: 线性插值
mixed_y = lambda * y_a + (1 - lambda) * y_b
```

### 5. SWA (Stochastic Weight Averaging)

```
训练后期，每个 epoch 结束后:
swa_model.update_parameters(model)

效果: 更平坦的损失面，更好的泛化
```

## 📈 性能对比 (预期)

| 指标 | 原版 | 优化版 | 提升 |
|------|------|--------|------|
| 收敛速度 | 100% | 70-80% | 20-30% 更快 |
| 最终 R² | baseline | +2-5% | 更好拟合 |
| 泛化差距 | baseline | -30-50% | 更小过拟合 |
| 推理速度 | 100% | 80-90%* | 10-20% 更快 |

*使用 PyTorch 2.0 编译

## 🐛 常见问题

### Q: 训练不稳定 / Loss 震荡
```python
# 降低学习率
training_config = TrainingConfig(learning_rate=5e-4)

# 增加梯度裁剪
training_config = TrainingConfig(gradient_clip=0.5)

# 使用更小的 batch size
training_config = TrainingConfig(batch_size=256)
```

### Q: 过拟合
```python
# 增加 dropout
model_config = ModelConfig(dropout=0.2)

# 增加 MixUp 强度
training_config = TrainingConfig(mixup_alpha=0.4)

# 减少模型容量
model_config = ModelConfig(
    num_transformer_layers=2,
    hidden_dims=[64, 32]
)
```

### Q: 欠拟合
```python
# 增加模型容量
model_config = ModelConfig(
    embedding_dim=64,
    num_transformer_layers=4,
    d_ff=256
)

# 关闭 MixUp
training_config = TrainingConfig(use_mixup=False)

# 增加训练轮数
training_config = TrainingConfig(epochs=500, patience=30)
```

### Q: GPU 内存不足
```python
# 减小 batch size
training_config = TrainingConfig(batch_size=128)

# 减小模型
model_config = ModelConfig(
    embedding_dim=16,
    d_ff=64
)

# 关闭编译 (会使用更多内存)
training_config = TrainingConfig(use_compile=False)
```

## 📁 输出文件

训练完成后会生成以下文件：

```
model_checkpoints_v2/
├── best_model_v2.pth      # 最佳验证损失模型
└── swa_model_v2.pth       # SWA 平均模型

model_output/
├── route_mapping.pkl      # 路线编码映射
└── preprocessors.pkl      # 所有预处理器

training_history_v2.png    # 训练曲线图
predictions_v2.png         # 预测分析图
```

## 🔗 参考文献

1. [TabTransformer](https://arxiv.org/abs/2012.06678) - 原始论文
2. [FT-Transformer](https://arxiv.org/abs/2106.11959) - 数值特征 Tokenization
3. [Pre-LN Transformer](https://arxiv.org/abs/2002.04745) - Pre-LayerNorm
4. [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) - Lookahead
5. [MixUp](https://arxiv.org/abs/1710.09412) - 数据增强
6. [SWA](https://arxiv.org/abs/1803.05407) - 权重平均
