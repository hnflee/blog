#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized TabTransformer Training Script V2
优化点：
1. Focal Loss + Label Smoothing 处理类别不平衡
2. 增大模型容量（embedding_dim=64, 更深MLP）
3. GELU 激活函数（更适合Transformer）
4. 学习率预热 + CosineAnnealingWarmRestarts 调度器
5. 类别权重平衡
6. 可选 SWA (Stochastic Weight Averaging)

FIX: 修复 dummy_cat 生成时索引越界问题
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import os
import warnings
import math
import argparse
import pickle
from collections import Counter

warnings.filterwarnings('ignore')

# Global Configuration - 优化后的配置
NUM_HEADS = 8  # 增加注意力头数
NUM_TRANSFORMER_LAYERS = 8  # 增加层数
EMBEDDING_DIM = 64  # 增大 embedding 维度


class TabularDataset(Dataset):
    """Custom dataset for tabular data with mixed feature types"""
    def __init__(self, categorical_features: torch.Tensor, numerical_features: torch.Tensor, targets: torch.Tensor):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.categorical_features[idx], self.numerical_features[idx]), self.targets[idx]


class FeatureEmbedding(nn.Module):
    """Embedding layer for categorical features"""
    def __init__(self, cardinalities: List[int], embedding_dim: int):
        super(FeatureEmbedding, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in cardinalities
        ])
        self.embedding_dim = embedding_dim

    def forward(self, categorical_inputs):
        embedded = [
            embedding(categorical_inputs[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return torch.stack(embedded, dim=1)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, V), attention_weights

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        return self.w_o(attention_output)


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function"""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class TransformerBlock(nn.Module):
    """Transformer encoder block with GELU activation"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 使用 GELU 替代 ReLU（更适合 Transformer）
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            GELU(),  # 使用 GELU 激活函数
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重（tensor）
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # 计算交叉熵损失（不使用权重，因为会在focal loss中处理）
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        # 计算 p_t (预测概率)
        p = torch.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 计算 focal loss
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        num_classes = pred.size(-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = torch.sum(-true_dist * log_prob, dim=-1)
        
        # 应用类别权重
        if self.weight is not None:
            loss = loss * self.weight[target]
        
        return loss.mean()


class TabTransformer(nn.Module):
    """TabTransformer model for tabular data prediction"""
    def __init__(self,
                 categorical_cardinalities: List[int],
                 numerical_features_dim: int,
                 embedding_dim: int = EMBEDDING_DIM,
                 num_heads: int = NUM_HEADS,
                 num_transformer_layers: int = NUM_TRANSFORMER_LAYERS,
                 d_ff: int = 128,
                 dropout: float = 0.1,
                 hidden_dims: List[int] = [128, 64],
                 output_dim: int = 12):  # Changed default, will be overridden
        super(TabTransformer, self).__init__()

        self.categorical_embedding = FeatureEmbedding(categorical_cardinalities, embedding_dim)
        self.numerical_layer_norm = nn.LayerNorm(numerical_features_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_transformer_layers)
        ])

        combined_dim = len(categorical_cardinalities) * embedding_dim + numerical_features_dim

        mlp_layers = []
        prev_dim = combined_dim

        # 使用 GELU 激活函数，加深 MLP 层
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                GELU(),  # 使用 GELU 替代 ReLU
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 修正：应用显式权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for better convergence"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot init works better with GELU
            # 对于GELU，使用较小的初始化范围
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            # Embeddings standard init
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, categorical_inputs, numerical_inputs):
        categorical_embeddings = self.categorical_embedding(categorical_inputs)

        for transformer_layer in self.transformer_layers:
            categorical_embeddings = transformer_layer(categorical_embeddings)

        categorical_flat = categorical_embeddings.view(categorical_embeddings.size(0), -1)
        numerical_normalized = self.numerical_layer_norm(numerical_inputs)

        combined = torch.cat([categorical_flat, numerical_normalized], dim=1)
        output = self.mlp(combined)  # Removed squeeze for classification

        return output


class TabTransformerTrainer:
    """Trainer class for TabTransformer model"""
    def __init__(self, model_save_dir: str, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.model_save_dir = model_save_dir
        self.device = self._setup_device()
        self._set_random_seeds()

        # Feature Definitions (Keep yours)
        self.categorical_features = ['seasonality', 'tod', 'day_prior_seg', 'dow', 'if_first_carrier', 'if_last_carrier']
        self.numerical_features = ['flt_duration', 'public_fare', 'price_1', 'cap_share_thisflt', 'cap_share_czflts', 
                                   'lf', 'rask', 'yield', 'incremental_lf', 'incremental_lf_cz', 'incremental_rask', 
                                   'incremental_rask_cz', 'incremental_yield', 'lf_cz', 'rask_cz', 'lf_yoy', 'rask_yoy']
        self.target_feature = 'pre_class'

        self.label_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.cardinalities = {}
        self.class_weights = None  # 用于类别权重平衡

    def _setup_device(self):
        if torch.cuda.is_available():
            try:
                device = torch.device('cuda')
                print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                torch.backends.cudnn.benchmark = True
                return device
            except Exception as e:
                print(f"⚠️ GPU init failed: {e}")
        print("💻 Using CPU")
        return torch.device('cpu')

    def _set_random_seeds(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def prepare_data(self, data_path: str, test_size: float = 0.2) -> Tuple:
        print("Preparing data...")
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Raw dataset size: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # 检查必要的列是否存在
        missing_cat = [f for f in self.categorical_features if f not in df.columns]
        missing_num = [f for f in self.numerical_features if f not in df.columns]
        missing_target = [self.target_feature] if self.target_feature not in df.columns else []
        
        if missing_cat or missing_num or missing_target:
            raise ValueError(
                f"Missing required columns:\n"
                f"  Categorical: {missing_cat}\n"
                f"  Numerical: {missing_num}\n"
                f"  Target: {missing_target}"
            )
        
        df_clean = df.dropna().reset_index(drop=True)
        print(f"Dataset size after cleaning: {df_clean.shape}")
        print(f"  Dropped {len(df) - len(df_clean)} rows with NaN")

        # Cardinalities
        print(f"\n=== Categorical Features ===")
        for feature in self.categorical_features:
            self.cardinalities[feature] = df_clean[feature].nunique()
            unique_vals = df_clean[feature].unique()[:5]  # 显示前5个唯一值
            print(f"  {feature}: {self.cardinalities[feature]} unique values (sample: {unique_vals})")

        # Categorical Encoding
        categorical_data = {}
        for feature in self.categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            categorical_data[feature] = self.label_encoders[feature].fit_transform(df_clean[feature])
        
        print(f"\n=== Numerical Features ===")
        print(f"  Number of numerical features: {len(self.numerical_features)}")
        numerical_data = df_clean[self.numerical_features].values
        print(f"  Numerical data shape: {numerical_data.shape}")
        print(f"  Numerical data stats:")
        print(f"    Min: {numerical_data.min(axis=0)}, Max: {numerical_data.max(axis=0)}")
        print(f"    Mean: {numerical_data.mean(axis=0)[:3]}... (showing first 3)")
        print(f"    Std: {numerical_data.std(axis=0)[:3]}... (showing first 3)")

        # Numerical Scaling
        numerical_data_scaled = self.numerical_scaler.fit_transform(numerical_data)
        print(f"  After scaling - Mean: {numerical_data_scaled.mean(axis=0)[:3]}... (should be ~0)")
        print(f"  After scaling - Std: {numerical_data_scaled.std(axis=0)[:3]}... (should be ~1)")

        # Target Encoding
        targets = df_clean[self.target_feature].values
        self.target_encoder.fit(targets)
        targets_encoded = self.target_encoder.transform(targets)
        targets_tensor = torch.tensor(targets_encoded, dtype=torch.long)
        
        # 打印类别分布信息
        num_classes = len(self.target_encoder.classes_)
        class_counts = Counter(targets_encoded)
        total_samples = len(targets_encoded)
        
        print(f"\n=== Target Distribution ===")
        print(f"Total samples: {total_samples}")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {self.target_encoder.classes_}")
        print(f"\nClass distribution:")
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 0)
            percentage = count / total_samples * 100
            print(f"  Class {class_idx} ({self.target_encoder.classes_[class_idx]}): {count} ({percentage:.2f}%)")
        
        # 计算权重：使用 inverse frequency 方法
        self.class_weights = torch.zeros(num_classes, dtype=torch.float32)
        for class_idx, count in class_counts.items():
            self.class_weights[class_idx] = total_samples / (num_classes * count)
        
        print(f"\nClass weights: {self.class_weights.numpy()}")
        print(f"Min weight: {self.class_weights.min():.4f}, Max weight: {self.class_weights.max():.4f}")

        # To Tensors
        categorical_tensor = torch.stack([
            torch.tensor(categorical_data[feature], dtype=torch.long)
            for feature in self.categorical_features
        ], dim=1)
        numerical_tensor = torch.tensor(numerical_data_scaled, dtype=torch.float32)

        # Split - 检查是否可以使用分层采样
        indices = np.arange(len(categorical_tensor))
        
        # 检查每个类别的样本数，确保所有类别至少有2个样本才能使用分层采样
        class_counts = Counter(targets_encoded)
        min_class_count = min(class_counts.values())
        use_stratify = min_class_count >= 2
        
        if use_stratify:
            print(f"\n=== Data Split (Stratified) ===")
            try:
                train_indices, test_indices = train_test_split(
                    indices, test_size=test_size, random_state=self.random_state, stratify=targets_encoded
                )
            except ValueError as e:
                print(f"  Warning: Stratified split failed: {e}")
                print(f"  Falling back to non-stratified split")
                use_stratify = False
        
        if not use_stratify:
            print(f"\n=== Data Split (Non-Stratified) ===")
            print(f"  Warning: Some classes have too few samples (< 2), using non-stratified split")
            print(f"  Minimum class count: {min_class_count}")
            train_indices, test_indices = train_test_split(
                indices, test_size=test_size, random_state=self.random_state
            )
        
        print(f"  Train samples: {len(train_indices)} ({len(train_indices)/len(indices)*100:.1f}%)")
        print(f"  Test samples: {len(test_indices)} ({len(test_indices)/len(indices)*100:.1f}%)")
        
        # 检查训练集和测试集的类别分布
        train_targets = targets_encoded[train_indices]
        test_targets = targets_encoded[test_indices]
        train_dist = Counter(train_targets)
        test_dist = Counter(test_targets)
        print(f"  Train class distribution: {dict(sorted(train_dist.items()))}")
        print(f"  Test class distribution: {dict(sorted(test_dist.items()))}")
        
        # 检查是否有类别在训练集或测试集中完全缺失
        train_classes = set(train_dist.keys())
        test_classes = set(test_dist.keys())
        all_classes = set(range(num_classes))
        missing_in_train = all_classes - train_classes
        missing_in_test = all_classes - test_classes
        
        if missing_in_train:
            print(f"  ⚠️ Warning: Classes missing in training set: {missing_in_train}")
        if missing_in_test:
            print(f"  ⚠️ Warning: Classes missing in test set: {missing_in_test}")

        return (
            (categorical_tensor[train_indices], numerical_tensor[train_indices], targets_tensor[train_indices]),
            (categorical_tensor[test_indices], numerical_tensor[test_indices], targets_tensor[test_indices])
        )

    def create_data_loaders(self, train_data: Tuple, test_data: Tuple, batch_size: int = 256):
        train_dataset = TabularDataset(*train_data)
        test_dataset = TabularDataset(*test_data)
        
        # num_workers=0 ensures compatibility on most systems; increase if on Linux
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        return train_loader, test_loader

    def build_model(self, embedding_dim=64, num_heads=8, num_layers=8, d_ff=256, dropout=0.1, hidden_dims=[256, 128, 64]):
        cardinalities = [self.cardinalities[f] for f in self.categorical_features]
        numerical_dim = len(self.numerical_features)
        num_classes = len(self.target_encoder.classes_)
        
        self.model = TabTransformer(
            categorical_cardinalities=cardinalities,
            numerical_features_dim=numerical_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_transformer_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            hidden_dims=hidden_dims,  # 更深的 MLP
            output_dim=num_classes
        )
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized.")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output dimension: {num_classes}")
        
        # 测试前向传播
        print("\nTesting forward pass...")
        self.model.eval()
        with torch.no_grad():
            # ========== FIX: 创建虚拟输入时，必须尊重每个特征的实际基数 ==========
            # 原代码问题: torch.randint(0, 10, ...) 会生成 0-9 的随机数
            # 如果某个特征的基数小于 10（比如 if_first_carrier 只有 2 个值），
            # 则会导致索引越界错误 (srcIndex < srcSelectDimSize)
            #
            # 修复: 为每个特征单独生成在其有效范围内的随机索引
            dummy_cat = torch.stack([
                torch.randint(0, max(1, card), (2,), dtype=torch.long)
                for card in cardinalities
            ], dim=1).to(self.device)
            # ========== END FIX ==========
            
            dummy_num = torch.randn(2, numerical_dim).to(self.device)
            dummy_output = self.model(dummy_cat, dummy_num)
            print(f"  Input shape - Categorical: {dummy_cat.shape}, Numerical: {dummy_num.shape}")
            print(f"  Output shape: {dummy_output.shape}")
            print(f"  Output range: [{dummy_output.min():.4f}, {dummy_output.max():.4f}]")
            print(f"  Output mean: {dummy_output.mean():.4f}, std: {dummy_output.std():.4f}")
        self.model.train()

    def train_model(self, train_loader, val_loader, epochs=50, learning_rate=0.001, patience=15, 
                    use_focal_loss=True, focal_gamma=2.0, label_smoothing=0.1, use_class_weights=True,
                    use_warmup=True, warmup_epochs=5):
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Optimization settings:")
        print(f"  - Focal Loss: {use_focal_loss} (gamma={focal_gamma})")
        print(f"  - Label Smoothing: {label_smoothing}")
        print(f"  - Class Weights: {use_class_weights}")
        print(f"  - Learning Rate Warmup: {use_warmup} ({warmup_epochs} epochs)")
        
        # Mixed Precision Setup
        use_amp = (self.device.type == 'cuda')
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # 损失函数选择：Focal Loss + Label Smoothing 或加权交叉熵
        class_weights = self.class_weights.to(self.device) if use_class_weights and self.class_weights is not None else None
        
        if use_focal_loss:
            criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            print(f"  Using Focal Loss (gamma={focal_gamma})")
            if class_weights is not None:
                print(f"    Class weights applied: {class_weights.cpu().numpy()}")
        elif label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=class_weights)
            print(f"  Using Label Smoothing CrossEntropy (smoothing={label_smoothing})")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"  Using standard CrossEntropyLoss")
            if class_weights is not None:
                print(f"    Class weights applied: {class_weights.cpu().numpy()}")
        
        # 优化器：AdamW（最适合 Transformer）
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器：使用组合调度器（Warmup + CosineAnnealingWarmRestarts）
        # 修复：使用 SequentialLR 来正确组合 warmup 和 cosine 调度器
        if use_warmup:
            # Warmup 调度器
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,  # 从 10% 的学习率开始
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            # Cosine 调度器（在 warmup 之后使用）
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # 初始周期长度
                T_mult=2,  # 周期倍增因子
                eta_min=learning_rate * 0.01  # 最小学习率
            )
            # 组合调度器
            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=learning_rate * 0.01
            )

        best_val_acc = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_idx, ((cat_x, num_x), y) in enumerate(train_loader):
                cat_x, num_x, y = cat_x.to(self.device), num_x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(cat_x, num_x)
                        loss = criterion(output, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = self.model(cat_x, num_x)
                    loss = criterion(output, y)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                train_loss += loss.item()
                
                # 打印第一个batch的详细信息（用于调试）
                if epoch == 0 and batch_idx == 0:
                    print(f"\n  First batch debug info:")
                    print(f"    Batch size: {y.shape[0]}")
                    print(f"    Output shape: {output.shape}")
                    print(f"    Output range: [{output.min():.4f}, {output.max():.4f}]")
                    print(f"    Output mean: {output.mean():.4f}, std: {output.std():.4f}")
                    print(f"    Loss: {loss.item():.4f}")
                    print(f"    Gradient norm: {grad_norm:.4f}")
                    preds_batch = torch.argmax(output, dim=1)
                    print(f"    Predictions: {preds_batch.cpu().numpy()[:10]}...")
                    print(f"    Targets: {y.cpu().numpy()[:10]}...")
                    print(f"    Batch accuracy: {(preds_batch == y).float().mean().item():.4f}")

            # 学习率调度（epoch 级别）
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_actuals = []
            with torch.no_grad():
                for (cat_x, num_x), y in val_loader:
                    cat_x, num_x, y = cat_x.to(self.device), num_x.to(self.device), y.to(self.device)
                    output = self.model(cat_x, num_x)
                    loss = criterion(output, y)
                    val_loss += loss.item()
                    val_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                    val_actuals.extend(y.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(val_actuals, val_preds)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # 打印预测分布（前几个epoch）
            if epoch < 3 or epoch % 10 == 0:
                pred_dist = Counter(val_preds)
                print(f"  Prediction distribution (top 5): {dict(list(pred_dist.most_common(5)))}")
                actual_dist = Counter(val_actuals)
                print(f"  Actual distribution (top 5): {dict(list(actual_dist.most_common(5)))}")

            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val Acc: {val_acc:.5f} | LR: {current_lr:.6f}")

            # Early Stopping Check based on val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint("best_model_optimized.pth")
                print(f"  ✓ New best model saved! (Acc: {best_val_acc:.5f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        self.load_checkpoint("best_model_optimized.pth")
        # Save preprocessors after training
        self.save_preprocessors()
        return history

    def evaluate(self, loader):
        self.model.eval()
        preds, actuals = [], []
        
        with torch.no_grad():
            for (cat_x, num_x), y in loader:
                cat_x, num_x = cat_x.to(self.device), num_x.to(self.device)
                output = self.model(cat_x, num_x)
                preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                actuals.extend(y.cpu().numpy())
        
        acc = accuracy_score(actuals, preds)
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(actuals, preds, labels=np.arange(len(self.target_encoder.classes_)), target_names=self.target_encoder.classes_))
        
        return actuals, preds, {'Accuracy': acc}

    def _save_checkpoint(self, filename):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        path = os.path.join(self.model_save_dir, filename)
        torch.save(self.model.state_dict(), path)
    
    def save_preprocessors(self):
        """Save all preprocessors (label_encoders, scaler, target_encoder, cardinalities)"""
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        preprocessor_path = os.path.join(self.model_save_dir, 'preprocessors.pkl')
        preprocessors = {
            'label_encoders': self.label_encoders,
            'numerical_scaler': self.numerical_scaler,
            'target_encoder': self.target_encoder,
            'cardinalities': self.cardinalities,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'target_feature': self.target_feature
        }
        
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"Preprocessors saved to {preprocessor_path}")

    def load_checkpoint(self, filename):
        path = os.path.join(self.model_save_dir, filename)
        # Load to CPU first to avoid mapping issues
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TabTransformer model')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name (e.g., "CAN-HGH_CANHGH_Y")')
    parser.add_argument('--data-dir', type=str, 
                        default='/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/data',
                        help='Directory containing data files (default: data directory)')
    parser.add_argument('--model-dir', type=str, 
                        default='/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/nh_work/model_v2_class/tab_transformer',
                        help='Directory to save model (default: model_v2_class/tab_transformer)')
    parser.add_argument('--data-suffix', type=str, default='.csv',
                        help='Suffix for data file (default: "_ClassV1.csv")')
    
    args = parser.parse_args()
    
    # Build paths based on arguments
    DATA_PATH = os.path.join(args.data_dir, f"data_slice_{args.dataset}{args.data_suffix}")
    MODEL_SAVE_DIR = os.path.join(args.model_dir, args.dataset)
    
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {DATA_PATH}")
    print(f"Model save directory: {MODEL_SAVE_DIR}")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    trainer = TabTransformerTrainer(model_save_dir=MODEL_SAVE_DIR)
    
    # 1. Prepare Data
    train_data, test_data = trainer.prepare_data(DATA_PATH)
    
    # 2. Create Loaders (Batch Size 512 for better updates)
    train_loader, test_loader = trainer.create_data_loaders(train_data, test_data, batch_size=64)
    
    # 3. Build Model (Optimized Config: 8 Layers, 8 Heads, embedding_dim=64)
    trainer.build_model(
        embedding_dim=EMBEDDING_DIM,  # 64
        num_heads=NUM_HEADS,           # 8 heads
        num_layers=NUM_TRANSFORMER_LAYERS,  # 8 layers
        d_ff=256,                      # 更大的前馈网络
        dropout=0.1,
        hidden_dims=[256, 128, 64]     # 更深的 MLP
    )
    
    # 4. Train with advanced optimizations
    history = trainer.train_model(
        train_loader, 
        test_loader, 
        epochs=300, 
        learning_rate=0.001,
        patience=20,  # 增加 patience
        use_focal_loss=True,      # 使用 Focal Loss
        focal_gamma=2.0,          # Focal Loss gamma 参数
        label_smoothing=0.1,      # Label Smoothing
        use_class_weights=True,   # 使用类别权重
        use_warmup=True,          # 学习率预热
        warmup_epochs=5           # 预热 5 个 epoch
    )
    
    # 5. Evaluate & Plot
    y_true, y_pred, metrics = trainer.evaluate(test_loader)
    
    # Plotting - Enhanced visualization
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    plt.plot(history['val_loss'], label='Val Loss', alpha=0.8)
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curve
    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green', alpha=0.8)
    plt.title(f'Validation Accuracy (Best: {max(history["val_acc"]):.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate curve
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate', color='orange', alpha=0.8)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'optimization_result.png'), dpi=300, bbox_inches='tight')
    print(f"Results saved to {os.path.join(MODEL_SAVE_DIR, 'optimization_result.png')}")

if __name__ == "__main__":
    main()
