#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized TabTransformer Training Script v2.0 (Classification)
==============================================================
分类任务优化版本 - 针对 pre_class 分类标签

优化点:
1. Pre-LayerNorm Transformer (更稳定的训练)
2. GELU 激活函数 (更平滑的梯度)
3. Column Embedding (区分不同特征列)
4. Numerical Feature Tokenization (FT-Transformer 风格)
5. CLS Token 聚合 (更高效的特征聚合)
6. Lookahead 优化器 (更好的泛化)
7. CrossEntropyLoss (分类任务损失函数)
8. MixUp 数据增强 (防止过拟合，支持分类任务)
9. PyTorch 2.0+ compile 支持
10. Cosine Annealing with Warm Restarts
11. Gradient Checkpointing (可选，节省显存)
12. Stochastic Weight Averaging (SWA)
13. Label Smoothing (提高泛化能力)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import os
import warnings
import math
import pickle
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

# =============================================================================
# Global Configuration Variables
# =============================================================================
# Transformer 架构配置
EMBEDDING_DIM = 32  # Embedding 维度
NUM_HEADS = 8  # 注意力头数
NUM_TRANSFORMER_LAYERS = 6  # Transformer 层数

# 训练配置
BATCH_SIZE = 1024  # 批次大小

# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ModelConfig:
    """模型配置"""
    embedding_dim: int = EMBEDDING_DIM
    num_heads: int = NUM_HEADS
    num_transformer_layers: int = NUM_TRANSFORMER_LAYERS
    d_ff: int = 128
    dropout: float = 0.1
    hidden_dims: List[int] = None
    output_dim: int = None  # 分类任务：类别数（从数据中获取）
    use_numerical_tokenization: bool = True  # FT-Transformer 风格
    use_cls_token: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = BATCH_SIZE
    epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_lookahead: bool = True
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    use_swa: bool = True
    swa_start_epoch: int = 50
    swa_lr: float = 5e-4
    use_weighted_sampling: bool = True
    sampling_alpha: float = 0.5
    label_smoothing: float = 0.0  # For classification tasks
    gradient_clip: float = 1.0
    use_compile: bool = True  # PyTorch 2.0+
    use_warmup: bool = True  # 使用学习率预热
    warmup_epochs: int = 10  # 预热轮数


class LossType(Enum):
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    LABEL_SMOOTHING = "label_smoothing"


# =============================================================================
# Dataset
# =============================================================================
class TabularDataset(Dataset):
    """Custom dataset for tabular data with mixed feature types"""
    def __init__(self, 
                 categorical_features: torch.Tensor, 
                 numerical_features: torch.Tensor, 
                 targets: torch.Tensor):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.categorical_features[idx], 
            self.numerical_features[idx]
        ), self.targets[idx]


# =============================================================================
# Model Components
# =============================================================================
class FeatureEmbedding(nn.Module):
    """Embedding layer for categorical features with column embedding"""
    def __init__(self, cardinalities: List[int], embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in cardinalities
        ])
        # Column embedding to differentiate feature columns
        self.column_embedding = nn.Parameter(
            torch.randn(len(cardinalities), embedding_dim) * 0.02
        )
        self.embedding_dim = embedding_dim

    def forward(self, categorical_inputs: torch.Tensor) -> torch.Tensor:
        embedded = [
            embedding(categorical_inputs[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        stacked = torch.stack(embedded, dim=1)
        # Add column position information
        return stacked + self.column_embedding.unsqueeze(0)


class NumericalFeatureTokenizer(nn.Module):
    """
    Tokenize numerical features (FT-Transformer style)
    Each numerical feature becomes a token with learned embedding
    """
    def __init__(self, num_features: int, embedding_dim: int):
        super().__init__()
        # Linear projection for each numerical feature
        self.weight = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, embedding_dim))
        # Column embedding for numerical features
        self.column_embedding = nn.Parameter(
            torch.randn(num_features, embedding_dim) * 0.02
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_features]
        # Output: [batch, num_features, embedding_dim]
        x = x.unsqueeze(-1)  # [batch, num_features, 1]
        tokens = x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return tokens + self.column_embedding.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional flash attention"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Combined QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_k]
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attention_output)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer encoder block
    More stable training than Post-LN
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward with GELU activation
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN: LayerNorm before attention/FFN
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class ResidualMLPBlock(nn.Module):
    """MLP block with optional residual connection"""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if dimensions don't match
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual * 0.1  # Scaled residual


class OptimizedTabTransformer(nn.Module):
    """
    Optimized TabTransformer model
    
    Features:
    - Pre-LayerNorm Transformer blocks
    - Column embeddings for categorical features
    - Optional numerical feature tokenization (FT-Transformer style)
    - Optional CLS token for aggregation
    - Residual MLP blocks
    """
    def __init__(self,
                 categorical_cardinalities: List[int],
                 numerical_features_dim: int,
                 config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.num_categorical = len(categorical_cardinalities)
        self.num_numerical = numerical_features_dim
        
        # Categorical feature embedding
        self.categorical_embedding = FeatureEmbedding(
            categorical_cardinalities, config.embedding_dim
        )
        
        # Numerical feature processing
        if config.use_numerical_tokenization:
            self.numerical_tokenizer = NumericalFeatureTokenizer(
                numerical_features_dim, config.embedding_dim
            )
        else:
            self.numerical_norm = nn.LayerNorm(numerical_features_dim)
        
        # CLS token for aggregation
        if config.use_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, config.embedding_dim) * 0.02
            )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                config.embedding_dim, 
                config.num_heads, 
                config.d_ff, 
                config.dropout
            )
            for _ in range(config.num_transformer_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        
        # Calculate MLP input dimension
        if config.use_cls_token:
            mlp_input_dim = config.embedding_dim
        elif config.use_numerical_tokenization:
            mlp_input_dim = (self.num_categorical + self.num_numerical) * config.embedding_dim
        else:
            mlp_input_dim = self.num_categorical * config.embedding_dim + numerical_features_dim
        
        # MLP head with residual connections
        self.mlp = self._build_mlp(
            mlp_input_dim, 
            config.hidden_dims, 
            config.dropout, 
            config.output_dim
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _build_mlp(self, 
                   input_dim: int, 
                   hidden_dims: List[int], 
                   dropout: float, 
                   output_dim: int) -> nn.Module:
        """Build MLP with residual blocks"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(ResidualMLPBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module):
        """Initialize weights for better convergence"""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, 
                categorical_inputs: torch.Tensor, 
                numerical_inputs: torch.Tensor) -> torch.Tensor:
        batch_size = categorical_inputs.size(0)
        
        # Categorical token embeddings
        cat_tokens = self.categorical_embedding(categorical_inputs)
        
        # Numerical feature processing
        if self.config.use_numerical_tokenization:
            num_tokens = self.numerical_tokenizer(numerical_inputs)
            tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        else:
            tokens = cat_tokens
            numerical_normalized = self.numerical_norm(numerical_inputs)
        
        # Add CLS token
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Transformer layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        
        tokens = self.final_norm(tokens)
        
        # Feature aggregation
        if self.config.use_cls_token:
            # Use CLS token output
            output = tokens[:, 0]
        else:
            # Flatten all tokens
            if self.config.use_numerical_tokenization:
                output = tokens.view(batch_size, -1)
            else:
                cat_flat = tokens.view(batch_size, -1)
                output = torch.cat([cat_flat, numerical_normalized], dim=1)
        
        # MLP head
        output = self.mlp(output)
        
        # 分类任务：不squeeze，保持 [batch, num_classes] 形状
        # 返回logits（不应用softmax，在损失函数中处理）
        return output


# =============================================================================
# Optimizers
# =============================================================================
class Lookahead(optim.Optimizer):
    """
    Lookahead optimizer wrapper
    Paper: https://arxiv.org/abs/1907.08610
    """
    def __init__(self, optimizer: optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = {}
        self.fast_state = self.optimizer.state
        
        for group in self.param_groups:
            group["counter"] = 0
        
        self._init_slow_weights()
    
    def _init_slow_weights(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p] = {'slow_param': p.data.clone()}
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        
        for group in self.param_groups:
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
                for p in group['params']:
                    if p.requires_grad and p in self.state:
                        slow = self.state[p]['slow_param']
                        slow.add_(self.alpha * (p.data - slow))
                        p.data.copy_(slow)
        
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value


class WarmupCosineAnnealingScheduler:
    """
    Learning Rate Scheduler with Warmup + Cosine Annealing with Warm Restarts
    
    During warmup: linear increase from 0 to learning_rate
    After warmup: Cosine Annealing with Warm Restarts
    """
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 warmup_epochs: int,
                 T_0: int = 20,
                 T_mult: int = 2,
                 eta_min: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        
        # Initialize learning rate to 0 for warmup
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.0
        
        # Cosine Annealing scheduler will be created after warmup
        self.cosine_scheduler = None
    
    def step(self):
        """Update learning rate"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase: linear increase
            warmup_factor = self.current_epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * warmup_factor
        else:
            # After warmup: initialize cosine scheduler if not already done
            if self.cosine_scheduler is None:
                # Ensure learning rate is at base_lr before starting cosine annealing
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.base_lr
                # Create cosine scheduler
                self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.T_0,
                    T_mult=self.T_mult,
                    eta_min=self.eta_min
                )
            # Step the cosine scheduler
            self.cosine_scheduler.step()
    
    def get_last_lr(self):
        """Get current learning rate"""
        if self.current_epoch <= self.warmup_epochs or self.cosine_scheduler is None:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return self.cosine_scheduler.get_last_lr()
    
    def state_dict(self):
        """Return the state of the scheduler as a dict."""
        state = {
            'warmup_epochs': self.warmup_epochs,
            'base_lr': self.base_lr,
            'current_epoch': self.current_epoch,
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'eta_min': self.eta_min,
        }
        if self.cosine_scheduler is not None:
            state['cosine_scheduler_state_dict'] = self.cosine_scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Load the scheduler state."""
        self.warmup_epochs = state_dict['warmup_epochs']
        self.base_lr = state_dict['base_lr']
        self.current_epoch = state_dict['current_epoch']
        self.T_0 = state_dict['T_0']
        self.T_mult = state_dict['T_mult']
        self.eta_min = state_dict['eta_min']
        
        # Restore cosine scheduler if it was active
        if 'cosine_scheduler_state_dict' in state_dict:
            if self.cosine_scheduler is None:
                # Recreate the cosine scheduler
                self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.T_0,
                    T_mult=self.T_mult,
                    eta_min=self.eta_min
                )
            self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler_state_dict'])


# =============================================================================
# Training Utilities
# =============================================================================
def mixup_data(cat_x: torch.Tensor, 
               num_x: torch.Tensor, 
               y: torch.Tensor, 
               alpha: float = 0.2,
               num_classes: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp data augmentation for tabular data
    For classification: convert labels to one-hot and interpolate
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = cat_x.size(0)
    index = torch.randperm(batch_size, device=cat_x.device)
    
    # Numerical features: linear interpolation
    mixed_num_x = lam * num_x + (1 - lam) * num_x[index]
    
    # Categorical features: randomly select from one of the two samples
    # Use a per-feature random selection
    cat_mask = (torch.rand(batch_size, cat_x.size(1), device=cat_x.device) < lam).long()
    mixed_cat_x = cat_mask * cat_x + (1 - cat_mask) * cat_x[index]
    
    # Target: for classification, convert to one-hot and interpolate
    if num_classes is not None:
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        y_onehot_mixed = F.one_hot(y[index], num_classes=num_classes).float()
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot_mixed
    else:
        # Regression: linear interpolation
        mixed_y = lam * y.float() + (1 - lam) * y[index].float()
    
    return mixed_cat_x, mixed_num_x, mixed_y, lam


def get_loss_function(loss_type: LossType, num_classes: int = None, **kwargs) -> nn.Module:
    """Get loss function by type for classification"""
    if loss_type == LossType.CROSS_ENTROPY:
        if kwargs.get('label_smoothing', 0.0) > 0:
            return nn.CrossEntropyLoss(label_smoothing=kwargs.get('label_smoothing', 0.0))
        return nn.CrossEntropyLoss()
    elif loss_type == LossType.FOCAL:
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, pred, target):
                ce_loss = F.cross_entropy(pred, target, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(
            alpha=kwargs.get('focal_alpha', 1.0),
            gamma=kwargs.get('focal_gamma', 2.0)
        )
    elif loss_type == LossType.LABEL_SMOOTHING:
        smoothing = kwargs.get('label_smoothing', 0.1)
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# =============================================================================
# Trainer
# =============================================================================
class OptimizedTabTransformerTrainer:
    """
    Trainer class for Optimized TabTransformer model
    """
    def __init__(self, 
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 random_state: int = 42,
                 weights_only: bool = False):
        
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.random_state = random_state
        self.weights_only = weights_only
        
        self.model = None
        self.swa_model = None
        self.device = self._setup_device()
        self._set_random_seeds()
        
        # Feature definitions
        self.categorical_features = [
            'seasonality', 'tod', 'day_prior_seg', 'dow', 
            'if_first_carrier', 'if_last_carrier', 'data_slice'
        ]
        self.numerical_features = [
            'flt_duration', 'public_fare', 'price_1', 'cap_share_thisflt', 
            'cap_share_czflts', 'lf', 'rask', 'yield', 'incremental_lf', 
            'incremental_lf_cz', 'incremental_rask', 'incremental_rask_cz', 
            'incremental_yield', 'lf_cz', 'rask_cz', 'lf_yoy', 'rask_yoy'
        ]
        self.target_feature = 'pre_class'
        
        # Preprocessors
        self.label_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.cardinalities = {}
        self.route_mapping = {}
        self.num_classes = None  # 分类任务：类别数
        
        # Training state
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        # Training components (for checkpoint saving)
        self.optimizer = None
        self.base_optimizer = None
        self.scheduler = None
        self.swa_scheduler = None
        self.scaler = None
        
        # Save directories
        base_save_dir = '/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/nh_work/model_alldata_v2_class/tab_transformer'
        self.model_save_dir = os.path.join(base_save_dir, 'model_checkpoints_v2')
        self.base_model_dir = base_save_dir

    def _setup_device(self) -> torch.device:
        """Setup compute device with optimizations"""
        if torch.cuda.is_available():
            try:
                device = torch.device('cuda')
                print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                return device
            except Exception as e:
                print(f"⚠️ GPU init failed: {e}")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Apple Silicon GPU detected")
            return torch.device('mps')
        
        print("💻 Using CPU")
        return torch.device('cpu')

    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def prepare_data(self, 
                     data_path: str, 
                     test_size: float = 0.2) -> Tuple[Tuple, Tuple]:
        """
        Prepare and preprocess data
        """
        print("📊 Preparing data...")
        df = pd.read_csv(data_path)
        print(f"   Raw dataset size: {df.shape}")
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        # Clean data
        df_clean = df.dropna().reset_index(drop=True)
        print(f"   Clean dataset size: {df_clean.shape}")
        
        # Process categorical features
        categorical_data = self._process_categorical(df_clean)
        
        # Process numerical features
        numerical_data = self._process_numerical(df_clean)
        
        # Process target
        targets = self._process_target(df_clean)
        
        # Convert to tensors
        categorical_tensor = torch.stack([
            torch.tensor(categorical_data[f], dtype=torch.long)
            for f in self.categorical_features
        ], dim=1)
        
        numerical_tensor = torch.tensor(numerical_data, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.long)  # 分类任务：使用long类型
        
        # Split data
        indices = np.arange(len(categorical_tensor))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=self.random_state
        )
        
        print(f"   Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        
        return (
            (categorical_tensor[train_idx], numerical_tensor[train_idx], targets_tensor[train_idx]),
            (categorical_tensor[test_idx], numerical_tensor[test_idx], targets_tensor[test_idx])
        )

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering"""
        df = df.copy()
        
        # Price gap feature
        if 'price_1' in df.columns and 'price' in df.columns:
            df['price_gap'] = df['price'] - df['price_1']
        
        # Boolean conversion
        bool_mapping = {'true': 1, 'false': 0, True: 1, False: 0, 'True': 1, 'False': 0}
        for col in ['if_first_carrier', 'if_last_carrier']:
            if col in df.columns:
                df[col] = df[col].map(bool_mapping).fillna(0).astype(int)
        
        # Route mapping
        if 'data_slice' in df.columns:
            unique_routes = sorted(df['data_slice'].dropna().unique())
            self.route_mapping = {route: idx for idx, route in enumerate(unique_routes)}
            df['data_slice'] = df['data_slice'].map(self.route_mapping).fillna(-1).astype(int)
            
            # Save mapping
            os.makedirs(self.base_model_dir, exist_ok=True)
            mapping_path = os.path.join(self.base_model_dir, 'route_mapping.pkl')
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.route_mapping, f)
            print(f"   Saved route mapping: {len(self.route_mapping)} routes")
        
        return df

    def _process_categorical(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process categorical features"""
        categorical_data = {}
        
        for feature in self.categorical_features:
            if feature == 'data_slice':
                # Handle data_slice specially (already mapped)
                values = df[feature].values.copy()
                values[values == -1] = len(self.route_mapping)
                categorical_data[feature] = values
                self.cardinalities[feature] = len(self.route_mapping) + 1
            else:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                categorical_data[feature] = self.label_encoders[feature].fit_transform(df[feature])
                self.cardinalities[feature] = df[feature].nunique()
            
            print(f"   {feature}: {self.cardinalities[feature]} categories")
        
        return categorical_data

    def _process_numerical(self, df: pd.DataFrame) -> np.ndarray:
        """Process numerical features"""
        numerical_data = df[self.numerical_features].values
        return self.numerical_scaler.fit_transform(numerical_data)

    def _process_target(self, df: pd.DataFrame) -> np.ndarray:
        """Process target variable for classification"""
        targets = df[self.target_feature].values
        
        # Encode categorical labels to integers
        if targets.dtype == 'object' or isinstance(targets[0], str):
            targets = self.target_encoder.fit_transform(targets)
        else:
            # If already numeric, ensure they are integers
            targets = self.target_encoder.fit_transform(targets.astype(str))
        
        self.num_classes = len(self.target_encoder.classes_)
        print(f"   Encoded target: {self.num_classes} classes")
        print(f"   Class distribution:")
        unique, counts = np.unique(targets, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = count / len(targets) * 100
            print(f"      Class {cls}: {count} samples ({percentage:.1f}%)")
        
        # Return as integer labels (not float)
        return targets.astype(np.int64)

    def create_data_loaders(self, 
                            train_data: Tuple, 
                            test_data: Tuple) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders with optional weighted sampling"""
        train_dataset = TabularDataset(*train_data)
        test_dataset = TabularDataset(*test_data)
        
        cfg = self.training_config
        
        if cfg.use_weighted_sampling:
            targets = train_data[2].numpy()
            unique_targets, counts = np.unique(targets, return_counts=True)
            max_count = counts.max()
            
            # Exponential smoothing weights
            target_to_weight = {
                target: np.exp(-cfg.sampling_alpha * (max_count / count - 1))
                for target, count in zip(unique_targets, counts)
            }
            sample_weights = np.array([target_to_weight[t] for t in targets])
            sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=cfg.batch_size, 
                sampler=sampler,
                pin_memory=True,
                num_workers=0
            )
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=cfg.batch_size, 
                shuffle=True,
                pin_memory=True,
                num_workers=0
            )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )
        
        return train_loader, test_loader

    def build_model(self):
        """Build and initialize model"""
        cardinalities = [self.cardinalities[f] for f in self.categorical_features]
        numerical_dim = len(self.numerical_features)
        
        # 设置输出维度为类别数
        if self.model_config.output_dim is None:
            if self.num_classes is None:
                raise ValueError("num_classes must be set before building model. Call prepare_data first.")
            self.model_config.output_dim = self.num_classes
        
        self.model = OptimizedTabTransformer(
            categorical_cardinalities=cardinalities,
            numerical_features_dim=numerical_dim,
            config=self.model_config
        )
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"   Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        # PyTorch 2.0+ compile
        if self.training_config.use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("   ✅ Model compiled with torch.compile()")
            except Exception as e:
                print(f"   ⚠️ torch.compile failed: {e}")
        
        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        return self.model

    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              loss_type: LossType = LossType.CROSS_ENTROPY) -> Dict:
        """
        Train the model with all optimizations
        """
        cfg = self.training_config
        print(f"\n🚀 Starting training for {cfg.epochs} epochs...")
        print(f"   Batch size: {cfg.batch_size}")
        print(f"   Learning rate: {cfg.learning_rate}")
        print(f"   MixUp: {cfg.use_mixup} (alpha={cfg.mixup_alpha})")
        print(f"   Lookahead: {cfg.use_lookahead}")
        print(f"   SWA: {cfg.use_swa}")
        
        # Mixed precision setup
        use_amp = (self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Loss function
        criterion = get_loss_function(
            loss_type, 
            num_classes=self.num_classes,
            label_smoothing=cfg.label_smoothing
        )
        print(f"   Loss function: {loss_type.value}")
        if cfg.label_smoothing > 0:
            print(f"   Label smoothing: {cfg.label_smoothing}")
        
        # Optimizer
        self.base_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999)
        )
        
        if cfg.use_lookahead:
            self.optimizer = Lookahead(
                self.base_optimizer, 
                k=cfg.lookahead_k, 
                alpha=cfg.lookahead_alpha
            )
        else:
            self.optimizer = self.base_optimizer
        
        # Scheduler: Cosine Annealing with Warm Restarts (with optional Warmup)
        if cfg.use_warmup:
            self.scheduler = WarmupCosineAnnealingScheduler(
                self.base_optimizer if cfg.use_lookahead else self.optimizer,
                warmup_epochs=cfg.warmup_epochs,
                T_0=20,  # Restart every 20 epochs
                T_mult=2,  # Double period after each restart
                eta_min=1e-6
            )
            print(f"   Learning Rate Warmup: {cfg.warmup_epochs} epochs")
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.base_optimizer if cfg.use_lookahead else self.optimizer,
                T_0=20,  # Restart every 20 epochs
                T_mult=2,  # Double period after each restart
                eta_min=1e-6
            )
        
        # SWA setup
        if cfg.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(
                self.base_optimizer if cfg.use_lookahead else self.optimizer,
                swa_lr=cfg.swa_lr
            )
        else:
            self.swa_scheduler = None
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(cfg.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for (cat_x, num_x), y in train_loader:
                cat_x = cat_x.to(self.device)
                num_x = num_x.to(self.device)
                y = y.to(self.device)
                
                # MixUp augmentation
                use_mixup = cfg.use_mixup and np.random.random() < 0.5
                if use_mixup:
                    cat_x, num_x, y_mixed, lam = mixup_data(
                        cat_x, num_x, y, cfg.mixup_alpha, num_classes=self.num_classes
                    )
                
                self.optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(cat_x, num_x)
                        if use_mixup:
                            # For MixUp with classification, use soft labels
                            loss = -torch.sum(y_mixed * F.log_softmax(output, dim=1), dim=1).mean()
                        else:
                            loss = criterion(output, y)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.base_optimizer if cfg.use_lookahead else self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(cat_x, num_x)
                    if use_mixup:
                        # For MixUp with classification, use soft labels
                        loss = -torch.sum(y_mixed * F.log_softmax(output, dim=1), dim=1).mean()
                    else:
                        loss = criterion(output, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                    self.optimizer.step()
                
                train_loss += loss.item()
            
            # Scheduler step
            if cfg.use_swa and epoch >= cfg.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                if self.swa_scheduler is not None:
                    self.swa_scheduler.step()
            else:
                self.scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss = self._validate(val_loader, criterion)
            self.history['val_loss'].append(val_loss)
            
            current_lr = self.scheduler.get_last_lr()[0]
            self.history['lr'].append(current_lr)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d} | Train: {avg_train_loss:.5f} | "
                      f"Val: {val_loss:.5f} | LR: {current_lr:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("best_model_v2.pth", epoch=epoch+1)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        # Finalize SWA
        if cfg.use_swa:
            print("   📊 Updating SWA batch normalization...")
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=self.device)
            self._save_checkpoint("swa_model_v2.pth", model=self.swa_model, epoch=cfg.epochs)
        
        # Load best model
        self.load_checkpoint("best_model_v2.pth")
        print(f"   ✅ Training complete. Best val loss: {best_val_loss:.5f}")
        
        return self.history

    def _validate(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for (cat_x, num_x), y in loader:
                cat_x = cat_x.to(self.device)
                num_x = num_x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(cat_x, num_x)
                loss = criterion(output, y)
                val_loss += loss.item()
        
        return val_loss / len(loader)

    def evaluate(self, loader: DataLoader, use_swa: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Evaluate model and return metrics for classification"""
        
        model = self.swa_model if (use_swa and self.swa_model is not None) else self.model
        model.eval()
        
        all_preds, all_probs, actuals = [], [], []
        
        with torch.no_grad():
            for (cat_x, num_x), y in loader:
                cat_x = cat_x.to(self.device)
                num_x = num_x.to(self.device)
                
                output = model(cat_x, num_x)
                probs = F.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                actuals.extend(y.numpy())
        
        actuals = np.array(actuals)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Classification metrics
        accuracy = accuracy_score(actuals, all_preds)
        precision = precision_score(actuals, all_preds, average='weighted', zero_division=0)
        recall = recall_score(actuals, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(actuals, all_preds, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_macro = precision_score(actuals, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(actuals, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(actuals, all_preds, average='macro', zero_division=0)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision (weighted)': precision,
            'Recall (weighted)': recall,
            'F1 (weighted)': f1,
            'Precision (macro)': precision_macro,
            'Recall (macro)': recall_macro,
            'F1 (macro)': f1_macro,
            'Confusion Matrix': confusion_matrix(actuals, all_preds)
        }
        
        print(f"\n📈 Classification Evaluation Results:")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   Precision (weighted): {metrics['Precision (weighted)']:.4f}")
        print(f"   Recall (weighted): {metrics['Recall (weighted)']:.4f}")
        print(f"   F1 Score (weighted): {metrics['F1 (weighted)']:.4f}")
        print(f"   Precision (macro): {metrics['Precision (macro)']:.4f}")
        print(f"   Recall (macro): {metrics['Recall (macro)']:.4f}")
        print(f"   F1 Score (macro): {metrics['F1 (macro)']:.4f}")
        
        # Print classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(actuals, all_preds, target_names=[f'Class {i}' for i in range(self.num_classes)], zero_division=0))
        
        return actuals, all_preds, metrics

    def _save_checkpoint(self, filename: str, model: nn.Module = None, epoch: int = None):
        """Save model checkpoint
        
        Args:
            filename: Checkpoint filename
            model: Model to save (default: self.model)
            epoch: Current epoch number (for full checkpoint)
        """
        os.makedirs(self.model_save_dir, exist_ok=True)
        path = os.path.join(self.model_save_dir, filename)
        
        model = model or self.model
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            'model_state_dict': state_dict,
            'model_config': self.model_config,
            'cardinalities': self.cardinalities,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'num_classes': self.num_classes,
        }
        
        # Save full checkpoint if weights_only is False
        if not self.weights_only:
            checkpoint['training_config'] = self.training_config
            checkpoint['history'] = self.history
            
            if epoch is not None:
                checkpoint['epoch'] = epoch
            
            # Save optimizer state
            if self.optimizer is not None:
                if isinstance(self.optimizer, Lookahead):
                    checkpoint['optimizer_state_dict'] = self.optimizer.optimizer.state_dict()
                    checkpoint['lookahead_state'] = {
                        'k': self.optimizer.k,
                        'alpha': self.optimizer.alpha,
                        'state': self.optimizer.state
                    }
                else:
                    checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            
            if self.base_optimizer is not None and self.base_optimizer != self.optimizer:
                checkpoint['base_optimizer_state_dict'] = self.base_optimizer.state_dict()
            
            # Save scheduler state
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            if self.swa_scheduler is not None:
                checkpoint['swa_scheduler_state_dict'] = self.swa_scheduler.state_dict()
            
            # Save scaler state (for mixed precision)
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            # Save SWA model if exists
            if self.swa_model is not None:
                swa_state_dict = self.swa_model.module.state_dict() if hasattr(self.swa_model, 'module') else self.swa_model.state_dict()
                checkpoint['swa_model_state_dict'] = swa_state_dict
        
        torch.save(checkpoint, path)
        checkpoint_type = "weights only" if self.weights_only else "full checkpoint"
        print(f"   💾 Saved {checkpoint_type}: {path}")

    def load_checkpoint(self, filename: str, load_training_state: bool = True):
        """Load model checkpoint
        
        Args:
            filename: Checkpoint filename
            load_training_state: Whether to load optimizer, scheduler, etc. (default: True)
        """
        path = os.path.join(self.model_save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state if available and requested
        if load_training_state and not self.weights_only:
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                if isinstance(self.optimizer, Lookahead):
                    self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'lookahead_state' in checkpoint:
                        lookahead_state = checkpoint['lookahead_state']
                        self.optimizer.k = lookahead_state.get('k', self.optimizer.k)
                        self.optimizer.alpha = lookahead_state.get('alpha', self.optimizer.alpha)
                        self.optimizer.state = lookahead_state.get('state', self.optimizer.state)
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'base_optimizer_state_dict' in checkpoint and self.base_optimizer is not None:
                self.base_optimizer.load_state_dict(checkpoint['base_optimizer_state_dict'])
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'swa_scheduler_state_dict' in checkpoint and self.swa_scheduler is not None:
                self.swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state_dict'])
            
            # Load scaler state
            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load training history
            if 'history' in checkpoint:
                self.history = checkpoint['history']
            
            # Load SWA model if exists
            if 'swa_model_state_dict' in checkpoint and self.swa_model is not None:
                if hasattr(self.swa_model, 'module'):
                    self.swa_model.module.load_state_dict(checkpoint['swa_model_state_dict'])
                else:
                    self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
            
            print(f"   ✅ Loaded full checkpoint from {path}")
            if 'epoch' in checkpoint:
                print(f"      Epoch: {checkpoint['epoch']}")
        else:
            print(f"   ✅ Loaded model weights from {path}")

    def save_preprocessors(self, path: str):
        """Save all preprocessors for inference"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        preprocessors = {
            'label_encoders': self.label_encoders,
            'numerical_scaler': self.numerical_scaler,
            'target_encoder': self.target_encoder,
            'cardinalities': self.cardinalities,
            'route_mapping': self.route_mapping,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'num_classes': self.num_classes,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"   Saved preprocessors to {path}")

    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if save_path is None:
            save_path = os.path.join(self.base_model_dir, 'training_history_v2.png')
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss curves
        axes[0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0].plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1].plot(self.history['lr'], color='green', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        # Loss ratio
        if len(self.history['train_loss']) > 0 and len(self.history['val_loss']) > 0:
            ratio = np.array(self.history['val_loss']) / (np.array(self.history['train_loss']) + 1e-8)
            axes[2].plot(ratio, color='orange', alpha=0.8)
            axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Val/Train Loss Ratio')
            axes[2].set_title('Generalization Gap')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved training history to {save_path}")
        plt.close()

    def plot_predictions(self, 
                        actuals: np.ndarray, 
                        preds: np.ndarray, 
                        metrics: Dict,
                        save_path: str = None):
        """Plot prediction analysis for classification"""
        if save_path is None:
            save_path = os.path.join(self.base_model_dir, 'predictions_v2.png')
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(actuals, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
                   xticklabels=[f'Class {i}' for i in range(self.num_classes)],
                   yticklabels=[f'Class {i}' for i in range(self.num_classes)])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title('Confusion Matrix')
        
        # Accuracy by class
        class_accuracies = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
        axes[0, 1].bar(range(len(class_accuracies)), class_accuracies, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Per-Class Accuracy')
        axes[0, 1].set_xticks(range(len(class_accuracies)))
        axes[0, 1].set_xticklabels([f'Class {i}' for i in range(len(class_accuracies))])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].axhline(y=metrics['Accuracy'], color='red', linestyle='--', label=f'Overall: {metrics["Accuracy"]:.3f}')
        axes[0, 1].legend()
        
        # Prediction distribution
        unique_actuals, counts_actuals = np.unique(actuals, return_counts=True)
        unique_preds, counts_preds = np.unique(preds, return_counts=True)
        x = np.arange(self.num_classes)
        width = 0.35
        axes[1, 0].bar(x - width/2, [counts_actuals[unique_actuals == i][0] if i in unique_actuals else 0 for i in range(self.num_classes)], 
                      width, label='Actual', alpha=0.7)
        axes[1, 0].bar(x + width/2, [counts_preds[unique_preds == i][0] if i in unique_preds else 0 for i in range(self.num_classes)], 
                      width, label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Class Distribution: Actual vs Predicted')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'Class {i}' for i in range(self.num_classes)])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Metrics summary
        axes[1, 1].axis('off')
        metrics_text = f"""
        Classification Metrics Summary
        
        Accuracy: {metrics['Accuracy']:.4f}
        
        Weighted Average:
          Precision: {metrics['Precision (weighted)']:.4f}
          Recall:    {metrics['Recall (weighted)']:.4f}
          F1 Score:  {metrics['F1 (weighted)']:.4f}
        
        Macro Average:
          Precision: {metrics['Precision (macro)']:.4f}
          Recall:    {metrics['Recall (macro)']:.4f}
          F1 Score:  {metrics['F1 (macro)']:.4f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, 
                       verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved predictions plot to {save_path}")
        plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🔥 Optimized TabTransformer Training Pipeline v2.0 (Classification)")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = '/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/data/all_route_data.csv'
    
    # Model configuration
    model_config = ModelConfig(
        embedding_dim=EMBEDDING_DIM,  # 使用全局变量
        num_heads=NUM_HEADS,  # 使用全局变量
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,  # 使用全局变量
        d_ff=128,
        dropout=0.1,
        hidden_dims=[128, 64],
        output_dim=None,  # 将从数据中自动获取类别数
        use_numerical_tokenization=True,  # FT-Transformer style
        use_cls_token=True
    )
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=BATCH_SIZE,  # 使用全局变量
        epochs=300,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=20,
        use_mixup=True,
        mixup_alpha=0.2,
        use_lookahead=True,
        lookahead_k=5,
        lookahead_alpha=0.5,
        use_swa=True,
        swa_start_epoch=50,
        swa_lr=5e-4,
        use_weighted_sampling=True,
        sampling_alpha=0.5,
        label_smoothing=0.1,  # 分类任务：使用标签平滑
        gradient_clip=1.0,
        use_compile=True,
        use_warmup=True,  # 启用学习率预热
        warmup_epochs=10  # 预热轮数
    )
    
    # Check data file
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        print("   Please update DATA_PATH to your data location.")
        return
    
    # Initialize trainer
    trainer = OptimizedTabTransformerTrainer(
        model_config=model_config,
        training_config=training_config,
        random_state=42,
        weights_only=False  # weights_only=False: Save full checkpoint (optimizer, scheduler, history, etc.)
    )
    
    # 1. Prepare data
    train_data, test_data = trainer.prepare_data(DATA_PATH)
    
    # 2. Create data loaders
    train_loader, test_loader = trainer.create_data_loaders(train_data, test_data)
    
    # 3. Build model
    trainer.build_model()
    
    # 4. Train model
    history = trainer.train(
        train_loader, 
        test_loader,
        loss_type=LossType.CROSS_ENTROPY  # 分类任务：使用交叉熵损失
    )
    
    # 5. Evaluate
    print("\n" + "=" * 60)
    print("📊 Final Evaluation")
    print("=" * 60)
    
    # Evaluate regular model
    print("\n🔹 Regular Model:")
    actuals, preds, metrics = trainer.evaluate(test_loader, use_swa=False)
    
    # Evaluate SWA model if available
    if trainer.swa_model is not None:
        print("\n🔹 SWA Model:")
        actuals_swa, preds_swa, metrics_swa = trainer.evaluate(test_loader, use_swa=True)
    
    # 6. Save results
    trainer.plot_training_history()
    trainer.plot_predictions(actuals, preds, metrics)
    trainer.save_preprocessors(os.path.join(trainer.base_model_dir, 'preprocessors.pkl'))
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
