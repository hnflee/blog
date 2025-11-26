#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized TabTransformer Training Script v2.0
==============================================
优化点:
1. Pre-LayerNorm Transformer (更稳定的训练)
2. GELU 激活函数 (更平滑的梯度)
3. Column Embedding (区分不同特征列)
4. Numerical Feature Tokenization (FT-Transformer 风格)
5. CLS Token 聚合 (更高效的特征聚合)
6. Lookahead 优化器 (更好的泛化)
7. SmoothL1Loss (对异常值更鲁棒)
8. MixUp 数据增强 (防止过拟合)
9. PyTorch 2.0+ compile 支持
10. Cosine Annealing with Warm Restarts
11. Gradient Checkpointing (可选，节省显存)
12. Stochastic Weight Averaging (SWA)
"""

import pandas as pd
import numpy as np
import logging
import os
import warnings
import math
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union

# Suppress warnings before importing torch
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Suppress torch inductor warnings (SM count warnings)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ModelConfig:
    """模型配置"""
    embedding_dim: int = 32
    num_heads: int = 4
    num_transformer_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.1
    hidden_dims: List[int] = None
    output_dim: int = 1
    use_numerical_tokenization: bool = True  # FT-Transformer 风格
    use_cls_token: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class CompileMode(Enum):
    """PyTorch 2.0 compile modes"""
    NONE = "none"                    # 不使用编译
    DEFAULT = "default"              # 默认模式，平衡编译时间和性能
    REDUCE_OVERHEAD = "reduce-overhead"  # 减少开销，适合小 batch
    MAX_AUTOTUNE = "max-autotune"    # 最大自动调优，需要更多 GPU SM


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 512
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
    compile_mode: str = "default"  # "none", "default", "reduce-overhead", "max-autotune"


class LossType(Enum):
    MSE = "mse"
    SMOOTH_L1 = "smooth_l1"
    HUBER = "huber"
    COMBINED = "combined"


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
        output = self.mlp(output).squeeze(-1)
        
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


# =============================================================================
# Training Utilities
# =============================================================================
def mixup_data(cat_x: torch.Tensor, 
               num_x: torch.Tensor, 
               y: torch.Tensor, 
               alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp data augmentation for tabular data
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
    
    # Target: linear interpolation
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_cat_x, mixed_num_x, mixed_y, lam


def get_loss_function(loss_type: LossType, **kwargs) -> nn.Module:
    """Get loss function by type"""
    if loss_type == LossType.MSE:
        return nn.MSELoss()
    elif loss_type == LossType.SMOOTH_L1:
        return nn.SmoothL1Loss(beta=kwargs.get('beta', 1.0))
    elif loss_type == LossType.HUBER:
        return nn.HuberLoss(delta=kwargs.get('delta', 1.0))
    elif loss_type == LossType.COMBINED:
        class CombinedLoss(nn.Module):
            def __init__(self, mse_weight=0.5):
                super().__init__()
                self.mse_weight = mse_weight
            
            def forward(self, pred, target):
                mse = F.mse_loss(pred, target)
                l1 = F.l1_loss(pred, target)
                return self.mse_weight * mse + (1 - self.mse_weight) * l1
        
        return CombinedLoss(kwargs.get('mse_weight', 0.5))
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
                 random_state: int = 42):
        
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.random_state = random_state
        
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
        self.target_scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.cardinalities = {}
        self.route_mapping = {}
        
        # Training state
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        # Save directories
        self.model_save_dir = './model_checkpoints_v2'
        self.base_model_dir = './model_output'

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
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
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
        """Process target variable"""
        targets = df[self.target_feature].values
        
        # Encode if categorical
        if targets.dtype == 'object' or isinstance(targets[0], str):
            targets = self.target_encoder.fit_transform(targets)
            print(f"   Encoded target: {len(self.target_encoder.classes_)} classes")
        
        targets = targets.astype(float)
        return self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()

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
            compile_mode = self.training_config.compile_mode
            if compile_mode != "none":
                try:
                    # Suppress inductor warnings about SM count
                    import logging
                    logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
                    
                    self.model = torch.compile(self.model, mode=compile_mode)
                    print(f"   ✅ Model compiled with torch.compile(mode='{compile_mode}')")
                except Exception as e:
                    print(f"   ⚠️ torch.compile failed: {e}")
                    print(f"   ℹ️ Continuing without compilation")
        
        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        return self.model

    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              loss_type: LossType = LossType.SMOOTH_L1) -> Dict:
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
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Loss function
        criterion = get_loss_function(loss_type)
        print(f"   Loss function: {loss_type.value}")
        
        # Optimizer
        base_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999)
        )
        
        if cfg.use_lookahead:
            optimizer = Lookahead(
                base_optimizer, 
                k=cfg.lookahead_k, 
                alpha=cfg.lookahead_alpha
            )
        else:
            optimizer = base_optimizer
        
        # Scheduler: Cosine Annealing with Warm Restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_optimizer if cfg.use_lookahead else optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-6
        )
        
        # SWA setup
        if cfg.use_swa:
            self.swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(
                base_optimizer if cfg.use_lookahead else optimizer,
                swa_lr=cfg.swa_lr
            )
        
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
                if cfg.use_mixup and np.random.random() < 0.5:
                    cat_x, num_x, y, _ = mixup_data(cat_x, num_x, y, cfg.mixup_alpha)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(cat_x, num_x)
                        loss = criterion(output, y)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(base_optimizer if cfg.use_lookahead else optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = self.model(cat_x, num_x)
                    loss = criterion(output, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                    optimizer.step()
                
                train_loss += loss.item()
            
            # Scheduler step
            if cfg.use_swa and epoch >= cfg.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                swa_scheduler.step()
            else:
                scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss = self._validate(val_loader, criterion)
            self.history['val_loss'].append(val_loss)
            
            current_lr = scheduler.get_last_lr()[0]
            self.history['lr'].append(current_lr)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d} | Train: {avg_train_loss:.5f} | "
                      f"Val: {val_loss:.5f} | LR: {current_lr:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("best_model_v2.pth")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        # Finalize SWA
        if cfg.use_swa:
            print("   📊 Updating SWA batch normalization...")
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=self.device)
            self._save_checkpoint("swa_model_v2.pth", model=self.swa_model)
        
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
        """Evaluate model and return metrics"""
        model = self.swa_model if (use_swa and self.swa_model is not None) else self.model
        model.eval()
        
        preds, actuals = [], []
        
        with torch.no_grad():
            for (cat_x, num_x), y in loader:
                cat_x = cat_x.to(self.device)
                num_x = num_x.to(self.device)
                
                output = model(cat_x, num_x)
                preds.extend(output.cpu().numpy())
                actuals.extend(y.numpy())
        
        # Inverse transform
        preds = self.target_scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()
        actuals = self.target_scaler.inverse_transform(
            np.array(actuals).reshape(-1, 1)
        ).flatten()
        
        # Metrics
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(actuals, preds)),
            'MAE': mean_absolute_error(actuals, preds),
            'R2': r2_score(actuals, preds),
            'MAPE': np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
        }
        
        print(f"\n📈 Evaluation Results:")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   MAE:  {metrics['MAE']:.4f}")
        print(f"   R2:   {metrics['R2']:.4f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        
        return actuals, preds, metrics

    def _save_checkpoint(self, filename: str, model: nn.Module = None):
        """Save model checkpoint"""
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
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = os.path.join(self.model_save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_preprocessors(self, path: str):
        """Save all preprocessors for inference"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        preprocessors = {
            'label_encoders': self.label_encoders,
            'numerical_scaler': self.numerical_scaler,
            'target_scaler': self.target_scaler,
            'target_encoder': self.target_encoder,
            'cardinalities': self.cardinalities,
            'route_mapping': self.route_mapping,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"   Saved preprocessors to {path}")

    def plot_training_history(self, save_path: str = 'training_history_v2.png'):
        """Plot training history"""
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
                        save_path: str = 'predictions_v2.png'):
        """Plot prediction analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plot
        axes[0, 0].scatter(actuals, preds, alpha=0.3, s=5)
        axes[0, 0].plot([actuals.min(), actuals.max()], 
                       [actuals.min(), actuals.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title(f'Predictions (R² = {metrics["R2"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = actuals - preds
        axes[0, 1].scatter(preds, residuals, alpha=0.3, s=5)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='white', alpha=0.7)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residual')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Residual Distribution (MAE = {metrics["MAE"]:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error by actual value bins
        bins = np.percentile(actuals, np.linspace(0, 100, 11))
        bin_indices = np.digitize(actuals, bins)
        bin_errors = []
        bin_centers = []
        
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_errors.append(np.abs(residuals[mask]).mean())
                bin_centers.append((bins[i-1] + bins[i]) / 2)
        
        axes[1, 1].bar(range(len(bin_errors)), bin_errors, alpha=0.7)
        axes[1, 1].set_xlabel('Actual Value Quantile')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Error by Value Range')
        axes[1, 1].grid(True, alpha=0.3)
        
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
    print("🔥 Optimized TabTransformer Training Pipeline v2.0")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = '/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/data/all_route_data.csv'
    
    # Model configuration
    model_config = ModelConfig(
        embedding_dim=32,
        num_heads=4,
        num_transformer_layers=3,
        d_ff=128,
        dropout=0.1,
        hidden_dims=[128, 64],
        output_dim=1,
        use_numerical_tokenization=True,  # FT-Transformer style
        use_cls_token=True
    )
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=512,
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
        gradient_clip=1.0,
        use_compile=True,
        compile_mode="default"  # "none", "default", "reduce-overhead", "max-autotune"
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
        random_state=42
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
        loss_type=LossType.SMOOTH_L1
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
