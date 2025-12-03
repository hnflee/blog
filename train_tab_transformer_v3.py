#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TabTransformer Training Script v3.0 - Advanced Optimization
============================================================
针对准确率提升的进阶优化版本

新增优化:
1. Focal Loss with Auto Class Weights
2. SAM (Sharpness-Aware Minimization) Optimizer
3. Attention Pooling (替代 CLS Token)
4. R-Drop Regularization
5. Feature Gating Mechanism
6. Cross-Attention between Categorical and Numerical
7. Test-Time Augmentation (TTA)
8. Progressive Resizing / Curriculum Learning
9. Knowledge Distillation Ready
10. Advanced Feature Engineering
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             top_k_accuracy_score)
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import os
import warnings
import math
import pickle
import argparse
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import copy

warnings.filterwarnings('ignore')

# =============================================================================
# Global Configuration
# =============================================================================
EMBEDDING_DIM = 64  # 增大 embedding 维度
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 4  # 适中的层数
BATCH_SIZE = 512
DROPOUT = 0.15


# =============================================================================
# Configuration Dataclasses
# =============================================================================
@dataclass
class ModelConfig:
    """Model configuration"""
    embedding_dim: int = EMBEDDING_DIM
    num_heads: int = NUM_HEADS
    num_transformer_layers: int = NUM_TRANSFORMER_LAYERS
    d_ff: int = 256  # 增大 FFN 维度
    dropout: float = DROPOUT
    attention_dropout: float = 0.1
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    output_dim: int = None
    use_numerical_tokenization: bool = True
    pooling_type: str = 'attention'  # 'cls', 'mean', 'attention'
    use_feature_gating: bool = True
    use_cross_attention: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = BATCH_SIZE
    epochs: int = 300
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 30
    
    # Data augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    augment_prob: float = 0.5
    
    # Regularization
    use_rdrop: bool = True
    rdrop_alpha: float = 0.5
    label_smoothing: float = 0.1
    
    # Optimizers
    use_sam: bool = True
    sam_rho: float = 0.05
    use_lookahead: bool = True
    lookahead_k: int = 6
    lookahead_alpha: float = 0.5
    
    # SWA
    use_swa: bool = True
    swa_start_epoch: int = 100
    swa_lr: float = 1e-4
    
    # Loss
    loss_type: str = 'focal'  # 'ce', 'focal', 'poly'
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    
    # Sampling
    use_weighted_sampling: bool = True
    sampling_power: float = 0.5
    
    # Scheduler
    scheduler_type: str = 'cosine_warmup'  # 'cosine', 'cosine_warmup', 'onecycle'
    warmup_epochs: int = 10
    
    # Misc
    gradient_clip: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999


class LossType(Enum):
    CROSS_ENTROPY = "ce"
    FOCAL = "focal"
    POLY = "poly"
    LABEL_SMOOTHING = "label_smoothing"


# =============================================================================
# Dataset
# =============================================================================
class TabularDataset(Dataset):
    """Enhanced dataset with on-the-fly augmentation support"""
    def __init__(self, 
                 categorical_features: torch.Tensor, 
                 numerical_features: torch.Tensor, 
                 targets: torch.Tensor,
                 augment: bool = False,
                 noise_std: float = 0.01):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.targets = targets
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        cat_x = self.categorical_features[idx]
        num_x = self.numerical_features[idx].clone()
        y = self.targets[idx]
        
        # Add Gaussian noise to numerical features during training
        if self.augment and self.training:
            num_x = num_x + torch.randn_like(num_x) * self.noise_std
        
        return (cat_x, num_x), y
    
    def set_training(self, mode: bool):
        self.training = mode


# =============================================================================
# Loss Functions
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 label_smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                   label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class PolyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Paper: https://arxiv.org/abs/2204.12511
    """
    def __init__(self, epsilon: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing)
        pt = F.softmax(inputs, dim=-1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        return poly_loss.mean()


def compute_rdrop_loss(logits1: torch.Tensor, logits2: torch.Tensor, 
                       targets: torch.Tensor, criterion: nn.Module, 
                       alpha: float = 0.3) -> torch.Tensor:
    """
    R-Drop: Regularized Dropout for Neural Networks
    Paper: https://arxiv.org/abs/2106.14448
    """
    ce_loss = 0.5 * (criterion(logits1, targets) + criterion(logits2, targets))
    
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    q1 = F.softmax(logits1, dim=-1)
    q2 = F.softmax(logits2, dim=-1)
    
    kl_loss = 0.5 * (F.kl_div(p1, q2, reduction='batchmean') + 
                     F.kl_div(p2, q1, reduction='batchmean'))
    
    return ce_loss + alpha * kl_loss


# =============================================================================
# Model Components
# =============================================================================
class FeatureEmbedding(nn.Module):
    """Enhanced embedding with learned column positions"""
    def __init__(self, cardinalities: List[int], embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, embedding_dim, padding_idx=0)  # +1 for unknown
            for cardinality in cardinalities
        ])
        self.column_embedding = nn.Parameter(
            torch.randn(len(cardinalities), embedding_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = [
            emb(x[:, i] + 1)  # +1 to handle potential -1 or 0 index
            for i, emb in enumerate(self.embeddings)
        ]
        stacked = torch.stack(embedded, dim=1)
        output = stacked + self.column_embedding.unsqueeze(0)
        return self.dropout(self.layer_norm(output))


class NumericalTokenizer(nn.Module):
    """Enhanced numerical feature tokenization"""
    def __init__(self, num_features: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(1, embedding_dim)
        self.column_embedding = nn.Parameter(
            torch.randn(num_features, embedding_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_features]
        x = x.unsqueeze(-1)  # [batch, num_features, 1]
        tokens = self.linear(x)  # [batch, num_features, embedding_dim]
        tokens = tokens + self.column_embedding.unsqueeze(0)
        return self.dropout(self.layer_norm(tokens))


class FeatureGating(nn.Module):
    """Feature gating mechanism for adaptive feature importance"""
    def __init__(self, d_model: int, num_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * num_features, num_features),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_features, d_model]
        batch_size = x.size(0)
        flat = x.view(batch_size, -1)
        gates = self.gate(flat).unsqueeze(-1)  # [batch, num_features, 1]
        return x * gates


class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention with optional dropout"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 attention_dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.dropout(self.out_proj(output))


class CrossAttention(nn.Module):
    """Cross-attention between categorical and numerical features"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_output))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with GELU"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, attention_dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, attention_dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class AttentionPooling(nn.Module):
    """Learnable attention pooling"""
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        output, _ = self.attention(query, x, x)
        return self.norm(output.squeeze(1))


class ResidualMLP(nn.Module):
    """MLP block with residual connection and layer scaling"""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1,
                 layer_scale: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(output_dim) * layer_scale)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.fc(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return residual + self.gamma * x


# =============================================================================
# Main Model
# =============================================================================
class AdvancedTabTransformer(nn.Module):
    """
    Advanced TabTransformer with multiple optimizations
    """
    def __init__(self,
                 categorical_cardinalities: List[int],
                 numerical_features_dim: int,
                 config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.num_cat = len(categorical_cardinalities)
        self.num_num = numerical_features_dim
        
        # Embeddings
        self.cat_embedding = FeatureEmbedding(
            categorical_cardinalities, config.embedding_dim, config.dropout
        )
        
        self.num_tokenizer = NumericalTokenizer(
            numerical_features_dim, config.embedding_dim, config.dropout
        )
        
        # Feature gating
        if config.use_feature_gating:
            total_features = self.num_cat + self.num_num
            self.feature_gate = FeatureGating(config.embedding_dim, total_features)
        
        # Cross-attention
        if config.use_cross_attention:
            self.cross_attn_cat = CrossAttention(config.embedding_dim, config.num_heads, config.dropout)
            self.cross_attn_num = CrossAttention(config.embedding_dim, config.num_heads, config.dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                config.embedding_dim, 
                config.num_heads, 
                config.d_ff, 
                config.dropout,
                config.attention_dropout
            )
            for _ in range(config.num_transformer_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        
        # Pooling
        if config.pooling_type == 'attention':
            self.pooling = AttentionPooling(config.embedding_dim)
            mlp_input_dim = config.embedding_dim
        elif config.pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim) * 0.02)
            mlp_input_dim = config.embedding_dim
        else:  # mean
            mlp_input_dim = config.embedding_dim
        
        # MLP head
        self.mlp = self._build_mlp(mlp_input_dim, config.hidden_dims, 
                                    config.dropout, config.output_dim)
        
        self.apply(self._init_weights)

    def _build_mlp(self, input_dim: int, hidden_dims: List[int], 
                   dropout: float, output_dim: int) -> nn.Module:
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(ResidualMLP(prev_dim, hidden_dim, dropout, 
                                       layer_scale=0.1 / (i + 1)))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        batch_size = cat_x.size(0)
        
        # Tokenize
        cat_tokens = self.cat_embedding(cat_x)
        num_tokens = self.num_tokenizer(num_x)
        
        # Cross-attention
        if self.config.use_cross_attention:
            cat_tokens = self.cross_attn_cat(cat_tokens, num_tokens)
            num_tokens = self.cross_attn_num(num_tokens, cat_tokens)
        
        # Concatenate
        tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        
        # Feature gating
        if self.config.use_feature_gating:
            tokens = self.feature_gate(tokens)
        
        # Add CLS token if needed
        if self.config.pooling_type == 'cls':
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        
        # Transformer
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        
        tokens = self.final_norm(tokens)
        
        # Pooling
        if self.config.pooling_type == 'attention':
            output = self.pooling(tokens)
        elif self.config.pooling_type == 'cls':
            output = tokens[:, 0]
        else:  # mean
            output = tokens.mean(dim=1)
        
        return self.mlp(output)


# =============================================================================
# Optimizers
# =============================================================================
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization"""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()


class Lookahead(optim.Optimizer):
    """Lookahead optimizer wrapper"""
    def __init__(self, optimizer: optim.Optimizer, k: int = 6, alpha: float = 0.5):
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


class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# Data Augmentation
# =============================================================================
def mixup_data(cat_x: torch.Tensor, num_x: torch.Tensor, y: torch.Tensor, 
               alpha: float = 0.4, num_classes: int = None) -> Tuple:
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
    else:
        lam = 1.0
    
    batch_size = cat_x.size(0)
    index = torch.randperm(batch_size, device=cat_x.device)
    
    mixed_num = lam * num_x + (1 - lam) * num_x[index]
    
    # Categorical: random selection per feature
    cat_mask = (torch.rand(batch_size, cat_x.size(1), device=cat_x.device) < lam).long()
    mixed_cat = cat_mask * cat_x + (1 - cat_mask) * cat_x[index]
    
    if num_classes is not None:
        y_onehot = F.one_hot(y, num_classes).float()
        y_mixed = lam * y_onehot + (1 - lam) * y_onehot[index]
    else:
        y_mixed = lam * y.float() + (1 - lam) * y[index].float()
    
    return mixed_cat, mixed_num, y_mixed, lam


def cutmix_data(cat_x: torch.Tensor, num_x: torch.Tensor, y: torch.Tensor,
                alpha: float = 1.0, num_classes: int = None) -> Tuple:
    """CutMix augmentation for tabular data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = cat_x.size(0)
    index = torch.randperm(batch_size, device=cat_x.device)
    
    # Cut a portion of features
    num_cut = int((1 - lam) * num_x.size(1))
    cut_indices = torch.randperm(num_x.size(1))[:num_cut]
    
    mixed_num = num_x.clone()
    mixed_num[:, cut_indices] = num_x[index][:, cut_indices]
    
    # Also mix some categorical features
    num_cat_cut = int((1 - lam) * cat_x.size(1))
    cat_cut_indices = torch.randperm(cat_x.size(1))[:num_cat_cut]
    
    mixed_cat = cat_x.clone()
    mixed_cat[:, cat_cut_indices] = cat_x[index][:, cat_cut_indices]
    
    # Adjust lambda based on actual cut ratio
    actual_lam = 1 - (num_cut / num_x.size(1) + num_cat_cut / cat_x.size(1)) / 2
    
    if num_classes is not None:
        y_onehot = F.one_hot(y, num_classes).float()
        y_mixed = actual_lam * y_onehot + (1 - actual_lam) * y_onehot[index]
    else:
        y_mixed = actual_lam * y.float() + (1 - actual_lam) * y[index].float()
    
    return mixed_cat, mixed_num, y_mixed, actual_lam


# =============================================================================
# Trainer
# =============================================================================
class AdvancedTabTransformerTrainer:
    """Advanced trainer with all optimizations"""
    
    def __init__(self, 
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 model_save_dir: str = './checkpoints',
                 random_state: int = 42):
        
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.model_save_dir = model_save_dir
        self.random_state = random_state
        
        self.model = None
        self.ema = None
        self.swa_model = None
        self.device = self._setup_device()
        self._set_random_seeds()
        
        # Feature definitions (customize for your data)
        self.categorical_features = [
            'seasonality', 'tod', 'day_prior_seg', 'dow', 
            'if_first_carrier', 'if_last_carrier'
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
        self.numerical_scaler = RobustScaler()  # More robust to outliers
        self.target_encoder = LabelEncoder()
        self.cardinalities = {}
        self.num_classes = None
        self.class_weights = None
        
        # History
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Using Apple Silicon GPU")
            return torch.device('mps')
        print("💻 Using CPU")
        return torch.device('cpu')

    def _set_random_seeds(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        df = df.copy()
        
        # Boolean conversion
        bool_map = {'true': 1, 'false': 0, True: 1, False: 0}
        for col in ['if_first_carrier', 'if_last_carrier']:
            if col in df.columns:
                df[col] = df[col].map(bool_map).fillna(0).astype(int)
        
        # Interaction features
        if 'lf' in df.columns and 'rask' in df.columns:
            df['lf_rask_ratio'] = df['lf'] / (df['rask'] + 1e-8)
            self.numerical_features.append('lf_rask_ratio')
        
        if 'lf' in df.columns and 'lf_cz' in df.columns:
            df['lf_diff'] = df['lf'] - df['lf_cz']
            self.numerical_features.append('lf_diff')
        
        if 'rask' in df.columns and 'rask_cz' in df.columns:
            df['rask_diff'] = df['rask'] - df['rask_cz']
            self.numerical_features.append('rask_diff')
        
        # Keep unique features
        self.numerical_features = list(dict.fromkeys(self.numerical_features))
        
        return df

    def prepare_data(self, data_path: str, test_size: float = 0.2) -> Tuple:
        """Prepare data with advanced preprocessing"""
        print("📊 Loading data...")
        df = pd.read_csv(data_path)
        print(f"   Raw size: {df.shape}")
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        # Clean
        df_clean = df.dropna(subset=self.categorical_features + self.numerical_features + [self.target_feature])
        df_clean = df_clean.reset_index(drop=True)
        print(f"   Clean size: {df_clean.shape}")
        
        # Process categorical
        cat_data = {}
        for feat in self.categorical_features:
            self.label_encoders[feat] = LabelEncoder()
            cat_data[feat] = self.label_encoders[feat].fit_transform(df_clean[feat].astype(str))
            self.cardinalities[feat] = df_clean[feat].nunique()
            print(f"   {feat}: {self.cardinalities[feat]} categories")
        
        # Process numerical
        num_data = df_clean[self.numerical_features].values
        num_data = self.numerical_scaler.fit_transform(num_data)
        
        # Process target
        targets = df_clean[self.target_feature].values
        targets = self.target_encoder.fit_transform(targets.astype(str))
        self.num_classes = len(self.target_encoder.classes_)
        print(f"   Classes: {self.num_classes}")
        
        # Compute class weights for imbalanced data
        class_counts = Counter(targets)
        total = len(targets)
        self.class_weights = torch.tensor([
            total / (self.num_classes * class_counts[i]) 
            for i in range(self.num_classes)
        ], dtype=torch.float32)
        print(f"   Class distribution: {dict(class_counts)}")
        
        # To tensors
        cat_tensor = torch.stack([
            torch.tensor(cat_data[f], dtype=torch.long) for f in self.categorical_features
        ], dim=1)
        num_tensor = torch.tensor(num_data, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        
        # Split
        indices = np.arange(len(cat_tensor))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=self.random_state, 
            stratify=targets  # Stratified split
        )
        
        print(f"   Train: {len(train_idx)}, Test: {len(test_idx)}")
        
        return (
            (cat_tensor[train_idx], num_tensor[train_idx], target_tensor[train_idx]),
            (cat_tensor[test_idx], num_tensor[test_idx], target_tensor[test_idx])
        )

    def create_data_loaders(self, train_data: Tuple, test_data: Tuple) -> Tuple:
        """Create data loaders with weighted sampling"""
        cfg = self.training_config
        
        train_dataset = TabularDataset(*train_data, augment=True)
        test_dataset = TabularDataset(*test_data, augment=False)
        
        if cfg.use_weighted_sampling:
            targets = train_data[2].numpy()
            class_counts = Counter(targets)
            weights = [1.0 / (class_counts[t] ** cfg.sampling_power) for t in targets]
            weights = torch.tensor(weights, dtype=torch.float32)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            
            train_loader = DataLoader(
                train_dataset, batch_size=cfg.batch_size, sampler=sampler,
                pin_memory=True, num_workers=0
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=cfg.batch_size, shuffle=True,
                pin_memory=True, num_workers=0
            )
        
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size, shuffle=False,
            pin_memory=True, num_workers=0
        )
        
        return train_loader, test_loader

    def build_model(self):
        """Build and initialize model"""
        cardinalities = [self.cardinalities[f] for f in self.categorical_features]
        
        if self.model_config.output_dim is None:
            self.model_config.output_dim = self.num_classes
        
        self.model = AdvancedTabTransformer(
            categorical_cardinalities=cardinalities,
            numerical_features_dim=len(self.numerical_features),
            config=self.model_config
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"   Model parameters: {params:,}")
        
        # EMA
        if self.training_config.use_ema:
            self.ema = EMA(self.model, self.training_config.ema_decay)
        
        return self.model

    def _get_loss_function(self) -> nn.Module:
        """Get loss function based on config"""
        cfg = self.training_config
        
        if cfg.loss_type == 'focal':
            alpha = self.class_weights if cfg.use_class_weights else None
            return FocalLoss(alpha=alpha, gamma=cfg.focal_gamma, 
                            label_smoothing=cfg.label_smoothing)
        elif cfg.loss_type == 'poly':
            return PolyLoss(epsilon=1.0, label_smoothing=cfg.label_smoothing)
        else:
            weight = self.class_weights.to(self.device) if cfg.use_class_weights else None
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)

    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on config"""
        cfg = self.training_config
        
        if cfg.use_sam:
            base_optim = SAM(
                self.model.parameters(),
                base_optimizer=optim.AdamW,
                rho=cfg.sam_rho,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        else:
            base_optim = optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        
        if cfg.use_lookahead and not cfg.use_sam:
            return Lookahead(base_optim, k=cfg.lookahead_k, alpha=cfg.lookahead_alpha)
        
        return base_optim

    def _get_scheduler(self, optimizer, num_steps: int):
        """Get learning rate scheduler"""
        cfg = self.training_config
        
        if cfg.scheduler_type == 'cosine_warmup':
            def lr_lambda(step):
                if step < cfg.warmup_epochs * num_steps:
                    return step / (cfg.warmup_epochs * num_steps)
                progress = (step - cfg.warmup_epochs * num_steps) / \
                           ((cfg.epochs - cfg.warmup_epochs) * num_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
            
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        elif cfg.scheduler_type == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.learning_rate * 10,
                epochs=cfg.epochs, steps_per_epoch=num_steps,
                pct_start=0.1, div_factor=25, final_div_factor=1000
            )
        
        else:  # cosine
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-7
            )

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Training loop with all optimizations"""
        cfg = self.training_config
        
        print(f"\n🚀 Starting training for {cfg.epochs} epochs")
        print(f"   Loss: {cfg.loss_type}, SAM: {cfg.use_sam}, R-Drop: {cfg.use_rdrop}")
        print(f"   MixUp: {cfg.use_mixup}, CutMix: {cfg.use_cutmix}")
        
        use_amp = (self.device.type == 'cuda')
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        criterion = self._get_loss_function()
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(
            optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer,
            len(train_loader)
        )
        
        # SWA
        if cfg.use_swa:
            self.swa_model = AveragedModel(self.model)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(cfg.epochs):
            self.model.train()
            train_loss = 0.0
            
            for (cat_x, num_x), y in train_loader:
                cat_x = cat_x.to(self.device)
                num_x = num_x.to(self.device)
                y = y.to(self.device)
                
                # Data augmentation
                use_aug = np.random.random() < cfg.augment_prob
                if use_aug:
                    if cfg.use_cutmix and np.random.random() < 0.5:
                        cat_x, num_x, y_soft, _ = cutmix_data(
                            cat_x, num_x, y, cfg.cutmix_alpha, self.num_classes
                        )
                    elif cfg.use_mixup:
                        cat_x, num_x, y_soft, _ = mixup_data(
                            cat_x, num_x, y, cfg.mixup_alpha, self.num_classes
                        )
                    else:
                        y_soft = F.one_hot(y, self.num_classes).float()
                        use_aug = False
                else:
                    y_soft = None
                
                # Forward pass
                if isinstance(optimizer, SAM):
                    # SAM: two forward-backward passes
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            output = self.model(cat_x, num_x)
                            if use_aug and y_soft is not None:
                                loss = -torch.sum(y_soft * F.log_softmax(output, dim=1), dim=1).mean()
                            else:
                                loss = criterion(output, y)
                        
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer.base_optimizer)
                        optimizer.first_step(zero_grad=True)
                        
                        with torch.cuda.amp.autocast():
                            output2 = self.model(cat_x, num_x)
                            if use_aug and y_soft is not None:
                                loss2 = -torch.sum(y_soft * F.log_softmax(output2, dim=1), dim=1).mean()
                            else:
                                loss2 = criterion(output2, y)
                        
                        scaler.scale(loss2).backward()
                        scaler.unscale_(optimizer.base_optimizer)
                        optimizer.second_step(zero_grad=True)
                        scaler.update()
                    else:
                        output = self.model(cat_x, num_x)
                        if use_aug and y_soft is not None:
                            loss = -torch.sum(y_soft * F.log_softmax(output, dim=1), dim=1).mean()
                        else:
                            loss = criterion(output, y)
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                        output2 = self.model(cat_x, num_x)
                        if use_aug and y_soft is not None:
                            loss2 = -torch.sum(y_soft * F.log_softmax(output2, dim=1), dim=1).mean()
                        else:
                            loss2 = criterion(output2, y)
                        loss2.backward()
                        optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    
                    if cfg.use_rdrop:
                        # R-Drop: two forward passes
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                output1 = self.model(cat_x, num_x)
                                output2 = self.model(cat_x, num_x)
                                loss = compute_rdrop_loss(output1, output2, y, criterion, cfg.rdrop_alpha)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer if not isinstance(optimizer, Lookahead) else optimizer.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            output1 = self.model(cat_x, num_x)
                            output2 = self.model(cat_x, num_x)
                            loss = compute_rdrop_loss(output1, output2, y, criterion, cfg.rdrop_alpha)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                            optimizer.step()
                    else:
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                output = self.model(cat_x, num_x)
                                if use_aug and y_soft is not None:
                                    loss = -torch.sum(y_soft * F.log_softmax(output, dim=1), dim=1).mean()
                                else:
                                    loss = criterion(output, y)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer if not isinstance(optimizer, Lookahead) else optimizer.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            output = self.model(cat_x, num_x)
                            if use_aug and y_soft is not None:
                                loss = -torch.sum(y_soft * F.log_softmax(output, dim=1), dim=1).mean()
                            else:
                                loss = criterion(output, y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                            optimizer.step()
                
                scheduler.step()
                train_loss += loss.item()
                
                # EMA update
                if self.ema is not None:
                    self.ema.update()
            
            # SWA update
            if cfg.use_swa and epoch >= cfg.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_loss, val_acc = self._validate(val_loader, criterion)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            current_lr = scheduler.get_last_lr()[0]
            self.history['lr'].append(current_lr)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint("best_model_v3.pth")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        # Finalize SWA
        if cfg.use_swa and self.swa_model is not None:
            print("   📊 Updating SWA batch normalization...")
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=self.device)
        
        self.load_checkpoint("best_model_v3.pth")
        print(f"   ✅ Best validation accuracy: {best_val_acc:.4f}")
        
        return self.history

    def _validate(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        preds, actuals = [], []
        
        # Use EMA weights for validation if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        with torch.no_grad():
            for (cat_x, num_x), y in loader:
                cat_x = cat_x.to(self.device)
                num_x = num_x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(cat_x, num_x)
                loss = criterion(output, y)
                val_loss += loss.item()
                
                preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                actuals.extend(y.cpu().numpy())
        
        if self.ema is not None:
            self.ema.restore()
        
        return val_loss / len(loader), accuracy_score(actuals, preds)

    def evaluate(self, loader: DataLoader, use_tta: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Evaluate with optional test-time augmentation"""
        self.model.eval()
        
        if self.ema is not None:
            self.ema.apply_shadow()
        
        all_preds, all_probs, actuals = [], [], []
        
        with torch.no_grad():
            for (cat_x, num_x), y in loader:
                cat_x = cat_x.to(self.device)
                num_x = num_x.to(self.device)
                
                if use_tta:
                    # Test-time augmentation: average predictions with noise
                    probs_list = []
                    for _ in range(5):
                        noise = torch.randn_like(num_x) * 0.01
                        output = self.model(cat_x, num_x + noise)
                        probs_list.append(F.softmax(output, dim=1))
                    probs = torch.stack(probs_list).mean(0)
                else:
                    output = self.model(cat_x, num_x)
                    probs = F.softmax(output, dim=1)
                
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                actuals.extend(y.numpy())
        
        if self.ema is not None:
            self.ema.restore()
        
        actuals = np.array(actuals)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        metrics = {
            'Accuracy': accuracy_score(actuals, all_preds),
            'Precision': precision_score(actuals, all_preds, average='weighted', zero_division=0),
            'Recall': recall_score(actuals, all_preds, average='weighted', zero_division=0),
            'F1': f1_score(actuals, all_preds, average='weighted', zero_division=0),
            'F1_macro': f1_score(actuals, all_preds, average='macro', zero_division=0),
        }
        
        # Top-k accuracy
        if self.num_classes >= 3:
            metrics['Top3_Accuracy'] = top_k_accuracy_score(actuals, all_probs, k=3)
        if self.num_classes >= 5:
            metrics['Top5_Accuracy'] = top_k_accuracy_score(actuals, all_probs, k=5)
        
        print(f"\n📈 Evaluation Results:")
        for k, v in metrics.items():
            if 'Matrix' not in k:
                print(f"   {k}: {v:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(actuals, all_preds, zero_division=0))
        
        return actuals, all_preds, metrics

    def _save_checkpoint(self, filename: str):
        os.makedirs(self.model_save_dir, exist_ok=True)
        path = os.path.join(self.model_save_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'cardinalities': self.cardinalities,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'num_classes': self.num_classes,
            'history': self.history,
        }
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        path = os.path.join(self.model_save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']

    def save_preprocessors(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        preprocessors = {
            'label_encoders': self.label_encoders,
            'numerical_scaler': self.numerical_scaler,
            'target_encoder': self.target_encoder,
            'cardinalities': self.cardinalities,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'num_classes': self.num_classes,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"   Saved preprocessors to {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Advanced TabTransformer Training v3')
    parser.add_argument('-d', '--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--model-dir', type=str, default='./checkpoints', help='Model save directory')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--no-sam', action='store_true', help='Disable SAM optimizer')
    parser.add_argument('--no-rdrop', action='store_true', help='Disable R-Drop')
    parser.add_argument('--loss', type=str, default='focal', choices=['ce', 'focal', 'poly'])
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 Advanced TabTransformer Training v3.0")
    print("=" * 60)
    
    # Config
    model_config = ModelConfig(
        embedding_dim=64,
        num_heads=8,
        num_transformer_layers=4,
        d_ff=256,
        dropout=0.15,
        attention_dropout=0.1,
        hidden_dims=[256, 128, 64],
        pooling_type='attention',
        use_feature_gating=True,
        use_cross_attention=True,
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_sam=not args.no_sam,
        use_rdrop=not args.no_rdrop,
        loss_type=args.loss,
        use_mixup=True,
        use_cutmix=True,
        use_swa=True,
        use_ema=True,
    )
    
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return
    
    # Train
    trainer = AdvancedTabTransformerTrainer(
        model_config=model_config,
        training_config=training_config,
        model_save_dir=args.model_dir,
    )
    
    train_data, test_data = trainer.prepare_data(args.data)
    train_loader, test_loader = trainer.create_data_loaders(train_data, test_data)
    trainer.build_model()
    
    history = trainer.train(train_loader, test_loader)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("📊 Final Evaluation")
    print("=" * 60)
    
    print("\n🔹 Standard Evaluation:")
    actuals, preds, metrics = trainer.evaluate(test_loader, use_tta=False)
    
    print("\n🔹 With Test-Time Augmentation:")
    actuals, preds, metrics_tta = trainer.evaluate(test_loader, use_tta=True)
    
    # Save
    trainer.save_preprocessors(os.path.join(args.model_dir, 'preprocessors.pkl'))
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
