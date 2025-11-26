#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Optimized TabTransformer
验证模型架构和前向传播是否正确
"""

import torch
import numpy as np
from optimized_tab_transformer import (
    OptimizedTabTransformer,
    ModelConfig,
    FeatureEmbedding,
    NumericalFeatureTokenizer,
    TransformerBlock,
    MultiHeadAttention,
    ResidualMLPBlock,
    mixup_data,
    Lookahead
)

def test_feature_embedding():
    """测试类别特征嵌入层"""
    print("Testing FeatureEmbedding...")
    cardinalities = [10, 5, 20, 7]  # 4 categorical features
    embedding_dim = 32
    batch_size = 16
    
    embed = FeatureEmbedding(cardinalities, embedding_dim)
    
    # Create fake input
    cat_input = torch.stack([
        torch.randint(0, c, (batch_size,)) for c in cardinalities
    ], dim=1)
    
    output = embed(cat_input)
    
    assert output.shape == (batch_size, len(cardinalities), embedding_dim), \
        f"Expected shape {(batch_size, len(cardinalities), embedding_dim)}, got {output.shape}"
    
    print(f"  ✅ Input shape: {cat_input.shape}")
    print(f"  ✅ Output shape: {output.shape}")
    

def test_numerical_tokenizer():
    """测试数值特征 Tokenization"""
    print("Testing NumericalFeatureTokenizer...")
    num_features = 10
    embedding_dim = 32
    batch_size = 16
    
    tokenizer = NumericalFeatureTokenizer(num_features, embedding_dim)
    
    num_input = torch.randn(batch_size, num_features)
    output = tokenizer(num_input)
    
    assert output.shape == (batch_size, num_features, embedding_dim), \
        f"Expected shape {(batch_size, num_features, embedding_dim)}, got {output.shape}"
    
    print(f"  ✅ Input shape: {num_input.shape}")
    print(f"  ✅ Output shape: {output.shape}")


def test_multi_head_attention():
    """测试多头注意力"""
    print("Testing MultiHeadAttention...")
    d_model = 32
    num_heads = 4
    seq_len = 10
    batch_size = 16
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = mha(x)
    
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"
    
    print(f"  ✅ Input shape: {x.shape}")
    print(f"  ✅ Output shape: {output.shape}")


def test_transformer_block():
    """测试 Transformer Block (Pre-LN)"""
    print("Testing TransformerBlock (Pre-LN)...")
    d_model = 32
    num_heads = 4
    d_ff = 128
    seq_len = 10
    batch_size = 16
    
    block = TransformerBlock(d_model, num_heads, d_ff)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
    
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"
    
    print(f"  ✅ Input shape: {x.shape}")
    print(f"  ✅ Output shape: {output.shape}")


def test_residual_mlp_block():
    """测试残差 MLP Block"""
    print("Testing ResidualMLPBlock...")
    input_dim = 128
    output_dim = 64
    batch_size = 16
    
    block = ResidualMLPBlock(input_dim, output_dim)
    
    x = torch.randn(batch_size, input_dim)
    output = block(x)
    
    assert output.shape == (batch_size, output_dim), \
        f"Expected shape {(batch_size, output_dim)}, got {output.shape}"
    
    print(f"  ✅ Input shape: {x.shape}")
    print(f"  ✅ Output shape: {output.shape}")


def test_optimized_tab_transformer():
    """测试完整的 OptimizedTabTransformer"""
    print("Testing OptimizedTabTransformer...")
    
    # Configuration
    categorical_cardinalities = [10, 5, 20, 7, 2, 2, 100]  # 7 categorical features
    numerical_features_dim = 17
    batch_size = 32
    
    config = ModelConfig(
        embedding_dim=32,
        num_heads=4,
        num_transformer_layers=3,
        d_ff=128,
        dropout=0.1,
        hidden_dims=[128, 64],
        output_dim=1,
        use_numerical_tokenization=True,
        use_cls_token=True
    )
    
    model = OptimizedTabTransformer(
        categorical_cardinalities=categorical_cardinalities,
        numerical_features_dim=numerical_features_dim,
        config=config
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Trainable parameters: {trainable_params:,}")
    
    # Create fake input
    cat_input = torch.stack([
        torch.randint(0, c, (batch_size,)) for c in categorical_cardinalities
    ], dim=1)
    num_input = torch.randn(batch_size, numerical_features_dim)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(cat_input, num_input)
    
    assert output.shape == (batch_size,), \
        f"Expected shape {(batch_size,)}, got {output.shape}"
    
    print(f"  ✅ Categorical input shape: {cat_input.shape}")
    print(f"  ✅ Numerical input shape: {num_input.shape}")
    print(f"  ✅ Output shape: {output.shape}")
    
    # Test backward pass
    model.train()
    cat_input.requires_grad_(False)
    num_input.requires_grad_(True)
    
    output = model(cat_input, num_input)
    loss = output.mean()
    loss.backward()
    
    print(f"  ✅ Backward pass successful")


def test_optimized_tab_transformer_no_cls():
    """测试不使用 CLS token 的 TabTransformer"""
    print("Testing OptimizedTabTransformer (no CLS token)...")
    
    categorical_cardinalities = [10, 5, 20, 7]
    numerical_features_dim = 10
    batch_size = 16
    
    config = ModelConfig(
        embedding_dim=32,
        num_heads=4,
        num_transformer_layers=2,
        use_numerical_tokenization=True,
        use_cls_token=False  # No CLS token
    )
    
    model = OptimizedTabTransformer(
        categorical_cardinalities=categorical_cardinalities,
        numerical_features_dim=numerical_features_dim,
        config=config
    )
    
    cat_input = torch.stack([
        torch.randint(0, c, (batch_size,)) for c in categorical_cardinalities
    ], dim=1)
    num_input = torch.randn(batch_size, numerical_features_dim)
    
    model.eval()
    with torch.no_grad():
        output = model(cat_input, num_input)
    
    assert output.shape == (batch_size,), \
        f"Expected shape {(batch_size,)}, got {output.shape}"
    
    print(f"  ✅ Output shape: {output.shape}")


def test_optimized_tab_transformer_no_tokenization():
    """测试不使用数值特征 Tokenization 的 TabTransformer"""
    print("Testing OptimizedTabTransformer (no numerical tokenization)...")
    
    categorical_cardinalities = [10, 5, 20, 7]
    numerical_features_dim = 10
    batch_size = 16
    
    config = ModelConfig(
        embedding_dim=32,
        num_heads=4,
        num_transformer_layers=2,
        use_numerical_tokenization=False,  # No tokenization
        use_cls_token=False
    )
    
    model = OptimizedTabTransformer(
        categorical_cardinalities=categorical_cardinalities,
        numerical_features_dim=numerical_features_dim,
        config=config
    )
    
    cat_input = torch.stack([
        torch.randint(0, c, (batch_size,)) for c in categorical_cardinalities
    ], dim=1)
    num_input = torch.randn(batch_size, numerical_features_dim)
    
    model.eval()
    with torch.no_grad():
        output = model(cat_input, num_input)
    
    assert output.shape == (batch_size,), \
        f"Expected shape {(batch_size,)}, got {output.shape}"
    
    print(f"  ✅ Output shape: {output.shape}")


def test_mixup():
    """测试 MixUp 数据增强"""
    print("Testing MixUp augmentation...")
    
    batch_size = 16
    num_cat_features = 5
    num_numerical_features = 10
    
    cat_x = torch.randint(0, 10, (batch_size, num_cat_features))
    num_x = torch.randn(batch_size, num_numerical_features)
    y = torch.randn(batch_size)
    
    mixed_cat, mixed_num, mixed_y, lam = mixup_data(cat_x, num_x, y, alpha=0.2)
    
    assert mixed_cat.shape == cat_x.shape
    assert mixed_num.shape == num_x.shape
    assert mixed_y.shape == y.shape
    assert 0 <= lam <= 1
    
    print(f"  ✅ Lambda: {lam:.4f}")
    print(f"  ✅ Mixed categorical shape: {mixed_cat.shape}")
    print(f"  ✅ Mixed numerical shape: {mixed_num.shape}")
    print(f"  ✅ Mixed target shape: {mixed_y.shape}")


def test_lookahead_optimizer():
    """测试 Lookahead 优化器"""
    print("Testing Lookahead optimizer...")
    
    model = torch.nn.Linear(10, 1)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    
    # Simulate training
    for i in range(20):
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    print(f"  ✅ Lookahead optimizer working correctly")


def test_gradient_flow():
    """测试梯度流动"""
    print("Testing gradient flow...")
    
    categorical_cardinalities = [10, 5, 20]
    numerical_features_dim = 10
    batch_size = 8
    
    config = ModelConfig(
        embedding_dim=32,
        num_heads=4,
        num_transformer_layers=3,
        use_numerical_tokenization=True,
        use_cls_token=True
    )
    
    model = OptimizedTabTransformer(
        categorical_cardinalities=categorical_cardinalities,
        numerical_features_dim=numerical_features_dim,
        config=config
    )
    
    cat_input = torch.stack([
        torch.randint(0, c, (batch_size,)) for c in categorical_cardinalities
    ], dim=1)
    num_input = torch.randn(batch_size, numerical_features_dim, requires_grad=True)
    target = torch.randn(batch_size)
    
    # Forward
    output = model(cat_input, num_input)
    loss = torch.nn.functional.mse_loss(output, target)
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_nan_grad = False
    has_zero_grad = True
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan_grad = True
                print(f"  ⚠️ NaN gradient in {name}")
            if param.grad.abs().max() > 0:
                has_zero_grad = False
    
    assert not has_nan_grad, "Found NaN gradients!"
    assert not has_zero_grad, "All gradients are zero!"
    
    print(f"  ✅ No NaN gradients")
    print(f"  ✅ Gradients flowing correctly")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 Running Optimized TabTransformer Tests")
    print("=" * 60 + "\n")
    
    try:
        test_feature_embedding()
        print()
        
        test_numerical_tokenizer()
        print()
        
        test_multi_head_attention()
        print()
        
        test_transformer_block()
        print()
        
        test_residual_mlp_block()
        print()
        
        test_optimized_tab_transformer()
        print()
        
        test_optimized_tab_transformer_no_cls()
        print()
        
        test_optimized_tab_transformer_no_tokenization()
        print()
        
        test_mixup()
        print()
        
        test_lookahead_optimizer()
        print()
        
        test_gradient_flow()
        print()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    run_all_tests()
