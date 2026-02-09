# NaN问题修复说明

## 问题描述

训练时大量批次出现NaN损失，导致批次被跳过，最终所有批次都被跳过导致除零错误。

```
[WARNING] NaN/Inf loss detected, skipping batch
训练失败: float division by zero
```

## 根本原因

1. **Transformer层不稳定**: 单模态编码器中的Transformer在训练初期容易产生梯度爆炸
2. **初始化过于激进**: 权重初始化的gain值过大（0.5），导致激活值过大
3. **跨模态注意力数值不稳定**: MultiheadAttention在某些输入下产生NaN
4. **学习率过高**: 默认学习率1e-4对于复杂模型过高

## 修复措施

### 1. 更保守的权重初始化

```python
# 之前: gain=0.5
nn.init.xavier_uniform_(m.weight, gain=0.5)

# 现在: gain=0.1（降低5倍）
nn.init.xavier_uniform_(m.weight, gain=0.1)
```

### 2. 多层NaN检查和替换

在关键位置添加NaN检查，如果检测到NaN则用零向量或回退值替代：

- **单模态编码输出**: U_face, U_of, U_audio
- **跨模态注意力输入**: X_face1, X_au1, X_face2, X_of1
- **跨模态注意力输出**: C_fa_au, C_fa_of
- **分类logits**: logits_fa_au, logits_fa_of
- **融合logits**: fused_logits

```python
# 示例
if torch.isnan(U_face).any():
    U_face = torch.zeros_like(U_face)
```

### 3. 降低学习率

```python
# 之前: lr=1e-4
# 现在: lr=5e-5（降低50%）
```

### 4. 优化器数值稳定性

```python
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=lr,
    weight_decay=weight_decay,
    eps=1e-8  # 增加eps提高数值稳定性
)
```

### 5. 防止除零错误

在trainer中添加检查，如果所有批次都被跳过则返回错误信息：

```python
if total == 0:
    print("\n[ERROR] 所有批次都被跳过（NaN/Inf），无法计算损失")
    return {
        'loss': float('inf'),
        'accuracy': 0.0,
        'loss_detail': {key: float('inf') for key in loss_details.keys()}
    }
```

## 使用方法

### 1. 测试单批次（推荐先运行）

```cmd
python test_nan_fix.py
```

这会测试单个批次是否产生NaN，输出详细的诊断信息。

### 2. 开始训练

```cmd
python train.py --num_epochs 10 --batch_size 8
```

默认学习率已降低到5e-5，更加稳定。

### 3. 如果仍有问题

尝试进一步降低学习率或减小模型复杂度：

```cmd
# 更低的学习率
python train.py --lr 1e-5

# 更小的模型
python train.py --visual_hidden 128 --dropout 0.5

# 更小的批次
python train.py --batch_size 4
```

## 预期效果

修复后应该看到：

1. **没有或很少NaN警告**: 偶尔1-2个批次可能仍有NaN（会被跳过），但不应该大量出现
2. **损失正常下降**: 所有6个损失（face, openface, audio, fa_au, fa_of, fused）都应该显示数值
3. **准确率提升**: 训练和验证准确率应该逐步提升

## 技术细节

### NaN产生的常见位置

1. **Transformer Self-Attention**: 当Q·K^T的值过大时，softmax会产生NaN
2. **LayerNorm**: 当输入方差接近0时，除以标准差会产生Inf
3. **梯度爆炸**: 深层网络中梯度累积导致参数更新过大

### 为什么用零向量替代

当检测到NaN时，用零向量替代是一种"安全回退"策略：

- 零向量不会传播NaN
- 对损失的贡献为0，不会破坏其他正常样本的训练
- 允许训练继续进行，而不是完全失败

这比跳过整个批次更好，因为批次中可能只有部分样本有问题。

## 监控建议

训练时注意观察：

1. **NaN警告频率**: 应该很少或没有
2. **损失趋势**: 应该平稳下降，不应该突然跳变
3. **梯度范数**: 可以添加梯度范数监控来提前发现问题

```python
# 在trainer中添加
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"梯度范数: {grad_norm:.4f}")
```

## 参考

- [Understanding NaN in Neural Networks](https://stackoverflow.com/questions/33962226)
- [Gradient Clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [Xavier Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_)
