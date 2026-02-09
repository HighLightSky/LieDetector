# 训练器使用指南

## 概述

本训练器专为 `FusionModel` 设计，采用多任务学习策略，同时优化各个模态和融合结果。

## 训练策略

### 核心特点

1. **冻结预训练Backbone**
   - MobileNetV3 backbone 保持冻结
   - 只训练融合层、注意力模块、分类头

2. **多任务学习**
   - 同时优化6个任务：
     - Face分类
     - OpenFace分类
     - Audio分类
     - Face-Audio融合分类
     - Face-OpenFace融合分类
     - 最终融合分类

3. **加权损失**
   - 各任务损失可配置权重
   - 默认融合损失权重最高

4. **早停机制**
   - 基于验证准确率
   - 可配置耐心值

## 快速开始

### 基本训练

```bash
# 使用默认参数
python train.py

# 指定数据路径
python train.py --csv_path src/dataset/video_labels.csv --features_dir src/features

# 调整训练参数
python train.py --num_epochs 50 --lr 1e-4 --batch_size 16
```

### 高级选项

```bash
# 自定义模型结构
python train.py \
    --visual_hidden 512 \
    --fusion_hidden 256 \
    --uni_layers 3 \
    --uni_nhead 8 \
    --com_heads 8

# 自定义损失权重
python train.py \
    --loss_weight_face 0.3 \
    --loss_weight_audio 0.3 \
    --loss_weight_fused 1.5

# 从检查点恢复
python train.py --resume checkpoints/latest_model.pth

# 使用CPU
python train.py --device cpu
```

## 完整训练流程

```bash
# 1. 确保特征已提取
python dataloader/precompute_features.py

# 2. 开始训练
python train.py --num_epochs 30 --batch_size 8 --lr 1e-4

# 3. 使用训练好的模型进行预测
python detect.py --checkpoint checkpoints/best_model.pth --video path/to/video.mp4
```

## 最佳实践

1. **数据准备**
   - 确保特征已预先提取
   - 检查数据集平衡性
   - 使用足够的验证集（20%以上）

2. **超参数调优**
   - 从默认参数开始
   - 逐步调整学习率
   - 根据验证集表现调整损失权重

3. **监控训练**
   - 观察详细损失变化
   - 检查各模态准确率
   - 注意过拟合迹象

4. **模型保存**
   - 定期备份检查点
   - 保存训练历史用于分析
   - 记录最佳超参数配置
