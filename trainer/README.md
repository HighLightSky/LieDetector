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


## 模块化架构（新版）

训练器现已采用模块化设计，将训练逻辑拆分为多个独立、可复用的组件。

### 核心模块

1. **LossComputer** (`loss_computer.py`) - 多任务损失计算
2. **MetricsEvaluator** (`metrics_evaluator.py`) - 指标评估
3. **TrainingLoop** (`training_loop.py`) - 训练循环
4. **ValidationLoop** (`validation_loop.py`) - 验证循环
5. **CheckpointManager** (`checkpoint_manager.py`) - 检查点管理
6. **HistoryTracker** (`history_tracker.py`) - 历史记录

### 使用示例

#### 基本使用（与旧版兼容）

```python
from trainer import FusionModelTrainer
from models.fusion import FusionModel

# 创建模型和数据加载器
model = FusionModel(...)
train_loader = DataLoader(...)
val_loader = DataLoader(...)

# 创建训练器
trainer = FusionModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    save_dir='checkpoints'
)

# 开始训练
trainer.train(num_epochs=30, lr=1e-5)
```

#### 高级使用（自定义模块）

```python
from trainer import (
    FusionModelTrainer, LossComputer, 
    TrainingLoop, ValidationLoop
)

# 自定义损失权重
custom_loss_computer = LossComputer(loss_weights={
    'face': 0.3,
    'audio': 0.3,
    'fused': 1.5
})

# 创建训练器并替换模块
trainer = FusionModelTrainer(...)
trainer.loss_computer = custom_loss_computer
```

#### 独立使用模块

```python
# 只使用检查点管理
from trainer import CheckpointManager

checkpoint_manager = CheckpointManager('my_checkpoints')
checkpoint_manager.save('model.pth', model.state_dict(), epoch=10)

# 只使用指标评估
from trainer import MetricsEvaluator

metrics = MetricsEvaluator()
for batch in dataloader:
    metrics.update(loss, predictions, labels, batch_size)
print(f"Accuracy: {metrics.get_accuracy():.2f}%")
```

### 模块化优势

1. **代码可读性** - 每个模块职责清晰
2. **可维护性** - 修改某个功能只需修改对应模块
3. **可测试性** - 每个模块可以独立单元测试
4. **可复用性** - 模块可以在不同项目中复用
5. **可扩展性** - 易于添加新功能或替换现有模块

### 详细文档

- [训练器模块化封装方案](../docs/方案-训练器模块化封装.md) - 完整的模块化设计说明
- [训练器模块关系图](../docs/架构-训练器模块关系图.md) - 架构和数据流图

### 测试

运行测试脚本验证模块功能：

```bash
python test_modular_trainer.py
```
