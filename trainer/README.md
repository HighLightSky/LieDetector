# 训练器模块

多模态谎言检测模型的分阶段训练器。

## 文件结构

```
trainer/
├── __init__.py          # 模块初始化
├── dataset.py           # 数据集类
├── trainer.py           # 训练器类
└── README.md            # 本文件
```

## 分阶段训练策略

### 阶段 1: 训练融合层
- **目标**: 学习如何融合三个子模型的输出
- **冻结**: 所有子模型（FacesModel, AudioModel, OpenfaceModel）
- **训练**: 只训练融合层
- **学习率**: 1e-3（较高）
- **轮数**: 10 epochs

### 阶段 2: 微调分类头
- **目标**: 微调子模型的分类头以适应融合层
- **冻结**: 预训练骨干（MobileNetV3, Wav2Vec2）
- **训练**: 子模型的分类头 + 融合层
- **学习率**: 5e-4（中等）
- **轮数**: 10 epochs

### 阶段 3: 端到端微调
- **目标**: 整体优化所有可训练参数
- **冻结**: 只冻结预训练骨干
- **训练**: 所有分类头 + 融合层
- **学习率**: 1e-4（较低）
- **轮数**: 10 epochs

## 使用方法

### 1. 测试训练（小批量）

```bash
python test_train.py
```

这会使用 `src/test_data/` 中的少量样本进行快速测试，验证训练流程是否正常。

### 2. 完整训练

```bash
# 基本用法
python train.py

# 自定义参数
python train.py \
    --csv_path src/dataset/video_labels.csv \
    --features_dir src/features \
    --batch_size 16 \
    --stage1_epochs 10 \
    --stage2_epochs 10 \
    --stage3_epochs 10 \
    --device cuda
```

### 3. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--csv_path` | `src/dataset/video_labels.csv` | 标签文件路径 |
| `--features_dir` | `src/features` | 特征目录 |
| `--use_openface` | `True` | 是否使用 OpenFace |
| `--batch_size` | `16` | 批大小 |
| `--num_workers` | `4` | 数据加载线程数 |
| `--val_split` | `0.2` | 验证集比例 |
| `--stage1_epochs` | `10` | 阶段1轮数 |
| `--stage2_epochs` | `10` | 阶段2轮数 |
| `--stage3_epochs` | `10` | 阶段3轮数 |
| `--stage1_lr` | `1e-3` | 阶段1学习率 |
| `--stage2_lr` | `5e-4` | 阶段2学习率 |
| `--stage3_lr` | `1e-4` | 阶段3学习率 |
| `--device` | `cuda` | 训练设备 |
| `--save_dir` | `checkpoints` | 保存目录 |
| `--seed` | `42` | 随机种子 |

## 数据集格式

### 标签 CSV 文件

```csv
video_name,label
AN_WILTY_EP15_lie10,deception
AN_WILTY_EP15_truth1,truth
...
```

### 特征文件

每个视频需要以下特征文件：

```
features_dir/
├── {video_name}_faces.pt      # 人脸特征 (T, C, H, W)
├── {video_name}_audio.pt      # 音频特征 (768,)
└── {video_name}_openface.pt   # OpenFace 特征 (T, 714) [可选]
```

## 输出文件

训练完成后会生成以下文件：

```
checkpoints/
├── best_stage1.pth           # 阶段1最佳模型
├── best_stage2.pth           # 阶段2最佳模型
├── best_stage3.pth           # 阶段3最佳模型（最终模型）
├── latest_stage1.pth         # 阶段1最新模型
├── latest_stage2.pth         # 阶段2最新模型
├── latest_stage3.pth         # 阶段3最新模型
└── training_history.json     # 训练历史
```

## 训练历史

`training_history.json` 包含：

```json
{
  "train_loss": [...],
  "train_acc": [...],
  "val_loss": [...],
  "val_acc": [...],
  "stage": [1, 1, ..., 2, 2, ..., 3, 3, ...]
}
```

可以用于绘制训练曲线。

## 代码示例

### 使用训练器

```python
from torch.utils.data import DataLoader
from detect import MultiModalFusionModel
from trainer import MultiModalTrainer, LieDetectionDataset

# 创建数据集
dataset = LieDetectionDataset(
    csv_path='src/dataset/video_labels.csv',
    features_dir='src/features',
    use_openface=True
)

# 创建数据加载器
train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=LieDetectionDataset.collate_fn
)

val_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=LieDetectionDataset.collate_fn
)

# 创建模型
model = MultiModalFusionModel(device='cuda', use_openface=True)

# 创建训练器
trainer = MultiModalTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    save_dir='checkpoints'
)

# 训练所有阶段
trainer.train_all_stages(
    stage1_epochs=10,
    stage2_epochs=10,
    stage3_epochs=10,
    stage1_lr=1e-3,
    stage2_lr=5e-4,
    stage3_lr=1e-4
)
```

### 加载训练好的模型

```python
from detect import MultiModalFusionModel
import torch

# 创建模型
model = MultiModalFusionModel(device='cuda', use_openface=True)

# 加载权重
checkpoint = torch.load('checkpoints/best_stage3.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"最佳验证准确率: {checkpoint['best_val_acc']:.2f}%")
```

## 注意事项

1. **内存占用**: 训练需要较大内存，建议至少 16GB RAM + 8GB VRAM
2. **数据预处理**: 训练前需要预计算所有视频的特征
3. **批大小**: 根据 GPU 内存调整 batch_size
4. **学习率**: 可以根据训练曲线调整学习率
5. **早停**: 训练器使用 ReduceLROnPlateau 自动调整学习率
6. **检查点**: 每个阶段都会保存最佳和最新模型

## 故障排除

### 问题: 数据集为空

**原因**: 特征文件不存在

**解决**: 先提取特征
```bash
python -c "from dataloader import precompute_features; precompute_features.main()"
```

### 问题: CUDA out of memory

**原因**: GPU 内存不足

**解决**: 
- 减小 batch_size
- 减少 num_frames
- 使用 CPU 训练（较慢）

### 问题: 训练速度慢

**原因**: 数据加载瓶颈

**解决**:
- 增加 num_workers
- 使用 SSD 存储特征文件
- 预加载数据到内存

## 性能优化建议

1. **数据增强**: 在 dataset.py 中添加数据增强
2. **混合精度训练**: 使用 torch.cuda.amp 加速训练
3. **梯度累积**: 模拟更大的 batch_size
4. **学习率调度**: 尝试不同的学习率策略
5. **正则化**: 调整 dropout 和 weight_decay
