# 数据加载器快速参考

## 🚀 一分钟上手

```python
from dataloader import create_dataloader

# 创建 DataLoader
dataloader = create_dataloader(
    annotation_file='data/annotations.csv',
    video_dir='data/videos',
    mode='realtime',
    batch_size=8
)

# 使用
for batch in dataloader:
    faces = batch['faces']          # (8, 16, 3, 160, 160)
    openfaces = batch['openfaces']  # (8, 16, 714)
    audios = batch['audios']        # (8, 768)
    labels = batch['labels']        # (8,)
```

---

## 📦 三个提取器

### FaceExtractor
```python
from dataloader import FaceExtractor

extractor = FaceExtractor(method='opencv', num_frames=16)
faces = extractor('video.mp4')  # (16, 3, 160, 160)
```

### OpenFaceExtractor
```python
from dataloader import OpenFaceExtractor

extractor = OpenFaceExtractor(num_frames=16)
openfaces = extractor('video.mp4')  # (16, 714)
```

### AudioExtractor
```python
from dataloader import AudioExtractor

extractor = AudioExtractor(method='wav2vec2', device='cuda')
audios = extractor('video.mp4', from_video=True)  # (768,)
```

---

## 📋 标注文件格式

```csv
video_id,label,split
video_001,0,train
video_002,1,train
video_003,0,val
```

---

## ⚡ 预提取特征 (推荐)

```bash
python dataloader/precompute_features.py \
    --annotation_file data/annotations.csv \
    --video_dir data/videos \
    --output_dir data/features \
    --device cuda
```

然后使用预提取模式:

```python
dataloader = create_dataloader(
    annotation_file='data/annotations.csv',
    mode='precomputed',
    precomputed_dir='data/features',
    batch_size=32
)
```

---

## 🎯 完整训练示例

```python
from dataloader import create_dataloader
from plan import FusionModel
import torch.nn as nn
import torch.optim as optim

# 数据
train_loader = create_dataloader(
    annotation_file='data/annotations.csv',
    mode='precomputed',
    precomputed_dir='data/features',
    batch_size=8,
    split='train'
)

# 模型
model = FusionModel(device='cuda').cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 训练
model.train()
for batch in train_loader:
    faces = batch['faces'].cuda()
    openfaces = batch['openfaces'].cuda()
    audios = batch['audios'].cuda()
    labels = batch['labels'].cuda()
    
    output = model(faces, openfaces, audios)
    loss = criterion(output['logits']['fused'], labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 🔧 常用参数

### FaceExtractor
| 参数 | 默认值 | 说明 |
|-----|--------|------|
| method | 'opencv' | 人脸检测方法 |
| target_size | (160, 160) | 输出图像大小 |
| num_frames | 16 | 提取的帧数 |
| device | 'cpu' | 设备 |

### OpenFaceExtractor
| 参数 | 默认值 | 说明 |
|-----|--------|------|
| openface_path | None | OpenFace 路径 |
| num_frames | 16 | 采样的帧数 |
| feature_dim | 714 | 特征维度 |

### AudioExtractor
| 参数 | 默认值 | 说明 |
|-----|--------|------|
| method | 'wav2vec2' | 提取方法 |
| model_name | 'facebook/wav2vec2-base' | 模型名称 |
| sample_rate | 16000 | 采样率 |
| device | 'cpu' | 设备 |

### LieDetectionDataset
| 参数 | 默认值 | 说明 |
|-----|--------|------|
| mode | 'realtime' | 'realtime' 或 'precomputed' |
| split | None | 'train', 'val', 'test' |
| transform | None | 数据增强 |

---

## 💡 最佳实践

1. **训练时使用预提取模式** - 快 10-100 倍
2. **使用 GPU** - `device='cuda'`
3. **合理设置 num_workers** - 通常设为 CPU 核心数
4. **批量大小** - 根据 GPU 内存调整 (8-32)
5. **人脸检测** - opencv (快) 或 mtcnn (准)

---

## ⚠️ 常见错误

### 错误 1: OpenFace 未找到
```python
# 解决: 指定路径
extractor = OpenFaceExtractor(
    openface_path='/path/to/FeatureExtraction'
)
```

### 错误 2: FFmpeg 未找到
```bash
# 解决: 安装 FFmpeg
# Windows: https://ffmpeg.org/download.html
# Linux: sudo apt-get install ffmpeg
```

### 错误 3: 内存不足
```python
# 解决: 减小 batch_size 和 num_workers
dataloader = create_dataloader(
    ...,
    batch_size=4,
    num_workers=2
)
```

---

## 📊 输出形状速查

| 数据 | 形状 | 说明 |
|-----|------|------|
| faces | (B, T, 3, H, W) | B=批次, T=帧数, H=W=160 |
| openfaces | (B, T, 714) | 714 维特征 |
| audios | (B, 768) | 768 维特征 |
| labels | (B,) | 0=真话, 1=谎言 |

---

**详细文档**: 查看 `DATALOADER_GUIDE.md`
