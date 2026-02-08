# 多模态谎言检测系统

基于深度学习的多模态融合谎言检测系统，通过分析面部表情、面部动作单元（OpenFace）和语音特征来判断说话者是否在说谎。

## 🎯 项目特点

- ✅ **三模态融合**: Face + OpenFace + Audio
- ✅ **预训练模型**: MobileNetV3 + Wav2Vec2
- ✅ **动态权重**: 自适应学习模态重要性
- ✅ **完整数据加载器**: 开箱即用的特征提取工具
- ✅ **详细文档**: 完整的使用指南和API文档

## 📦 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_win.txt
```

### 2. 测试模型

```bash
python test_models.py
```

### 3. 使用数据加载器

```python
from dataloader import create_dataloader

# 创建 DataLoader
dataloader = create_dataloader(
    annotation_file='data/annotations.csv',
    video_dir='data/videos',
    mode='realtime',
    batch_size=8
)

# 训练
for batch in dataloader:
    faces = batch['faces']          # (8, 16, 3, 160, 160)
    openfaces = batch['openfaces']  # (8, 16, 714)
    audios = batch['audios']        # (8, 768)
    labels = batch['labels']        # (8,)
```

## 📚 文档

- **[数据加载器指南](docs/DATALOADER_GUIDE.md)** - 详细的数据加载器使用说明
- **[快速参考](docs/DATALOADER_QUICK_REF.md)** - 常用代码片段和API速查

## 🏗️ 项目结构

```
lie_detector/
├── models/              # 模型定义
│   ├── faces.py        # 面部表情模型
│   ├── openface.py     # OpenFace特征分类器
│   └── audio.py        # 音频模型
├── dataloader/          # 数据加载器 ⭐
│   ├── face_extractor.py
│   ├── openface_extractor.py
│   ├── audio_extractor.py
│   ├── dataset.py
│   └── precompute_features.py
├── docs/               # 文档
├── plan.py            # 核心融合模型
└── test_models.py     # 测试脚本
```

## 🔧 数据准备

### 方式 1: 实时提取（快速原型）

```python
from dataloader import FaceExtractor, OpenFaceExtractor, AudioExtractor

# 提取单个视频的特征
face_extractor = FaceExtractor(method='opencv', num_frames=16)
openface_extractor = OpenFaceExtractor(num_frames=16)
audio_extractor = AudioExtractor(method='wav2vec2', device='cuda')

faces = face_extractor('video.mp4')
openfaces = openface_extractor('video.mp4')
audios = audio_extractor('video.mp4', from_video=True)
```

### 方式 2: 预提取特征（推荐用于训练）

```bash
python dataloader/precompute_features.py \
    --annotation_file data/annotations.csv \
    --video_dir data/videos \
    --output_dir data/features \
    --device cuda
```

## 🎓 使用示例

### 训练模型

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

# 训练循环
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

### 推理

```python
model.eval()
with torch.no_grad():
    output = model(faces, openfaces, audios)
    probs = output['probs']['fused']  # (B, 2)
    pred = probs.argmax(dim=1)        # 0=真话, 1=谎言
```

## 📊 模型架构

```
输入: Face (B,T,3,160,160) + OpenFace (B,T,714) + Audio (B,768)
  ↓
子模型: FacesModel + OpenfaceModel + AudioModel
  ↓
单模态编码: Transformer Encoder
  ↓
跨模态注意力: Face↔Audio, Face↔OpenFace
  ↓
动态权重融合: W * [P_face, P_of, P_audio]
  ↓
输出: 融合概率 (B, 2)
```

## 🔑 核心特性

### 1. 三个子模型
- **FacesModel**: MobileNetV3 提取面部特征 (1024维)
- **OpenfaceModel**: 处理面部动作单元 (714维)
- **AudioModel**: Wav2Vec2 提取音频特征 (768维)

### 2. 融合策略
- **单模态编码器**: Transformer 处理时序信息
- **跨模态注意力**: 捕捉模态间互补信息
- **动态权重模块**: 自适应学习模态重要性

### 3. 数据加载器
- **FaceExtractor**: 人脸检测和裁剪
- **OpenFaceExtractor**: OpenFace 特征提取
- **AudioExtractor**: Wav2Vec2/MFCC 特征提取
- **LieDetectionDataset**: 整合数据集

## 📝 标注文件格式

```csv
video_id,label,split
video_001,0,train
video_002,1,train
video_003,0,val
```

- `video_id`: 视频文件名（不含扩展名）
- `label`: 0=真话, 1=谎言
- `split`: train/val/test（可选）

## ⚙️ 依赖项

### 核心依赖
- PyTorch 2.5.1
- transformers 4.46.3
- timm 1.0.19
- opencv-python 4.12.0
- librosa 0.11.0

### 外部工具
- **FFmpeg**: 音频提取（必需）
- **OpenFace**: 面部动作单元提取（可选）

## 🚀 性能

- **参数量**: ~5.3M
- **推理速度**: ~100 FPS (GPU)
- **内存占用**: ~80 MB (batch_size=8)

## 📖 详细文档

查看 `docs/` 目录获取完整文档：

- **DATALOADER_GUIDE.md**: 数据加载器详细指南
- **DATALOADER_QUICK_REF.md**: 快速参考手册

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

**最后更新**: 2025-02-08
