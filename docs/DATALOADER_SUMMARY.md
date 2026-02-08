# 数据加载器模块总结

## 📦 已创建的文件

### 核心代码 (6个文件)

```
dataloader/
├── __init__.py                  # 模块初始化
├── face_extractor.py            # 面部图像提取器 (300+ 行)
├── openface_extractor.py        # OpenFace 特征提取器 (300+ 行)
├── audio_extractor.py           # 音频特征提取器 (300+ 行)
├── dataset.py                   # PyTorch 数据集 (250+ 行)
└── precompute_features.py       # 预提取工具 (150+ 行)
```

### 文档 (4个文件)

```
docs/
├── DATALOADER_GUIDE.md          # 详细使用指南 (800+ 行)
├── DATALOADER_QUICK_REF.md      # 快速参考 (200+ 行)
├── INSTALLATION.md              # 安装指南
└── DATALOADER_SUMMARY.md        # 本文档
```

### 测试 (1个文件)

```
test_dataloader.py               # 数据加载器测试脚本
```

---

## 🎯 功能概览

### 1. FaceExtractor - 面部图像提取器

**功能**:
- 从视频中检测并提取面部图像序列
- 支持多种人脸检测方法 (opencv, dlib, mtcnn)
- 自动裁剪、调整大小、归一化

**输入**: 视频文件或图像列表
**输出**: `torch.Tensor (T, 3, 160, 160)`

**关键方法**:
```python
extract_from_video(video_path)    # 从视频提取
extract_from_images(image_paths)  # 从图像列表提取
detect_face(frame)                # 检测单帧人脸
crop_face(frame, bbox)            # 裁剪人脸
```

---

### 2. OpenFaceExtractor - OpenFace 特征提取器

**功能**:
- 使用 OpenFace 工具提取面部动作单元 (AU)
- 提取 714 维特征向量
- 包含关键点、头部姿态、眼睛注视等

**输入**: 视频文件或 OpenFace CSV 文件
**输出**: `torch.Tensor (T, 714)`

**关键方法**:
```python
extract_from_video(video_path)    # 从视频提取
extract_from_csv(csv_path)        # 从 CSV 提取
batch_extract(video_paths)        # 批量提取
```

**特征组成**:
- 2D 面部关键点: 136 维
- 3D 面部关键点: 204 维
- 头部姿态: 6 维
- 眼睛注视: 6 维
- 动作单元: ~34 维

---

### 3. AudioExtractor - 音频特征提取器

**功能**:
- 使用 Wav2Vec2 或 MFCC 提取音频特征
- 自动从视频中提取音频
- 支持批量处理

**输入**: 音频文件或视频文件
**输出**: `torch.Tensor (768,)` 或 `(40,)`

**关键方法**:
```python
extract_from_audio(audio_path)    # 从音频提取
extract_from_video(video_path)    # 从视频提取
extract_audio_from_video(...)     # 提取音频文件
batch_extract(paths)              # 批量提取
```

**支持的方法**:
- `wav2vec2`: 768 维，效果好
- `mfcc`: 40 维，速度快

---

### 4. LieDetectionDataset - 数据集类

**功能**:
- 整合三种模态的数据
- 支持实时提取和预提取两种模式
- 自动处理批次数据

**输入**: 标注文件 + 视频目录或特征目录
**输出**: 批次字典

**关键方法**:
```python
__getitem__(idx)                  # 获取单个样本
collate_fn(batch)                 # 批次整理
```

**输出格式**:
```python
{
    'faces': (B, T, 3, 160, 160),
    'openfaces': (B, T, 714),
    'audios': (B, 768),
    'labels': (B,),
    'video_ids': List[str]
}
```

---

### 5. precompute_features.py - 预提取工具

**功能**:
- 批量处理视频
- 提取并保存特征到磁盘
- 加速训练过程

**使用**:
```bash
python dataloader/precompute_features.py \
    --annotation_file data/annotations.csv \
    --video_dir data/videos \
    --output_dir data/features \
    --device cuda
```

**输出结构**:
```
data/features/
├── video_001/
│   ├── faces.pt
│   ├── openfaces.pt
│   └── audios.pt
├── video_002/
│   ├── faces.pt
│   ├── openfaces.pt
│   └── audios.pt
...
```

---

## 🚀 使用流程

### 流程 1: 快速原型（实时提取）

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
    # 使用 batch['faces'], batch['openfaces'], batch['audios']
    pass
```

**优点**: 简单，无需预处理
**缺点**: 慢，每次都要提取特征

---

### 流程 2: 生产训练（预提取）

**步骤 1**: 预提取特征
```bash
python dataloader/precompute_features.py \
    --annotation_file data/annotations.csv \
    --video_dir data/videos \
    --output_dir data/features \
    --device cuda
```

**步骤 2**: 使用预提取特征
```python
dataloader = create_dataloader(
    annotation_file='data/annotations.csv',
    mode='precomputed',
    precomputed_dir='data/features',
    batch_size=32
)
```

**优点**: 快，适合大规模训练
**缺点**: 需要预处理，占用磁盘空间

---

## 📊 性能对比

| 模式 | 速度 | 内存 | 磁盘 | 适用场景 |
|-----|------|------|------|---------|
| 实时提取 | ⚡ 慢 | 💾 低 | 💿 无 | 快速原型 |
| 预提取 | ⚡⚡⚡ 快 | 💾 低 | 💿 高 | 生产训练 |

**推荐**: 开发时使用实时提取，训练时使用预提取

---

## 🔧 配置选项

### FaceExtractor 配置

```python
FaceExtractor(
    method='opencv',           # 检测方法: opencv/dlib/mtcnn
    target_size=(160, 160),    # 输出大小
    num_frames=16,             # 帧数
    device='cpu'               # 设备
)
```

**method 选择**:
- `opencv`: 快，精度中
- `dlib`: 中，精度高
- `mtcnn`: 慢，精度很高

---

### OpenFaceExtractor 配置

```python
OpenFaceExtractor(
    openface_path=None,        # OpenFace 路径
    num_frames=16,             # 帧数
    feature_dim=714,           # 特征维度
    device='cpu'               # 设备
)
```

---

### AudioExtractor 配置

```python
AudioExtractor(
    method='wav2vec2',         # 方法: wav2vec2/mfcc
    model_name='facebook/wav2vec2-base',
    sample_rate=16000,         # 采样率
    device='cpu'               # 设备
)
```

**method 选择**:
- `wav2vec2`: 效果好，推荐
- `mfcc`: 速度快，传统方法

---

### LieDetectionDataset 配置

```python
LieDetectionDataset(
    annotation_file,           # 标注文件
    video_dir=None,            # 视频目录
    mode='realtime',           # realtime/precomputed
    precomputed_dir=None,      # 预提取目录
    split=None,                # train/val/test
    transform=None             # 数据增强
)
```

---

## 📝 标注文件格式

```csv
video_id,label,split
video_001,0,train
video_002,1,train
video_003,0,val
video_004,1,test
```

**字段说明**:
- `video_id`: 视频文件名（不含扩展名）
- `label`: 0=真话, 1=谎言
- `split`: train/val/test（可选）

---

## 🎓 完整示例

### 示例 1: 提取单个视频

```python
from dataloader import FaceExtractor, OpenFaceExtractor, AudioExtractor

# 初始化
face_ext = FaceExtractor(method='opencv', num_frames=16)
of_ext = OpenFaceExtractor(num_frames=16)
audio_ext = AudioExtractor(method='wav2vec2', device='cuda')

# 提取
video_path = 'path/to/video.mp4'
faces = face_ext(video_path)          # (16, 3, 160, 160)
openfaces = of_ext(video_path)        # (16, 714)
audios = audio_ext(video_path, from_video=True)  # (768,)

print(f"Faces: {faces.shape}")
print(f"OpenFaces: {openfaces.shape}")
print(f"Audios: {audios.shape}")
```

---

### 示例 2: 训练模型

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
for epoch in range(num_epochs):
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

## ⚠️ 注意事项

### 1. 依赖安装

确保安装所有依赖:
```bash
pip install opencv-python librosa soundfile tqdm
```

### 2. 外部工具

- **FFmpeg**: 必需，用于音频提取
- **OpenFace**: 可选，用于精细特征

### 3. 性能优化

- 使用 GPU: `device='cuda'`
- 预提取特征: 使用 `precompute_features.py`
- 合理设置 `num_workers`: 通常为 CPU 核心数

### 4. 错误处理

代码已包含错误处理:
- 人脸检测失败 → 使用上一帧
- 特征提取失败 → 返回零特征
- 文件不存在 → 警告并跳过

---

## 📚 文档索引

1. **DATALOADER_GUIDE.md**: 详细使用指南
2. **DATALOADER_QUICK_REF.md**: 快速参考
3. **INSTALLATION.md**: 安装指南
4. **DATALOADER_SUMMARY.md**: 本文档

---

## 🎯 下一步

1. ✅ 安装依赖: 查看 `INSTALLATION.md`
2. ✅ 准备数据: 创建标注文件
3. ✅ 预提取特征: 运行 `precompute_features.py`
4. ✅ 开始训练: 使用 `create_dataloader`

---

## 🤝 贡献

数据加载器模块是训练和推理的基础，欢迎:
- 报告 Bug
- 提出改进建议
- 贡献代码

---

**版本**: v1.0  
**最后更新**: 2025-02-08  
**作者**: Kiro AI Assistant
