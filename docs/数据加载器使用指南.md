# 数据加载器使用指南

## 📚 目录

- [概述](#概述)
- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
  - [FaceExtractor](#faceextractor)
  - [OpenFaceExtractor](#openfaceextractor)
  - [AudioExtractor](#audioextractor)
  - [LieDetectionDataset](#liedetectiondataset)
- [数据准备流程](#数据准备流程)
- [常见问题](#常见问题)

---

## 🎯 概述

数据加载器模块提供了完整的多模态特征提取和数据加载功能，包括：

1. **FaceExtractor**: 从视频中检测并提取面部图像序列
2. **OpenFaceExtractor**: 使用 OpenFace 工具提取面部动作单元特征
3. **AudioExtractor**: 使用 Wav2Vec2 或 MFCC 提取音频特征
4. **LieDetectionDataset**: 整合三种模态的 PyTorch 数据集

---

## 📦 安装依赖

### 基础依赖

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install librosa
pip install transformers
pip install pandas tqdm
```

### 可选依赖

```bash
# 用于更好的人脸检测
pip install dlib
pip install facenet-pytorch

# 用于 OpenFace (需要单独安装)
# 参考: https://github.com/TadasBaltrusaitis/OpenFace
```

### 外部工具

1. **FFmpeg** (必需，用于音频提取)
   - Windows: https://ffmpeg.org/download.html
   - Linux: `sudo apt-get install ffmpeg`
   - Mac: `brew install ffmpeg`

2. **OpenFace** (可选，用于精细面部特征)
   - 下载: https://github.com/TadasBaltrusaitis/OpenFace
   - 编译并添加到 PATH

---

## 🚀 快速开始

### 示例 1: 提取单个视频的特征

```python
from dataloader import FaceExtractor, OpenFaceExtractor, AudioExtractor

# 初始化提取器
face_extractor = FaceExtractor(method='opencv', num_frames=16)
openface_extractor = OpenFaceExtractor(num_frames=16)
audio_extractor = AudioExtractor(method='wav2vec2', device='cuda')

# 提取特征
video_path = 'path/to/video.mp4'

faces = face_extractor(video_path)          # (16, 3, 160, 160)
openfaces = openface_extractor(video_path)  # (16, 714)
audios = audio_extractor(video_path, from_video=True)  # (768,)

print(f"Faces: {faces.shape}")
print(f"OpenFaces: {openfaces.shape}")
print(f"Audios: {audios.shape}")
```

### 示例 2: 使用数据集和 DataLoader

```python
from dataloader import create_dataloader

# 创建 DataLoader
dataloader = create_dataloader(
    annotation_file='data/annotations.csv',
    video_dir='data/videos',
    mode='realtime',
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# 训练循环
for batch in dataloader:
    faces = batch['faces']          # (8, 16, 3, 160, 160)
    openfaces = batch['openfaces']  # (8, 16, 714)
    audios = batch['audios']        # (8, 768)
    labels = batch['labels']        # (8,)
    
    # 训练模型...
```

---

## 📖 详细使用说明

### FaceExtractor

从视频中检测并提取面部图像序列。

#### 初始化参数

```python
FaceExtractor(
    method='opencv',           # 人脸检测方法: 'opencv', 'dlib', 'mtcnn'
    target_size=(160, 160),    # 输出图像大小
    num_frames=16,             # 提取的帧数，None 表示全部
    device='cpu'               # 设备
)
```

#### 方法

##### `extract_from_video(video_path, return_tensor=True, normalize=True)`

从视频中提取面部图像序列。

**参数**:
- `video_path`: 视频文件路径
- `return_tensor`: 是否返回 PyTorch tensor (默认 True)
- `normalize`: 是否归一化到 [0, 1] (默认 True)

**返回**:
- `torch.Tensor (T, 3, H, W)` 或 `np.ndarray (T, H, W, 3)`

**示例**:
```python
extractor = FaceExtractor(method='opencv', num_frames=16)

# 返回 tensor
faces = extractor.extract_from_video('video.mp4')
print(faces.shape)  # (16, 3, 160, 160)

# 返回 numpy array
faces_np = extractor.extract_from_video('video.mp4', return_tensor=False)
print(faces_np.shape)  # (16, 160, 160, 3)
```

##### `extract_from_images(image_paths, return_tensor=True, normalize=True)`

从图像列表中提取面部。

**参数**:
- `image_paths`: 图像文件路径列表
- `return_tensor`: 是否返回 PyTorch tensor
- `normalize`: 是否归一化

**示例**:
```python
image_paths = ['frame1.jpg', 'frame2.jpg', 'frame3.jpg']
faces = extractor.extract_from_images(image_paths)
print(faces.shape)  # (3, 3, 160, 160)
```

##### `__call__(source, **kwargs)`

便捷调用接口，自动判断输入类型。

**示例**:
```python
# 从视频提取
faces = extractor('video.mp4')

# 从图像列表提取
faces = extractor(['img1.jpg', 'img2.jpg'])
```

#### 人脸检测方法对比

| 方法 | 速度 | 精度 | 依赖 |
|-----|------|------|------|
| opencv | ⚡⚡⚡ 快 | ⭐⭐ 中 | opencv-python |
| dlib | ⚡⚡ 中 | ⭐⭐⭐ 高 | dlib |
| mtcnn | ⚡ 慢 | ⭐⭐⭐⭐ 很高 | facenet-pytorch |

**推荐**:
- 快速原型: `opencv`
- 生产环境: `dlib` 或 `mtcnn`

---

### OpenFaceExtractor

使用 OpenFace 工具提取面部动作单元和关键点特征。

#### 初始化参数

```python
OpenFaceExtractor(
    openface_path=None,        # OpenFace 可执行文件路径，None 表示自动查找
    num_frames=16,             # 采样的帧数
    feature_dim=714,           # 特征维度
    device='cpu'               # 设备
)
```

#### 方法

##### `extract_from_video(video_path, return_tensor=True)`

从视频中提取 OpenFace 特征。

**参数**:
- `video_path`: 视频文件路径
- `return_tensor`: 是否返回 PyTorch tensor

**返回**:
- `torch.Tensor (T, 714)` 或 `np.ndarray (T, 714)`

**示例**:
```python
extractor = OpenFaceExtractor(num_frames=16)

# 从视频提取
features = extractor.extract_from_video('video.mp4')
print(features.shape)  # (16, 714)
```

##### `extract_from_csv(csv_path, return_tensor=True)`

从 OpenFace 输出的 CSV 文件中提取特征。

**参数**:
- `csv_path`: OpenFace 输出的 CSV 文件路径
- `return_tensor`: 是否返回 PyTorch tensor

**示例**:
```python
# 如果已经运行过 OpenFace
features = extractor.extract_from_csv('openface_output.csv')
print(features.shape)  # (16, 714)
```

##### `batch_extract(video_paths, return_tensor=True)`

批量提取多个视频的特征。

**示例**:
```python
video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']
features_list = extractor.batch_extract(video_paths)

for i, features in enumerate(features_list):
    print(f"Video {i}: {features.shape}")
```

#### OpenFace 特征说明

714 维特征包含:

1. **2D 面部关键点** (136 维): 68 个点的 (x, y) 坐标
2. **3D 面部关键点** (204 维): 68 个点的 (X, Y, Z) 坐标
3. **头部姿态** (6 维): 平移和旋转
4. **眼睛注视** (6 维): 左右眼注视方向
5. **动作单元强度** (~17 维): AU 激活强度
6. **动作单元存在** (~17 维): AU 是否存在

---

### AudioExtractor

使用 Wav2Vec2 或 MFCC 提取音频特征。

#### 初始化参数

```python
AudioExtractor(
    method='wav2vec2',                      # 'wav2vec2' 或 'mfcc'
    model_name='facebook/wav2vec2-base',    # Wav2Vec2 模型名称
    feature_dim=None,                       # 特征维度，None 表示自动
    sample_rate=16000,                      # 采样率
    device='cpu'                            # 设备
)
```

#### 方法

##### `extract_from_audio(audio_path, return_tensor=True)`

从音频文件中提取特征。

**参数**:
- `audio_path`: 音频文件路径
- `return_tensor`: 是否返回 PyTorch tensor

**返回**:
- `torch.Tensor (feature_dim,)` 或 `np.ndarray (feature_dim,)`

**示例**:
```python
# Wav2Vec2 提取器
extractor_w2v = AudioExtractor(method='wav2vec2', device='cuda')
features = extractor_w2v.extract_from_audio('audio.wav')
print(features.shape)  # (768,)

# MFCC 提取器
extractor_mfcc = AudioExtractor(method='mfcc', feature_dim=40)
features = extractor_mfcc.extract_from_audio('audio.wav')
print(features.shape)  # (40,)
```

##### `extract_from_video(video_path, return_tensor=True, keep_audio=False)`

从视频文件中提取音频特征。

**参数**:
- `video_path`: 视频文件路径
- `return_tensor`: 是否返回 PyTorch tensor
- `keep_audio`: 是否保留提取的音频文件

**示例**:
```python
extractor = AudioExtractor(method='wav2vec2', device='cuda')

# 从视频提取音频特征
features = extractor.extract_from_video('video.mp4')
print(features.shape)  # (768,)
```

##### `batch_extract(paths, from_video=False, return_tensor=True)`

批量提取多个文件的特征。

**示例**:
```python
# 批量提取音频文件
audio_paths = ['audio1.wav', 'audio2.wav', 'audio3.wav']
features_list = extractor.batch_extract(audio_paths, from_video=False)

# 批量提取视频文件
video_paths = ['video1.mp4', 'video2.mp4']
features_list = extractor.batch_extract(video_paths, from_video=True)
```

#### 特征提取方法对比

| 方法 | 特征维度 | 速度 | 效果 | 依赖 |
|-----|---------|------|------|------|
| wav2vec2 | 768 | ⚡⚡ 中 | ⭐⭐⭐⭐ 很好 | transformers |
| mfcc | 40 | ⚡⚡⚡ 快 | ⭐⭐⭐ 好 | librosa |

**推荐**: `wav2vec2` (效果更好)

---

### LieDetectionDataset

整合三种模态的 PyTorch 数据集。

#### 初始化参数

```python
LieDetectionDataset(
    annotation_file,           # 标注文件路径 (CSV)
    video_dir=None,            # 视频目录
    mode='realtime',           # 'realtime' 或 'precomputed'
    face_extractor=None,       # FaceExtractor 实例
    openface_extractor=None,   # OpenFaceExtractor 实例
    audio_extractor=None,      # AudioExtractor 实例
    precomputed_dir=None,      # 预提取特征目录
    transform=None,            # 数据增强
    split=None                 # 数据集划分: 'train', 'val', 'test'
)
```

#### 标注文件格式

CSV 文件，包含以下列:

```csv
video_id,label,split
video_001,0,train
video_002,1,train
video_003,0,val
video_004,1,test
...
```

- `video_id`: 视频文件名 (不含扩展名)
- `label`: 0=真话, 1=谎言
- `split`: train/val/test (可选)

#### 使用模式

##### 模式 1: 实时提取 (realtime)

每次访问数据时实时提取特征。

**优点**: 灵活，不需要预处理
**缺点**: 慢，不适合大规模训练

**示例**:
```python
from dataloader import LieDetectionDataset

dataset = LieDetectionDataset(
    annotation_file='data/annotations.csv',
    video_dir='data/videos',
    mode='realtime',
    split='train'
)

# 获取一个样本
sample = dataset[0]
print(sample['faces'].shape)      # (16, 3, 160, 160)
print(sample['openfaces'].shape)  # (16, 714)
print(sample['audios'].shape)     # (768,)
print(sample['label'])            # 0 或 1
```

##### 模式 2: 预提取 (precomputed)

从预先提取的特征文件加载。

**优点**: 快，适合大规模训练
**缺点**: 需要预处理，占用磁盘空间

**示例**:
```python
dataset = LieDetectionDataset(
    annotation_file='data/annotations.csv',
    mode='precomputed',
    precomputed_dir='data/features',
    split='train'
)

sample = dataset[0]
```

#### 创建 DataLoader

使用便捷函数:

```python
from dataloader import create_dataloader

# 实时提取模式
dataloader = create_dataloader(
    annotation_file='data/annotations.csv',
    video_dir='data/videos',
    mode='realtime',
    batch_size=8,
    shuffle=True,
    num_workers=4,
    split='train'
)

# 预提取模式
dataloader = create_dataloader(
    annotation_file='data/annotations.csv',
    mode='precomputed',
    precomputed_dir='data/features',
    batch_size=32,
    shuffle=True,
    num_workers=8,
    split='train'
)

# 迭代
for batch in dataloader:
    faces = batch['faces']          # (B, T, 3, H, W)
    openfaces = batch['openfaces']  # (B, T, 714)
    audios = batch['audios']        # (B, 768)
    labels = batch['labels']        # (B,)
    video_ids = batch['video_ids']  # List[str]
    
    # 训练...
```

---

## 🔄 数据准备流程

### 完整流程

```
原始数据 (视频 + 标注)
         ↓
    预提取特征 (可选但推荐)
         ↓
    创建 DataLoader
         ↓
    训练模型
```

### 步骤 1: 准备标注文件

创建 `annotations.csv`:

```csv
video_id,label,split
lie_001,1,train
truth_001,0,train
lie_002,1,train
truth_002,0,val
lie_003,1,test
```

### 步骤 2: 预提取特征 (推荐)

使用提供的脚本批量提取:

```bash
python dataloader/precompute_features.py \
    --annotation_file data/annotations.csv \
    --video_dir data/videos \
    --output_dir data/features \
    --face_method opencv \
    --audio_method wav2vec2 \
    --num_frames 16 \
    --device cuda \
    --skip_existing
```

**参数说明**:
- `--annotation_file`: 标注文件路径
- `--video_dir`: 视频目录
- `--output_dir`: 输出目录
- `--face_method`: 人脸检测方法 (opencv/dlib/mtcnn)
- `--openface_path`: OpenFace 可执行文件路径 (可选)
- `--audio_method`: 音频特征方法 (wav2vec2/mfcc)
- `--num_frames`: 提取的帧数
- `--device`: 设备 (cuda/cpu)
- `--skip_existing`: 跳过已存在的特征

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

### 步骤 3: 创建训练脚本

```python
from dataloader import create_dataloader
from plan import FusionModel
import torch

# 创建数据加载器
train_loader = create_dataloader(
    annotation_file='data/annotations.csv',
    mode='precomputed',
    precomputed_dir='data/features',
    batch_size=8,
    shuffle=True,
    num_workers=4,
    split='train'
)

val_loader = create_dataloader(
    annotation_file='data/annotations.csv',
    mode='precomputed',
    precomputed_dir='data/features',
    batch_size=8,
    shuffle=False,
    num_workers=4,
    split='val'
)

# 创建模型
model = FusionModel(device='cuda')
model = model.cuda()

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        faces = batch['faces'].cuda()
        openfaces = batch['openfaces'].cuda()
        audios = batch['audios'].cuda()
        labels = batch['labels'].cuda()
        
        # 前向传播
        output = model(faces, openfaces, audios)
        
        # 计算损失
        loss = criterion(output['logits']['fused'], labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## ❓ 常见问题

### Q1: OpenFace 安装失败怎么办？

**A**: OpenFace 安装较复杂，如果遇到问题:
1. 参考官方文档: https://github.com/TadasBaltrusaitis/OpenFace/wiki
2. 使用预编译版本
3. 或者跳过 OpenFace，只使用 Face 和 Audio 模态

### Q2: 提取特征很慢怎么办？

**A**: 
1. 使用 GPU: `device='cuda'`
2. 使用更快的人脸检测方法: `method='opencv'`
3. 减少帧数: `num_frames=8`
4. 使用预提取模式而不是实时提取

### Q3: 内存不足怎么办？

**A**:
1. 减小 batch_size
2. 减少 num_workers
3. 使用预提取模式
4. 减少提取的帧数

### Q4: 视频中检测不到人脸怎么办？

**A**:
1. 尝试不同的检测方法 (dlib, mtcnn)
2. 检查视频质量
3. 代码会自动使用上一帧的人脸填充

### Q5: FFmpeg 未找到怎么办？

**A**:
1. 安装 FFmpeg: https://ffmpeg.org/download.html
2. 添加到系统 PATH
3. 或指定完整路径

### Q6: 如何只使用部分模态？

**A**: 修改 FusionModel 或创建自定义模型，只使用需要的模态。

### Q7: 如何自定义数据增强？

**A**:
```python
def my_transform(sample):
    # 对 faces 进行增强
    faces = sample['faces']
    # ... 增强操作
    sample['faces'] = faces
    return sample

dataset = LieDetectionDataset(
    ...,
    transform=my_transform
)
```

---

## 📝 总结

数据加载器提供了完整的多模态特征提取和数据加载功能:

1. **FaceExtractor**: 提取面部图像
2. **OpenFaceExtractor**: 提取精细面部特征
3. **AudioExtractor**: 提取音频特征
4. **LieDetectionDataset**: 整合数据集

**推荐工作流程**:
1. 准备标注文件
2. 使用 `precompute_features.py` 预提取特征
3. 使用 `create_dataloader` 创建 DataLoader
4. 开始训练

**下一步**: 查看 `训练指南` 了解如何训练模型。

---

**版本**: v1.0  
**最后更新**: 2025-02-08
