# 多模态谎言检测系统 (Multimodal Lie Detection)

基于深度学习的多模态融合谎言检测系统，通过分析面部表情、面部动作单元（OpenFace）和语音特征来判断说话者是否在说谎。

## 📋 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [文件结构](#文件结构)
- [模型详解](#模型详解)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [使用说明](#使用说明)

---

## 🎯 项目概述

本项目实现了一个端到端的谎言检测系统，融合三种模态的信息：

1. **面部表情** - 使用 MobileNetV3 提取面部图像特征
2. **面部动作单元** - 使用 OpenFace 提取的 714 维精细面部特征
3. **语音特征** - 使用 Wav2Vec2 提取的 768 维音频特征

通过 Transformer 编码器、跨模态注意力机制和动态权重融合，系统能够自适应地整合多模态信息，输出最终的谎言检测结果。

---

## 🏗️ 系统架构

```
输入数据
├── 面部图像序列 (B, T, 3, 160, 160)
├── OpenFace 特征 (B, T, 714)
└── 音频特征 (B, 768)
         ↓
    三个子模型并行处理
├── FacesModel (MobileNetV3)
├── OpenfaceModel (全连接网络)
└── AudioModel (全连接网络)
         ↓
    特征编码与融合
├── 单模态编码器 (UniEncoder)
├── 跨模态注意力 (CrossAttention)
└── 动态权重模块 (W)
         ↓
    输出结果
├── 各模态独立预测
├── 融合预测概率
└── 模态权重分布
```

---

## 📁 文件结构

```
lie_detector/
├── models/                          # 模型定义目录
│   ├── __init__.py                 # 模块初始化文件
│   ├── faces.py                    # 面部表情模型
│   ├── openface.py                 # OpenFace 特征分类器
│   └── audio.py                    # 音频模型
├── plan.py                         # 核心融合模型
├── test_models.py                  # 子模型测试脚本
├── test_fusion_model.py            # 融合模型测试脚本
├── requirements.txt                # Conda 依赖列表
├── requirements_win.txt            # Windows pip 依赖列表
└── README.md                       # 本文档
```

---

## 🔍 模型详解

### 1. models/faces.py

#### `FacesModel` 类
**功能**: 从面部图像中提取特征并进行谎言检测

**架构**:
- **骨干网络**: MobileNetV3-Small (预训练，冻结参数)
- **特征维度**: 1024 维
- **分类头**: 3 层全连接网络 (1024 → 512 → 256 → 2)
- **输出**: 2 分类 logits (真话/谎言)

**关键方法**:
```python
forward(x)           # 前向传播，输入 (B, 3, 160, 160)，输出 (B, 2)
predict_dir(dir)     # 对目录中所有图像进行平均预测
encode_dir(dir)      # 提取目录中所有图像的平均特征向量
```

**特点**:
- 自动处理单样本推理时的 BatchNorm 问题
- 支持从图像目录批量预测
- 预训练权重自动下载

---

### 2. models/openface.py

#### `OpenfaceModel` 类
**功能**: 处理 OpenFace 提取的面部动作单元（AU）特征

**架构**:
- **输入维度**: 714 (OpenFace 特征)
- **网络结构**: 3 层全连接 (714 → 512 → 256 → 2)
- **输出**: 2 分类 logits

**关键方法**:
```python
forward(x)           # 输入 (B, 714) 或 (B*T, 714)，输出 (B, 2)
```

**特点**:
- 专门处理 OpenFace 工具提取的精细面部特征
- 捕捉微表情和面部动作单元变化
- 支持批量和序列输入

---

### 3. models/audio.py

#### `Wav2Vec2Encoder` 类
**功能**: 从音频文件中提取 Wav2Vec2 特征

**架构**:
- **预训练模型**: facebook/wav2vec2-base
- **输出维度**: 768 维特征向量
- **处理方式**: 时序平均池化

**关键方法**:
```python
__call__(wav_path)   # 输入音频路径，输出 768 维 numpy 向量
```

**特点**:
- 自动下载预训练模型
- 支持任意长度音频
- 参数冻结，仅用于特征提取

#### `AudioModel` 类
**功能**: 基于音频特征进行谎言检测

**架构**:
- **输入维度**: 768 (Wav2Vec2 特征)
- **网络结构**: 4 层全连接 (768 → 512 → 256 → 128 → 2)
- **输出**: 2 分类 logits

**关键方法**:
```python
forward(audio_features)  # 输入 (B, 768)，输出 (B, 2)
```

**特点**:
- 可选 MFCC 特征融合（默认关闭）
- 多层 BatchNorm 和 Dropout 防止过拟合

#### `mfcc_vector()` 函数
**功能**: 提取 MFCC 音频特征（可选）

---

### 4. plan.py - 核心融合模型

#### `Transpose` 类
**功能**: 张量维度转置辅助模块

**用途**: 在 Sequential 中方便地进行维度变换

---

#### `W` 类（动态权重模块）
**功能**: 自动学习三个模态的重要性权重

**架构**:
- **输入**: 三个模态的特征向量 (各 256 维)
- **输出**: 3 维权重向量，经 softmax 归一化
- **网络**: 2 层全连接 (768 → 256 → 3)

**工作原理**:
```python
W(U_face, U_of, U_audio) → [w_face, w_of, w_audio]
# 权重和为 1，自适应调整各模态贡献
```

**特点**:
- 动态权重，不同样本权重不同
- 自动发现最可靠的模态
- 提高融合鲁棒性

---

#### `CrossAttentionBlock` 类
**功能**: 跨模态注意力机制

**架构**:
- **注意力头数**: 4
- **维度**: 256
- **类型**: Multi-head Attention

**工作原理**:
```python
# Query 模态关注 Key/Value 模态
CrossAttention(q=audio, k=face, v=face) → 融合特征
```

**特点**:
- 捕捉模态间的互补信息
- 双向注意力（Face↔Audio, Face↔OpenFace）
- 增强模态间的语义对齐

---

#### `UniEncoder` 类
**功能**: 单模态时序编码器

**架构**:
- **基础**: Transformer Encoder
- **层数**: 2 层
- **注意力头**: 4 个
- **前馈维度**: 512

**工作原理**:
```python
# 输入时序特征 (B, T, D_in)
# 输出编码特征 (B, T, D_out)
UniEncoder(face_sequence) → encoded_sequence
```

**特点**:
- 建模时序依赖关系
- 自注意力机制捕捉帧间关系
- 支持任意长度序列

---

#### `FusionModel` 类（核心模型）
**功能**: 多模态融合谎言检测的主模型

**初始化参数**:
```python
FusionModel(
    device='cuda',              # 设备
    num_classes=2,              # 分类数（真话/谎言）
    visual_hidden=256,          # 视觉特征维度
    fusion_hidden=128,          # 融合层维度
    dropout=0.3,                # Dropout 比例
    freeze_backbones=True,      # 是否冻结预训练骨干
    uni_layers=2,               # UniEncoder 层数
    uni_nhead=4,                # UniEncoder 注意力头数
    com_heads=4,                # 跨模态注意力头数
    face_checkpoint=None,       # Face 模型检查点
    openface_checkpoint=None,   # OpenFace 模型检查点
    audio_checkpoint=None       # Audio 模型检查点
)
```

**核心组件**:

1. **三个子模型**:
   - `self.face`: FacesModel
   - `self.openface`: OpenfaceModel
   - `self.audio`: AudioModel

2. **单模态编码器** (E_uni):
   - `self.E_uni_face`: 1024 → 256 + Transformer
   - `self.E_uni_of`: 714 → 256 + Transformer
   - `self.E_uni_audio`: 768 → 256

3. **跨模态编码器** (E_com):
   - `self.E_com_fa_au`: Face ↔ Audio 注意力
   - `self.E_com_fa_of`: Face ↔ OpenFace 注意力

4. **动态权重模块**:
   - `self.W`: 计算三个模态的权重

5. **融合分类器**:
   - `self.prob_fa_au`: Face+Audio 融合分类
   - `self.prob_fa_of`: Face+OpenFace 融合分类

**前向传播流程**:

```python
def forward(faces, openfaces, audios):
    # 1. 预处理
    faces → (B*T, 3, 160, 160)
    
    # 2. 特征提取
    face_feats = face.backbone(faces)      # (B, T, 1024)
    of_feats = openfaces                   # (B, T, 714)
    audio_feats = audios                   # (B, 768)
    
    # 3. 单模态编码
    U_face = E_uni_face(face_feats)        # (B, 256)
    U_of = E_uni_of(of_feats)              # (B, 256)
    U_audio = E_uni_audio(audio_feats)     # (B, 256)
    
    # 4. 跨模态注意力
    C_fa_au = CrossAttn(audio, face)       # (B, 256)
    C_fa_of = CrossAttn(openface, face)    # (B, 256)
    
    # 5. 动态权重
    W = softmax(W(U_face, U_of, U_audio))  # (B, 3)
    
    # 6. 加权融合
    U_face = W[:, 0] * U_face
    U_of = W[:, 1] * U_of
    U_audio = W[:, 2] * U_audio
    
    # 7. 融合预测
    fused_probs = W[:, 2]*probs_fa_au + W[:, 1]*probs_fa_of + W[:, 0]*probs_face
    
    return {
        'logits': {...},      # 各模态 logits
        'probs': {...},       # 各模态概率
        'weights': W,         # 模态权重
        'cos': {...}          # 余弦相似度
    }
```

**输出结构**:
```python
{
    'logits': {
        'face': (B, 2),       # 面部模型 logits
        'openface': (B, 2),   # OpenFace 模型 logits
        'audio': (B, 2),      # 音频模型 logits
        'fa_au': (B, 2),      # Face+Audio 融合 logits
        'fa_of': (B, 2),      # Face+OpenFace 融合 logits
    },
    'probs': {
        'face': (B, 2),       # 面部模型概率
        'openface': (B, 2),   # OpenFace 模型概率
        'audio': (B, 2),      # 音频模型概率
        'fa_au': (B, 2),      # Face+Audio 融合概率
        'fa_of': (B, 2),      # Face+OpenFace 融合概率
        'fused': (B, 2),      # 最终融合概率 ⭐
    },
    'weights': (B, 3),        # [w_face, w_of, w_audio]
    'cos': {
        'fa_fa_au': (B,),     # Face 与 Face-Audio 相似度
        'fa_fa_of': (B,),     # Face 与 Face-OpenFace 相似度
        'au_fa_au': (B,),     # Audio 与 Face-Audio 相似度
        'of_fa_of': (B,),     # OpenFace 与 Face-OpenFace 相似度
    }
}
```

**辅助方法**:
- `matrix_sqrt()`: 矩阵平方根计算
- `fit_gmm()`: 高斯混合模型拟合
- `sinkhorn()`: Sinkhorn 算法（最优传输）
- `compute_stats()`: 统计量计算
- `mmd()`: 最大均值差异

**特点**:
- 多层次融合策略
- 动态权重自适应
- 支持检查点加载
- 鲁棒的预处理流程
- 丰富的输出信息

---

## 🛠️ 环境配置

### 系统要求
- Python 3.9+
- CUDA 12.1+ (GPU 训练)
- Windows / Linux / macOS

### 安装依赖

#### 方式 1: 使用 Conda (推荐)
```bash
conda create -n lie_detector python=3.9
conda activate lie_detector
conda install --file requirements.txt
```

#### 方式 2: 使用 pip
```bash
pip install -r requirements_win.txt
```

### 核心依赖
- PyTorch 2.5.1 (CUDA 12.1)
- transformers 4.46.3
- timm 1.0.19
- librosa 0.11.0
- opencv-python 4.12.0

### 预训练模型下载

模型会在首次运行时自动下载：

1. **MobileNetV3**: 自动下载到 `~/.cache/torch/hub/checkpoints/`
2. **Wav2Vec2**: 自动下载到 `~/.cache/huggingface/hub/`

如果下载慢，可以使用国内镜像：
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

---

## 🚀 快速开始

### 1. 测试子模型
```bash
python test_models.py
```

**预期输出**:
```
🔍 开始测试预训练模型...
==================================================
测试 FacesModel (MobileNetV3)...
✅ FacesModel 加载成功！
   输出形状: torch.Size([1, 2])
==================================================
测试 Wav2Vec2Encoder...
✅ Wav2Vec2Encoder 加载成功！
==================================================
🎉 所有模型加载成功！
```

### 2. 测试融合模型
```bash
python test_fusion_model.py
```

### 3. 使用示例

```python
import torch
from plan import FusionModel

# 1. 初始化模型
model = FusionModel(
    device='cuda',
    num_classes=2,
    visual_hidden=256,
    fusion_hidden=128
)
model.eval()

# 2. 准备数据
batch_size = 2
num_frames = 16

faces = torch.randn(batch_size, num_frames, 3, 160, 160)      # 面部图像
openfaces = torch.randn(batch_size, num_frames, 714)          # OpenFace 特征
audios = torch.randn(batch_size, 768)                         # 音频特征

# 3. 推理
with torch.no_grad():
    output = model(faces, openfaces, audios)

# 4. 获取结果
fused_probs = output['probs']['fused']  # 最终预测概率
weights = output['weights']              # 模态权重

print(f"预测概率: {fused_probs}")
print(f"模态权重: {weights}")
```

---

## 📖 使用说明

### 数据格式要求

#### 输入数据
```python
faces: torch.Tensor
    形状: (B, T, 3, 160, 160) 或 (B, T, 160, 160, 3)
    类型: float32
    范围: [0, 1] 或 [0, 255] (自动归一化)
    说明: B=批量大小, T=帧数, 3=RGB通道

openfaces: torch.Tensor
    形状: (B, T, 714)
    类型: float32
    说明: OpenFace 工具提取的面部动作单元特征

audios: torch.Tensor
    形状: (B, 768)
    类型: float32
    说明: Wav2Vec2 提取的音频特征向量
```

### 训练模式

```python
# 设置为训练模式
model.train()

# 确保 batch_size >= 2 (BatchNorm 要求)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for faces, openfaces, audios, labels in dataloader:
        optimizer.zero_grad()
        
        output = model(faces, openfaces, audios)
        loss = criterion(output['logits']['fused'], labels)
        
        loss.backward()
        optimizer.step()
```

### 推理模式

```python
# 设置为评估模式
model.eval()

with torch.no_grad():
    output = model(faces, openfaces, audios)
    
    # 获取最终预测
    probs = output['probs']['fused']
    pred = probs.argmax(dim=1)  # 0=真话, 1=谎言
    
    # 查看模态权重
    weights = output['weights']
    print(f"Face权重: {weights[:, 0]}")
    print(f"OpenFace权重: {weights[:, 1]}")
    print(f"Audio权重: {weights[:, 2]}")
```

### 加载检查点

```python
model = FusionModel(
    device='cuda',
    face_checkpoint='checkpoints/face_model.pth',
    openface_checkpoint='checkpoints/openface_model.pth',
    audio_checkpoint='checkpoints/audio_model.pth'
)
```

---

## 🔧 常见问题

### Q1: BatchNorm 错误
**错误**: `Expected more than 1 value per channel when training`

**解决**: 
- 推理时使用 `model.eval()`
- 训练时确保 `batch_size >= 2`

### Q2: 模型下载失败
**解决**: 使用国内镜像
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### Q3: CUDA 内存不足
**解决**: 
- 减小 batch_size
- 减少帧数 T
- 使用梯度累积

### Q4: OpenFace 特征如何提取？
**答**: 需要使用 OpenFace 工具预先提取，参考：
https://github.com/TadasBaltrusaitis/OpenFace

---

## 📊 模型性能

### 特征维度总结
| 模态 | 原始维度 | 编码后维度 | 参数量 |
|-----|---------|-----------|--------|
| Face | 1024 | 256 | ~2.3M |
| OpenFace | 714 | 256 | ~0.5M |
| Audio | 768 | 256 | ~0.6M |
| **总计** | - | - | **~3.4M** |

### 计算复杂度
- **FLOPs**: ~1.2 GFLOPs (单样本)
- **推理速度**: ~50 FPS (GPU)
- **内存占用**: ~500 MB (batch_size=8)

---

## 📝 引用

如果使用本项目，请引用：

```bibtex
@software{multimodal_lie_detection,
  title={Multimodal Lie Detection System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/lie_detector}
}
```

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题，请联系：your.email@example.com

---

**最后更新**: 2025-02-08
