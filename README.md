# 多模态谎言检测系统

基于深度学习的多模态融合谎言检测系统，整合人脸图像、音频和OpenFace特征进行高精度谎言识别。

## 🎯 项目特点

- **多模态融合**: 整合视觉、音频、面部行为三种模态
- **Transformer架构**: 使用Transformer进行时序建模和跨模态交互
- **动态权重学习**: 自适应学习各模态的重要性
- **多任务学习**: 同时优化6个任务，提高泛化能力
- **端到端训练**: 从视频到预测的完整流程

## 📋 目录

- [快速开始](#快速开始)
- [完整流程](#完整流程)
- [项目结构](#项目结构)
- [模型架构](#模型架构)
- [性能指标](#性能指标)
- [文档](#文档)

## 🚀 快速开始

### 环境要求

```bash
# Python 3.8+
# PyTorch 2.0+
# CUDA 11.0+ (可选，用于GPU加速)
```

### 安装依赖

```bash
pip install -r requirements_win.txt
```

### 完整流程（三步走）

```bash
# 步骤1: 数据准备（下载视频、切片、提取特征）
python run_utils.py

# 步骤2: 训练模型
python train.py --num_epochs 10 --batch_size 8 --lr 1e-5

# 步骤3: 模型预测（随机50个视频，计算准确率）
python detect.py
```

### 快速训练

```bash
# 训练模型（需要先运行 python run_utils.py 准备数据）
python train.py --num_epochs 10 --batch_size 8 --lr 1e-5

# 训练完成后会生成:
# - checkpoints/best_model.pth (最佳模型)
# - checkpoints/latest_model.pth (最新模型)
# - checkpoints/training_history.json (训练历史)
```

### 快速推理

```bash
# 方法1: 默认测试（随机抽取50个视频，计算准确率）
python detect.py

# 方法2: 单个视频检测
python detect.py --video path/to/video.mp4

# 方法3: 批量检测（检测目录下所有视频）
python detect.py --video_dir path/to/videos/

# 方法4: 指定检查点和设备
python detect.py --checkpoint checkpoints/best_model.pth --device cuda

# 方法5: Python API
python
>>> from detector import LieDetector
>>> detector = LieDetector.from_checkpoint('checkpoints/best_model.pth')
>>> result = detector.predict('video.mp4')
>>> print(f"{result['label_cn']}: {result['confidence']:.2%}")
```

---

**注意**: 详见下方"完整流程"部分获取更多细节。

## 📖 完整流程

### 步骤1: 数据准备

#### 方法1: 一键运行（推荐）

```bash
# 使用run_utils.py一键完成所有数据准备工作
python run_utils.py
```

**该脚本会自动执行以下步骤：**
1. 从YouTube下载视频并根据时间戳切片
2. 生成视频标签CSV文件
3. 批量提取所有视频的三大特征（人脸、音频、OpenFace）

**输出：**
- `src/videos/full/` - 完整YouTube视频
- `src/videos/cut/` - 切片后的视频片段（~1,500个）
- `src/dataset/video_labels.csv` - 视频标签文件
- `src/features/` - 预提取的特征文件（~4,300个.pt文件）

**预计时间：**
- 下载视频：~30-60分钟（取决于网络速度）
- 切片视频：~10-20分钟
- 提取特征：~60-90分钟（GPU）或 ~120-180分钟（CPU）
- **总计：~2-4小时**

---

#### 方法2: 分步执行（可选）

如果需要更细粒度的控制，可以分步执行：

##### 1.1 下载和切片视频

```bash
# 从YouTube下载视频并根据时间戳切片
python -c "from dataloader.video_downloader import VideoDownloader; \
           downloader = VideoDownloader(); \
           downloader.run()"
```

**输出**:
- `src/videos/full/` - 完整YouTube视频
- `src/videos/cut/` - 切片后的视频片段
- `src/dataset/video_labels.csv` - 视频标签文件

**数据说明**:
- 数据集: DOLOS (Deception Detection Dataset)
- 视频数量: ~1,500个片段
- 标签: truth (说真话) / deception (说谎)

##### 1.2 提取特征

```bash
# 批量提取所有视频的特征
python dataloader/precompute_features.py
```

**提取的特征**:
1. **人脸特征** (1024维)
   - 使用MobileNetV3提取
   - 每个视频提取16帧
   - 保存为 `video_name_faces.pt`

2. **音频特征** (768维)
   - 使用Wav2Vec2提取
   - 整个音频提取一次
   - 保存为 `video_name_audios.pt`

3. **OpenFace特征** (714维)
   - 使用OpenFace工具提取
   - 包含面部动作单元、头部姿态等
   - 保存为 `video_name_openfaces.pt`

**输出目录**: `src/features/`

**预计时间**: 
- CPU: ~90分钟
- GPU: ~54分钟

### 步骤2: 训练模型

#### 2.1 基本训练

```bash
# 使用默认参数训练
python train.py
```

#### 2.2 自定义训练

```bash
# 自定义超参数
python train.py \
    --num_epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --visual_hidden 512 \
    --fusion_hidden 256 \
    --dropout 0.3
```

#### 2.3 从检查点恢复

```bash
# 从上次中断的地方继续训练
python train.py --resume latest_model.pth
```

#### 2.4 完整参数说明

```bash
python train.py \
    --csv_path src/dataset/video_labels.csv \     # 标签文件
    --features_dir src/features \                  # 特征目录
    --use_openface \                               # 使用OpenFace特征
    --num_epochs 30 \                              # 训练轮数
    --batch_size 8 \                               # 批次大小
    --lr 1e-4 \                                    # 学习率
    --weight_decay 1e-4 \                          # 权重衰减
    --patience 5 \                                 # 早停耐心值
    --val_split 0.2 \                              # 验证集比例
    --visual_hidden 256 \                          # 视觉隐藏层维度
    --fusion_hidden 128 \                          # 融合隐藏层维度
    --dropout 0.3 \                                # Dropout比例
    --device cuda \                                # 设备 (cuda/cpu)
    --save_dir checkpoints \                       # 保存目录
    --seed 42                                      # 随机种子
```

**训练策略**:
- **多任务学习**: 同时优化6个任务
  - Face分类、OpenFace分类、Audio分类
  - Face-Audio融合、Face-OpenFace融合
  - 最终融合（主任务）
- **冻结Backbone**: MobileNetV3保持冻结，只训练融合层
- **加权损失**: 融合任务权重最高（1.0），其他任务辅助（0.15-0.2）
- **早停机制**: 防止过拟合，自动保存最佳模型
- **学习率调度**: 验证准确率不提升时自动降低学习率

**输出**:
- `checkpoints/best_model.pth` - 最佳模型（验证准确率最高）
- `checkpoints/latest_model.pth` - 最新模型（每个epoch更新）
- `checkpoints/training_history.json` - 训练历史（损失、准确率曲线）

**预计时间**: 
- 小数据集（<100样本）: 10-20分钟
- 中等数据集（100-1000样本）: 30-60分钟
- 大数据集（>1000样本）: 1-2小时

### 步骤3: 模型预测

#### 3.1 使用detect.py脚本

```bash
# 默认测试（随机抽取50个视频，自动计算准确率）
python detect.py

# 单个视频检测
python detect.py --video path/to/video.mp4

# 批量检测目录下所有视频
python detect.py --video_dir path/to/videos/

# 指定检查点和设备
python detect.py --checkpoint checkpoints/best_model.pth --device cuda
```

#### 3.2 编程接口（推荐）

```python
from detector import LieDetector

# 从检查点加载
detector = LieDetector.from_checkpoint(
    checkpoint_path='checkpoints/best_model.pth',
    device='cuda'
)

# 单个视频预测
result = detector.predict('video.mp4', verbose=True)

print(f"预测: {result['label_cn']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"模态权重: {result['weights']}")

# 批量预测
video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']
results = detector.predict_batch(video_paths, verbose=False)

for i, result in enumerate(results):
    print(f"视频{i+1}: {result['label_cn']} ({result['confidence']:.2%})")
```

**输出格式**:
```python
{
    'prediction': 0,              # 0: truth, 1: deception
    'label': 'truth',             # 英文标签
    'label_cn': '说真话',         # 中文标签
    'confidence': 0.85,           # 置信度
    'probs': {
        'face': [0.6, 0.4],       # 人脸模态概率
        'openface': [0.7, 0.3],   # OpenFace模态概率
        'audio': [0.8, 0.2],      # 音频模态概率
        'fa_au': [0.75, 0.25],    # Face-Audio融合概率
        'fa_of': [0.72, 0.28],    # Face-OpenFace融合概率
        'fused': [0.85, 0.15]     # 最终融合概率
    },
    'weights': {
        'face': 0.4,              # 人脸权重
        'openface': 0.3,          # OpenFace权重
        'audio': 0.3              # 音频权重
    },
    'details': {
        'cosine_similarities': {  # 余弦相似度
            'fa_fa_au': 0.85,
            'fa_fa_of': 0.82,
            'au_fa_au': 0.78,
            'of_fa_of': 0.80
        },
        'video_path': 'video.mp4'
    }
}
```

## 📁 项目结构

```
lie_detector/
├── dataloader/              # 数据加载和特征提取
│   ├── face_extractor.py    # 人脸特征提取
│   ├── audio_extractor.py   # 音频特征提取
│   ├── openface_extractor.py # OpenFace特征提取
│   ├── dataset.py           # 数据集类
│   ├── video_downloader.py  # 视频下载和切片
│   └── precompute_features.py # 批量特征提取
│
├── models/                  # 模型定义
│   ├── faces.py             # 人脸模型 (MobileNetV3)
│   ├── audio.py             # 音频模型 (Wav2Vec2)
│   ├── openface.py          # OpenFace模型
│   └── fusion.py            # 融合模型 ⭐
│
├── trainer/                 # 训练器
│   ├── trainer.py           # 训练逻辑
│   ├── dataset.py           # 训练数据集
│   └── README.md            # 训练器文档
│
├── detector/                # 检测器
│   ├── detector.py          # 检测器类
│   └── README.md            # 检测器文档
│
├── docs/                    # 文档
│   ├── 模型总体架构.md       # 架构说明
│   ├── 数据流.md            # 数据流和张量形状
│   ├── 模型API速查表.md     # API参考
│   └── 训练方案.md          # 训练策略
│
├── src/                     # 数据目录
│   ├── videos/              # 视频文件
│   │   ├── full/            # 完整视频
│   │   └── cut/             # 切片视频
│   ├── features/            # 预提取特征
│   └── dataset/             # 标签文件
│
├── checkpoints/             # 模型检查点
│   ├── best_model.pth       # 最佳模型
│   ├── latest_model.pth     # 最新模型
│   └── training_history.json # 训练历史
│
├── run_utils.py             # 数据准备入口 ⭐
├── train.py                 # 训练脚本 ⭐
├── detect.py                # 检测脚本 ⭐
├── test_fusion_model.py     # 模型测试
└── README.md                # 本文件
```

## 🏗️ 模型架构

### 整体架构

```
输入视频
    ↓
┌─────────────────────────────────────────┐
│          特征提取层 (预训练)              │
├─────────────────────────────────────────┤
│  MobileNetV3  │  Wav2Vec2  │  OpenFace  │
│   (1024维)    │   (768维)   │  (714维)   │
└─────────────────────────────────────────┘
    ↓           ↓           ↓
┌─────────────────────────────────────────┐
│           融合模型 (FusionModel)         │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  单模态编码器 (Transformer)      │   │
│  │  Face → 256维                   │   │
│  │  OpenFace → 256维               │   │
│  │  Audio → 256维                  │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  跨模态注意力 (Cross-Attention)  │   │
│  │  Face ↔ Audio                   │   │
│  │  Face ↔ OpenFace                │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  动态权重学习 (W Module)         │   │
│  │  学习各模态重要性                │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  多任务分类                      │   │
│  │  6个任务同时优化                 │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
    ↓
预测结果 (Truth/Deception)
```

### 关键特性

1. **保留高维特征**: 使用256维表示而非2维logits
2. **时序建模**: Transformer处理视频帧序列
3. **跨模态交互**: Cross-Attention学习模态间关系
4. **动态权重**: 自适应学习各模态重要性
5. **多任务学习**: 6个任务共享表示，提高泛化

### 参数统计

| 组件 | 参数量 | 是否训练 |
|------|--------|---------|
| MobileNetV3 Backbone | ~1.5M | ❌ 冻结 |
| 融合层 + 注意力 | ~4.7M | ✅ 训练 |
| **总计** | **~6.2M** | **~4.7M可训练** |

## 📊 性能指标

### 特征提取速度

| 模态 | CPU | GPU |
|------|-----|-----|
| 人脸 | ~0.4s | ~0.2s |
| 音频 | ~1.5s | ~0.8s |
| OpenFace | ~1.5s | ~1.0s |
| **总计** | **~3.5s** | **~2.0s** |

### 训练速度

| 数据量 | 时间 (GPU) |
|--------|-----------|
| 100样本 | ~10分钟 |
| 500样本 | ~30分钟 |
| 1500样本 | ~60分钟 |

### 推理速度

| 模式 | 速度 |
|------|------|
| 单视频（含特征提取） | ~5秒 |
| 批量（特征已提取） | ~0.1秒/视频 |

## 📚 文档

### 核心文档

- [模型总体架构](docs/模型总体架构.md) - 详细的架构说明
- [数据流](docs/数据流.md) - 每一层的张量形状变化
- [模型API速查表](docs/模型API速查表.md) - API参考手册
- [训练方案](docs/训练方案.md) - 完整训练策略
- [训练示例](docs/训练示例.md) - 不同场景的训练示例 ⭐
- [时序建模详解](docs/时序建模详解.md) - 时序建模机制详解 🆕
- [时序建模可视化](docs/时序建模可视化.md) - 时序建模可视化说明 🆕

### 实验工具

- [实验工具文档](docs/实验-实验工具.md) - 实验数据可视化工具 🆕

### 模块文档

- [检测器使用指南](detector/README.md) - 检测器详细说明
- [训练器使用指南](trainer/README.md) - 训练器详细说明（如果存在）

### 训练相关

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 快速训练指南
- [TRAINING_SUCCESS.md](TRAINING_SUCCESS.md) - 训练成功确认 ✅

## 🔧 配置说明

### 训练配置

```python
# 基础配置（小数据集）
{
    'batch_size': 4,
    'num_epochs': 20,
    'lr': 5e-5,
    'dropout': 0.5,
    'visual_hidden': 128
}

# 标准配置（中等数据集）
{
    'batch_size': 8,
    'num_epochs': 30,
    'lr': 1e-4,
    'dropout': 0.3,
    'visual_hidden': 256
}

# 大规模配置（大数据集）
{
    'batch_size': 16,
    'num_epochs': 50,
    'lr': 1e-4,
    'dropout': 0.2,
    'visual_hidden': 512
}
```

### 损失权重配置

```python
loss_weights = {
    'face': 0.2,        # 人脸分类
    'openface': 0.2,    # OpenFace分类
    'audio': 0.2,       # 音频分类
    'fa_au': 0.15,      # Face-Audio融合
    'fa_of': 0.15,      # Face-OpenFace融合
    'fused': 1.0        # 最终融合（主任务）
}
```

## 🛠️ 系统要求

### 最低配置

- CPU: 4核
- RAM: 8GB
- GPU: 可选
- 存储: 50GB

### 推荐配置

- CPU: 8核+
- RAM: 16GB+
- GPU: NVIDIA GPU (8GB+ VRAM)
- 存储: 100GB SSD

## 🐛 故障排除

### CUDA out of memory

```bash
# 减小batch_size
python train.py --batch_size 4

# 或使用CPU
python train.py --device cpu
```

### OpenFace不可用

```bash
# 系统会自动跳过OpenFace特征
# 只使用人脸和音频特征
```

### 训练不收敛

```bash
# 降低学习率
python train.py --lr 5e-5

# 增加Dropout
python train.py --dropout 0.5
```

## 📈 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.x |
| 预训练模型 | MobileNetV3, Wav2Vec2 |
| 人脸检测 | OpenCV |
| 面部特征 | OpenFace 2.0 |
| 音频处理 | librosa, transformers |
| 视频处理 | ffmpeg, yt-dlp |

## 🎓 引用

如果使用本项目，请引用：

```bibtex
@misc{multimodal_lie_detection,
  title={Multi-Modal Lie Detection System},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/lie-detector}
}
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**最后更新**: 2026-02-09

**项目状态**: 🟢 开发完成，可用于训练和推理

**版本**: v1.0.0
