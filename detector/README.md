# 谎言检测器使用指南

## 目录结构

```
detector/
├── __init__.py          # 模块初始化
├── detector.py          # 检测器核心类
└── README.md           # 使用指南
```

## 快速开始

### 1. 使用未训练模型测试推理流程

```bash
# 自动选择测试视频
python detect.py

# 指定视频文件
python detect.py --video path/to/video.mp4

# 使用CPU
python detect.py --device cpu

# 批量检测（前5个视频）
python detect.py --batch
```

### 2. 使用训练好的模型

```bash
# 从检查点加载模型
python detect.py --checkpoint checkpoints/best_model.pth --video path/to/video.mp4
```

## 编程接口

### 基本使用

```python
from detector import LieDetector
from models.fusion import FusionModel
from dataloader import FaceExtractor, AudioExtractor, OpenFaceExtractor

# 1. 创建模型
model = FusionModel(device='cuda')

# 2. 创建特征提取器
face_extractor = FaceExtractor(method='opencv', target_size=(160, 160), num_frames=16)
audio_extractor = AudioExtractor(method='wav2vec2', device='cuda')
openface_extractor = OpenFaceExtractor(num_frames=16)

# 3. 创建检测器
detector = LieDetector(
    model=model,
    face_extractor=face_extractor,
    audio_extractor=audio_extractor,
    openface_extractor=openface_extractor,
    device='cuda'
)

# 4. 检测视频
result = detector.predict('path/to/video.mp4', verbose=True)

print(f"预测: {result['label_cn']}")
print(f"置信度: {result['confidence']:.2%}")
```

### 从检查点加载

```python
from detector import LieDetector

# 直接从检查点创建检测器
detector = LieDetector.from_checkpoint(
    checkpoint_path='checkpoints/best_model.pth',
    device='cuda'
)

# 检测
result = detector.predict('video.mp4')
```

### 批量检测

```python
from pathlib import Path

# 获取所有视频
video_files = list(Path('videos').glob('*.mp4'))

# 批量检测
results = detector.predict_batch(video_files, verbose=False)

# 统计结果
for video, result in zip(video_files, results):
    print(f"{video.name}: {result['label_cn']} ({result['confidence']:.2%})")
```

## 输出格式

`predict()` 方法返回一个字典，包含以下信息：

```python
{
    'prediction': 0,              # 预测类别 (0: truth, 1: deception)
    'label': 'truth',             # 英文标签
    'label_cn': '说真话',         # 中文标签
    'confidence': 0.85,           # 置信度
    'probs': {                    # 各模态概率
        'face': [0.6, 0.4],
        'openface': [0.7, 0.3],
        'audio': [0.8, 0.2],
        'fa_au': [0.75, 0.25],
        'fa_of': [0.65, 0.35],
        'fused': [0.85, 0.15]     # 融合概率
    },
    'weights': {                  # 模态权重
        'face': 0.4,
        'openface': 0.3,
        'audio': 0.3
    },
    'details': {                  # 详细信息
        'cosine_similarities': {
            'fa_fa_au': 0.8,
            'fa_fa_of': 0.7,
            'au_fa_au': 0.6,
            'of_fa_of': 0.5
        },
        'video_path': 'path/to/video.mp4'
    }
}
```

## 特征提取

检测器会自动从视频中提取三种特征：

1. **人脸特征** (1024维)
   - 使用 MobileNetV3 提取
   - 每帧提取一次，最多16帧

2. **音频特征** (768维)
   - 使用 Wav2Vec2 提取
   - 整个音频提取一次

3. **OpenFace特征** (714维，可选)
   - 使用 OpenFace 工具提取
   - 包含面部动作单元、头部姿态等

## 注意事项

1. **未训练模型**：如果使用未训练的模型，预测结果是随机的，仅用于测试数据流。

2. **OpenFace依赖**：如果没有安装OpenFace，检测器会自动跳过OpenFace特征，只使用人脸和音频。

3. **设备选择**：
   - 推荐使用GPU (`device='cuda'`)
   - CPU模式会慢很多，但可以运行

4. **视频格式**：支持常见视频格式（mp4, avi, mov等）

## 性能优化

1. **批量处理**：使用 `predict_batch()` 可以提高效率
2. **特征缓存**：可以预先提取特征并保存，避免重复提取
3. **GPU加速**：使用CUDA可以显著提升速度

## 故障排除

### 问题1：OpenFace不可用
```
解决方案：检测器会自动跳过OpenFace，只使用人脸和音频特征
```

### 问题2：CUDA out of memory
```
解决方案：
1. 减小batch_size
2. 使用CPU模式
3. 减少num_frames
```

### 问题3：视频无法读取
```
解决方案：
1. 检查视频格式是否支持
2. 确保视频文件完整
3. 尝试重新编码视频
```
