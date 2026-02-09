# 模型API速查表

## 快速索引

- [FusionModel](#fusionmodel) - 主融合模型
- [FacesModel](#facesmodel) - 人脸特征提取
- [AudioModel](#audiomodel) - 音频特征提取
- [OpenfaceModel](#openfacemodel) - OpenFace特征提取
- [LieDetector](#liedetector) - 检测器
- [FusionModelTrainer](#fusionmodeltrainer) - 训练器
- [LieDetectionDataset](#liedetectiondataset) - 数据集

---

## FusionModel

**路径**: `models/fusion.py`

### 初始化

```python
from models.fusion import FusionModel

model = FusionModel(
    device='cuda',                    # 设备
    face_checkpoint=None,             # 人脸模型检查点
    openface_checkpoint=None,         # OpenFace模型检查点
    audio_checkpoint=None,            # 音频模型检查点
    weights=[1.0, 1.0, 1.0],         # 初始权重
    freeze_backbones=True,            # 是否冻结backbone
    num_classes=2,                    # 分类数
    visual_hidden=256,                # 视觉隐藏层维度
    fusion_hidden=128,                # 融合隐藏层维度
    dropout=0.3,                      # Dropout比例
    uni_layers=2,                     # 单模态Transformer层数
    uni_nhead=4,                      # 单模态注意力头数
    com_heads=4                       # 跨模态注意力头数
)
```

### 前向传播

```python
output = model(
    faces,      # (B, T, 3, 160, 160) 或 (B, T, 160, 160, 3)
    openfaces,  # (B, T, 714)
    audios      # (B, 768)
)
```

### 输出格式

```python
{
    'logits': {
        'face': torch.Tensor,      # (B, 2)
        'openface': torch.Tensor,  # (B, 2)
        'audio': torch.Tensor,     # (B, 2)
        'fa_au': torch.Tensor,     # (B, 2)
        'fa_of': torch.Tensor,     # (B, 2)
    },
    'probs': {
        'face': torch.Tensor,      # (B, 2)
        'openface': torch.Tensor,  # (B, 2)
        'audio': torch.Tensor,     # (B, 2)
        'fa_au': torch.Tensor,     # (B, 2)
        'fa_of': torch.Tensor,     # (B, 2)
        'fused': torch.Tensor      # (B, 2) ⭐ 最终预测
    },
    'weights': torch.Tensor,       # (B, 3) [w_face, w_of, w_audio]
    'cos': {
        'fa_fa_au': torch.Tensor,  # (B,)
        'fa_fa_of': torch.Tensor,  # (B,)
        'au_fa_au': torch.Tensor,  # (B,)
        'of_fa_of': torch.Tensor   # (B,)
    }
}
```

### 示例

```python
import torch
from models.fusion import FusionModel

# 创建模型
model = FusionModel(device='cuda')

# 准备数据
faces = torch.randn(2, 16, 3, 160, 160).cuda()
openfaces = torch.randn(2, 16, 714).cuda()
audios = torch.randn(2, 768).cuda()

# 推理
model.eval()
with torch.no_grad():
    output = model(faces, openfaces, audios)

# 获取预测
fused_probs = output['probs']['fused']  # (2, 2)
predictions = torch.argmax(fused_probs, dim=1)  # (2,)
```

---

## FacesModel

**路径**: `models/faces.py`

### 初始化

```python
from models.faces import FacesModel

model = FacesModel(
    device='cpu',                           # 设备
    model_name='mobilenetv3_small_100'     # 骨干网络名称
)
```

### 前向传播

```python
# 返回logits (默认)
logits = model(x)  # x: (B, 3, 160, 160) → logits: (B, 2)

# 返回特征
features = model(x, return_features=True)  # → (B, 1024)

# 便捷方法
features = model.get_features(x)  # → (B, 1024)
```

### 示例

```python
import torch
from models.faces import FacesModel

model = FacesModel(device='cuda')
x = torch.randn(4, 3, 160, 160).cuda()

# 获取logits
logits = model(x)  # (4, 2)

# 获取特征
features = model.get_features(x)  # (4, 1024)
```

---

## AudioModel

**路径**: `models/audio.py`

### 初始化

```python
from models.audio import AudioModel

model = AudioModel(
    use_mfcc=False,  # 是否使用MFCC特征
    drop=0.2         # Dropout比例
)
```

### 前向传播

```python
# 输入已经是Wav2Vec2特征
audio_features = torch.randn(B, 768)

# 返回logits (默认)
logits = model(audio_features)  # → (B, 2)

# 返回特征 (直接返回输入)
features = model(audio_features, return_features=True)  # → (B, 768)

# 便捷方法
features = model.get_features(audio_features)  # → (B, 768)
```

---

## OpenfaceModel

**路径**: `models/openface.py`

### 初始化

```python
from models.openface import OpenfaceModel

model = OpenfaceModel(
    input_dim=714,    # 输入维度
    num_classes=2,    # 分类数
    dropout=0.3       # Dropout比例
)
```

### 前向传播

```python
# 输入已经是OpenFace特征
openface_features = torch.randn(B, 714)

# 返回logits (默认)
logits = model(openface_features)  # → (B, 2)

# 返回特征 (直接返回输入)
features = model(openface_features, return_features=True)  # → (B, 714)

# 便捷方法
features = model.get_features(openface_features)  # → (B, 714)
```

---

## LieDetector

**路径**: `detector/detector.py`

### 初始化

```python
from detector import LieDetector
from models.fusion import FusionModel
from dataloader import FaceExtractor, AudioExtractor, OpenFaceExtractor

# 创建模型
model = FusionModel(device='cuda')

# 创建特征提取器
face_extractor = FaceExtractor(method='opencv', target_size=(160, 160), num_frames=16)
audio_extractor = AudioExtractor(method='wav2vec2', device='cuda')
openface_extractor = OpenFaceExtractor(num_frames=16)

# 创建检测器
detector = LieDetector(
    model=model,
    face_extractor=face_extractor,
    audio_extractor=audio_extractor,
    openface_extractor=openface_extractor,
    device='cuda'
)
```

### 从检查点加载

```python
detector = LieDetector.from_checkpoint(
    checkpoint_path='checkpoints/best_model.pth',
    device='cuda'
)
```

### 预测

```python
# 单个视频
result = detector.predict('video.mp4', verbose=True)

# 批量预测
results = detector.predict_batch(['video1.mp4', 'video2.mp4'], verbose=False)
```

### 输出格式

```python
{
    'prediction': 0,              # 0: truth, 1: deception
    'label': 'truth',             # 英文标签
    'label_cn': '说真话',         # 中文标签
    'confidence': 0.85,           # 置信度
    'probs': {
        'face': [0.6, 0.4],
        'openface': [0.7, 0.3],
        'audio': [0.8, 0.2],
        'fa_au': [0.75, 0.25],
        'fa_of': [0.65, 0.35],
        'fused': [0.85, 0.15]     # 最终概率
    },
    'weights': {
        'face': 0.4,
        'openface': 0.3,
        'audio': 0.3
    },
    'details': {
        'cosine_similarities': {...},
        'video_path': 'video.mp4'
    }
}
```

---

## FusionModelTrainer

**路径**: `trainer/trainer.py`

### 初始化

```python
from trainer import FusionModelTrainer
from models.fusion import FusionModel
from torch.utils.data import DataLoader

model = FusionModel(device='cuda')

trainer = FusionModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    save_dir='checkpoints',
    loss_weights={
        'face': 0.2,
        'openface': 0.2,
        'audio': 0.2,
        'fa_au': 0.15,
        'fa_of': 0.15,
        'fused': 1.0
    }
)
```

### 训练

```python
trainer.train(
    num_epochs=30,
    lr=1e-4,
    weight_decay=1e-4,
    patience=5,
    verbose=True
)
```

### 保存/加载检查点

```python
# 保存
trainer.save_checkpoint('my_model.pth', epoch=10)

# 加载
trainer.load_checkpoint('my_model.pth')
```

---

## LieDetectionDataset

**路径**: `trainer/dataset.py`

### 初始化

```python
from trainer import LieDetectionDataset

dataset = LieDetectionDataset(
    csv_path='src/dataset/video_labels.csv',
    features_dir='src/features',
    use_openface=True,
    transform=None
)
```

### 数据格式

```python
# __getitem__ 返回
{
    'faces': torch.Tensor,      # (T, 3, 160, 160)
    'audios': torch.Tensor,     # (768,)
    'openfaces': torch.Tensor,  # (T, 714) 或 None
    'label': torch.Tensor,      # (,) 标量
    'video_name': str
}

# collate_fn 批处理后
{
    'faces': torch.Tensor,      # (B, T, 3, 160, 160)
    'audios': torch.Tensor,     # (B, 768)
    'openfaces': torch.Tensor,  # (B, T, 714) 或 None
    'labels': torch.Tensor,     # (B,)
    'video_names': List[str]
}
```

### 使用DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=LieDetectionDataset.collate_fn,
    num_workers=4
)

for batch in loader:
    faces = batch['faces']      # (8, T, 3, 160, 160)
    audios = batch['audios']    # (8, 768)
    labels = batch['labels']    # (8,)
```

---

## 特征提取器

### FaceExtractor

```python
from dataloader import FaceExtractor

extractor = FaceExtractor(
    method='opencv',           # 'opencv' 或 'mtcnn'
    target_size=(160, 160),    # 目标尺寸
    num_frames=16,             # 提取帧数
    device='cpu'
)

# 从视频提取
faces = extractor.extract_from_video('video.mp4', return_tensor=True)
# 输出: (T, 3, 160, 160)
```

### AudioExtractor

```python
from dataloader import AudioExtractor

extractor = AudioExtractor(
    method='wav2vec2',         # 'wav2vec2' 或 'mfcc'
    device='cuda'
)

# 从视频提取
audio = extractor.extract_from_video('video.mp4', return_tensor=True)
# 输出: (768,)
```

### OpenFaceExtractor

```python
from dataloader import OpenFaceExtractor

extractor = OpenFaceExtractor(
    openface_path=None,        # OpenFace可执行文件路径 (None=自动查找)
    num_frames=16,             # 提取帧数
    device='cpu'
)

# 从视频提取
openface = extractor.extract_from_video('video.mp4', return_tensor=True)
# 输出: (T, 714)
```

---

## 常用代码片段

### 1. 完整推理流程

```python
from detector import LieDetector

# 从检查点加载
detector = LieDetector.from_checkpoint('checkpoints/best_model.pth', device='cuda')

# 预测
result = detector.predict('video.mp4')

print(f"预测: {result['label_cn']}")
print(f"置信度: {result['confidence']:.2%}")
```

### 2. 完整训练流程

**方法1: 使用命令行脚本（推荐）**

```bash
# 基本训练
python train.py

# 自定义参数
python train.py \
    --num_epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --visual_hidden 512 \
    --fusion_hidden 256 \
    --dropout 0.3 \
    --device cuda \
    --save_dir checkpoints

# 从检查点恢复
python train.py --resume latest_model.pth

# 查看所有参数
python train.py --help
```

**方法2: 使用Python代码**

```python
from models.fusion import FusionModel
from trainer import FusionModelTrainer, LieDetectionDataset
from torch.utils.data import DataLoader, random_split

# 加载数据
dataset = LieDetectionDataset('src/dataset/video_labels.csv', 'src/features')
train_set, val_set = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, 
                          collate_fn=LieDetectionDataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False,
                        collate_fn=LieDetectionDataset.collate_fn)

# 创建模型和训练器
model = FusionModel(device='cuda')
trainer = FusionModelTrainer(model, train_loader, val_loader, device='cuda')

# 训练
trainer.train(num_epochs=30, lr=1e-4)
```

### 3. 提取特征

```python
from dataloader import FaceExtractor, AudioExtractor, OpenFaceExtractor

face_ext = FaceExtractor(method='opencv', num_frames=16)
audio_ext = AudioExtractor(method='wav2vec2', device='cuda')
openface_ext = OpenFaceExtractor(num_frames=16)

# 提取
faces = face_ext.extract_from_video('video.mp4', return_tensor=True)
audios = audio_ext.extract_from_video('video.mp4', return_tensor=True)
openfaces = openface_ext.extract_from_video('video.mp4', return_tensor=True)

# 保存
import torch
torch.save(faces, 'video_faces.pt')
torch.save(audios, 'video_audios.pt')
torch.save(openfaces, 'video_openfaces.pt')
```

### 4. 批量预测

```python
from pathlib import Path
from detector import LieDetector

detector = LieDetector.from_checkpoint('checkpoints/best_model.pth')

# 获取所有视频
videos = list(Path('videos').glob('*.mp4'))

# 批量预测
results = detector.predict_batch(videos, verbose=False)

# 统计
truth_count = sum(1 for r in results if r['prediction'] == 0)
deception_count = sum(1 for r in results if r['prediction'] == 1)

print(f"Truth: {truth_count}, Deception: {deception_count}")
```

---

## 参数推荐值

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| batch_size | 8-16 | GPU内存允许的情况下 |
| lr | 1e-4 | 初始学习率 |
| weight_decay | 1e-4 | 权重衰减 |
| dropout | 0.3 | Dropout比例 |
| num_frames | 16 | 视频帧数 |
| visual_hidden | 256 | 视觉隐藏层维度 |
| fusion_hidden | 128 | 融合隐藏层维度 |
| uni_layers | 2 | Transformer层数 |
| uni_nhead | 4 | 注意力头数 |
| patience | 5 | 早停耐心值 |

---

## 故障排除

### CUDA out of memory
```python
# 减小batch_size
trainer.train(batch_size=4)

# 或使用CPU
model = FusionModel(device='cpu')
```

### 特征提取失败
```python
# 检查视频文件
import cv2
cap = cv2.VideoCapture('video.mp4')
print(f"可读取: {cap.isOpened()}")
print(f"帧数: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
```

### 模型加载失败
```python
# 检查检查点
checkpoint = torch.load('model.pth', map_location='cpu')
print(checkpoint.keys())
```
