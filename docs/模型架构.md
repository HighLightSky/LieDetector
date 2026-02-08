# 系统架构详解

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         输入数据层                                │
├─────────────────────────────────────────────────────────────────┤
│  Faces (B,T,3,160,160)  │  OpenFace (B,T,714)  │  Audio (B,768) │
└────────┬────────────────┴──────────┬───────────┴────────┬────────┘
         │                           │                     │
         ▼                           ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      特征提取层 (Backbone)                        │
├─────────────────────────────────────────────────────────────────┤
│  MobileNetV3          │  OpenfaceModel      │  AudioModel       │
│  (冻结预训练)          │  (全连接网络)        │  (全连接网络)      │
│  1024-D               │  714-D              │  768-D            │
└────────┬──────────────┴──────────┬──────────┴────────┬──────────┘
         │                         │                    │
         ▼                         ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    单模态编码层 (E_uni)                           │
├─────────────────────────────────────────────────────────────────┤
│  Transformer Encoder  │  Transformer Encoder │  FC Layers       │
│  + Pooling            │  + Pooling           │                  │
│  256-D                │  256-D               │  256-D           │
└────────┬──────────────┴──────────┬───────────┴────────┬─────────┘
         │                         │                     │
         └─────────────┬───────────┴──────────┬──────────┘
                       ▼                      ▼
              ┌─────────────────────────────────────┐
              │      动态权重模块 (W)                 │
              │   输入: [U_face, U_of, U_audio]     │
              │   输出: [w_face, w_of, w_audio]     │
              │   (Softmax 归一化)                   │
              └─────────────────┬───────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
    w_face * U_face      w_of * U_of         w_audio * U_audio
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                              ▼
┌─────────────────────────┐                  ┌─────────────────────┐
│  跨模态注意力 (E_com)     │                  │  跨模态注意力 (E_com) │
│  Face ↔ Audio           │                  │  Face ↔ OpenFace    │
│  CrossAttention         │                  │  CrossAttention     │
│  C_fa_au (256-D)        │                  │  C_fa_of (256-D)    │
└────────┬────────────────┘                  └────────┬────────────┘
         │                                             │
         ▼                                             ▼
┌─────────────────────────┐                  ┌─────────────────────┐
│  融合分类器 (prob_fa_au) │                  │  融合分类器 (prob_fa_of)│
│  [U_face, U_audio,      │                  │  [U_face, U_of,     │
│   C_fa_au] → 2 classes  │                  │   C_fa_of] → 2 classes│
└────────┬────────────────┘                  └────────┬────────────┘
         │                                             │
         └──────────────────┬──────────────────────────┘
                            ▼
                   ┌─────────────────┐
                   │   加权融合       │
                   │  w_audio*P_fa_au │
                   │  + w_of*P_fa_of  │
                   │  + w_face*P_face │
                   └────────┬─────────┘
                            ▼
                   ┌─────────────────┐
                   │   最终输出       │
                   │[P(真话), P(谎言)]│
                   └─────────────────┘
```

---

## 数据流详解

### 阶段 1: 特征提取

```python
# 输入
faces:     (B, T, 3, 160, 160)  # B个样本，T帧，RGB图像
openfaces: (B, T, 714)          # OpenFace 特征序列
audios:    (B, 768)             # Wav2Vec2 音频特征

# 处理
faces → MobileNetV3 → face_feats (B, T, 1024)
openfaces → 保持原样 → of_feats (B, T, 714)
audios → 保持原样 → audio_feats (B, 768)
```

### 阶段 2: 单模态编码

```python
# Face 编码
face_feats (B, T, 1024)
  → Linear(1024→256)
  → Transformer Encoder (2层, 4头)
  → Temporal Pooling
  → U_face (B, 256)

# OpenFace 编码
of_feats (B, T, 714)
  → Linear(714→256)
  → Transformer Encoder (2层, 4头)
  → Temporal Pooling
  → U_of (B, 256)

# Audio 编码
audio_feats (B, 768)
  → Linear(768→512)
  → Linear(512→256)
  → U_audio (B, 256)
```

### 阶段 3: 动态权重计算

```python
# 拼接三个模态
concat = [U_face, U_of, U_audio]  # (B, 768)

# 通过 W 模块
W_scores = FC(768→256→3)          # (B, 3)
W = Softmax(W_scores)             # (B, 3), 和为1

# 加权
U_face_weighted = W[:, 0:1] * U_face
U_of_weighted = W[:, 1:2] * U_of
U_audio_weighted = W[:, 2:3] * U_audio
```

### 阶段 4: 跨模态注意力

```python
# Face-Audio 交叉注意力
X_face1 = Linear(face_feats)      # (B, T, 256)
X_audio1 = Linear(audio_feats)    # (B, 256)
C_fa_au = CrossAttention(
    q=X_audio1,                   # Query: Audio
    k=X_face1,                    # Key: Face
    v=X_face1                     # Value: Face
)  # (B, 256)

# Face-OpenFace 交叉注意力
X_face2 = Linear(face_feats)      # (B, T, 256)
X_of1 = Linear(of_feats)          # (B, 256)
C_fa_of = CrossAttention(
    q=X_of1,                      # Query: OpenFace
    k=X_face2,                    # Key: Face
    v=X_face2                     # Value: Face
)  # (B, 256)
```

### 阶段 5: 融合预测

```python
# 两路融合
fuse_fa_au = [U_face, U_audio, C_fa_au]  # (B, 768)
fuse_fa_of = [U_face, U_of, C_fa_of]     # (B, 768)

# 分类
probs_fa_au = Softmax(FC(fuse_fa_au))    # (B, 2)
probs_fa_of = Softmax(FC(fuse_fa_of))    # (B, 2)

# 加权融合
fused_probs = (
    W[:, 2:3] * probs_fa_au +
    W[:, 1:2] * probs_fa_of +
    W[:, 0:1] * probs_face
)  # (B, 2)
```

---

## 关键设计思想

### 1. 多层次融合

```
Level 1: 单模态独立预测
  ├─ Face → probs_face
  ├─ OpenFace → probs_of
  └─ Audio → probs_au

Level 2: 双模态融合预测
  ├─ Face + Audio → probs_fa_au
  └─ Face + OpenFace → probs_fa_of

Level 3: 动态加权融合
  └─ W * [probs_face, probs_fa_of, probs_fa_au] → fused_probs
```

### 2. 动态权重机制

**为什么需要动态权重？**
- 不同样本中，各模态的可靠性不同
- 例如：噪音环境下，音频不可靠，应降低权重
- 动态权重自动学习最优组合

**如何实现？**
```python
# 输入三个模态的特征
W_scores = MLP([U_face, U_of, U_audio])
W = Softmax(W_scores)  # 归一化到 [0, 1]，和为 1

# 示例输出
W = [0.4, 0.3, 0.3]  # Face 最重要
W = [0.2, 0.2, 0.6]  # Audio 最重要
```

### 3. 跨模态注意力

**为什么需要跨模态注意力？**
- 捕捉模态间的互补信息
- 例如：面部微笑 + 语音颤抖 → 可能在说谎
- 注意力机制自动发现关联

**如何工作？**
```python
# Audio 关注 Face 的哪些部分？
Attention(q=audio, k=face, v=face)
  → 找到与音频相关的面部特征
  → 融合成新的表示 C_fa_au

# 类似地
Attention(q=openface, k=face, v=face)
  → 找到与 OpenFace 相关的面部特征
  → 融合成新的表示 C_fa_of
```

### 4. 时序建模

**Transformer Encoder 的作用**:
- 输入：帧序列 (B, T, D)
- 输出：编码序列 (B, T, D)
- 功能：捕捉帧间的时序依赖

**示例**:
```python
# 输入：16 帧面部特征
face_sequence = [frame_1, frame_2, ..., frame_16]

# Transformer 处理
encoded = TransformerEncoder(face_sequence)
# 每一帧都能"看到"其他帧的信息

# 时序池化
U_face = AvgPool(encoded)  # 聚合所有帧
```

---

## 模型优势

### 1. 鲁棒性
- **多模态冗余**: 一个模态失效，其他模态补偿
- **动态权重**: 自动降低不可靠模态的影响
- **预训练骨干**: 利用大规模数据的先验知识

### 2. 可解释性
- **模态权重**: 知道哪个模态最重要
- **余弦相似度**: 衡量模态间的一致性
- **独立预测**: 可以查看每个模态的判断

### 3. 灵活性
- **任意帧数**: 支持不同长度的视频
- **模块化设计**: 可以单独训练/替换子模型
- **检查点加载**: 支持预训练模型微调

---

## 计算复杂度分析

### 参数量统计

```
FacesModel:
  - Backbone (MobileNetV3): 1.5M (冻结)
  - FC Block: 0.8M
  小计: 2.3M

OpenfaceModel:
  - FC Layers: 0.5M

AudioModel:
  - FC Layers: 0.6M

FusionModel:
  - E_uni_face: 0.3M
  - E_uni_of: 0.2M
  - E_uni_audio: 0.4M
  - CrossAttention: 0.5M
  - W Module: 0.2M
  - Classifiers: 0.3M
  小计: 1.9M

总计: ~5.3M 参数
```

### 推理时间（单样本）

```
GPU (RTX 3090):
  - Face 特征提取: ~5ms
  - OpenFace 处理: ~1ms
  - Audio 处理: ~1ms
  - 融合模块: ~3ms
  总计: ~10ms (100 FPS)

CPU (Intel i7):
  - 总计: ~50ms (20 FPS)
```

### 内存占用

```
模型参数: ~20 MB
中间激活 (batch_size=8, T=16):
  - Face features: ~50 MB
  - OpenFace features: ~1 MB
  - Audio features: ~0.1 MB
  - Fusion activations: ~10 MB
  总计: ~80 MB
```

---

## 训练策略建议

### 1. 分阶段训练

```python
# 阶段 1: 预训练子模型
train(FacesModel, face_dataset)
train(OpenfaceModel, openface_dataset)
train(AudioModel, audio_dataset)

# 阶段 2: 冻结子模型，训练融合层
freeze(face, openface, audio)
train(FusionModel, multimodal_dataset)

# 阶段 3: 端到端微调
unfreeze(all)
finetune(FusionModel, multimodal_dataset, lr=1e-5)
```

### 2. 损失函数设计

```python
# 多任务损失
loss = (
    λ1 * CE(logits_face, labels) +
    λ2 * CE(logits_of, labels) +
    λ3 * CE(logits_audio, labels) +
    λ4 * CE(logits_fa_au, labels) +
    λ5 * CE(logits_fa_of, labels) +
    λ6 * CE(logits_fused, labels)
)

# 推荐权重
λ = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
```

### 3. 数据增强

```python
# 视觉增强
- 随机裁剪
- 颜色抖动
- 随机翻转

# 音频增强
- 时间拉伸
- 音高变换
- 添加噪声

# 时序增强
- 随机采样帧
- 时间反转
```

---

**文档版本**: v1.0  
**最后更新**: 2025-02-08
