# 快速参考手册

## 📦 类与方法速查表

### models/faces.py

#### `FacesModel`
```python
# 初始化
model = FacesModel(device='cpu', model_name='mobilenetv3_small_100')

# 方法
forward(x)              # 输入: (B, 3, 160, 160) → 输出: (B, 2)
predict_dir(dir)        # 输入: 图像目录路径 → 输出: 平均概率
encode_dir(dir)         # 输入: 图像目录路径 → 输出: 1024维特征向量
```

**特征**:
- 骨干网络: MobileNetV3 (预训练，冻结)
- 特征维度: 1024
- 输出类别: 2 (真话/谎言)

---

### models/openface.py

#### `OpenfaceModel`
```python
# 初始化
model = OpenfaceModel(input_dim=714, num_classes=2, dropout=0.3)

# 方法
forward(x)              # 输入: (B, 714) → 输出: (B, 2)
```

**特征**:
- 输入: OpenFace 714维特征
- 网络: 3层全连接 (714→512→256→2)
- 输出类别: 2

---

### models/audio.py

#### `Wav2Vec2Encoder`
```python
# 初始化
encoder = Wav2Vec2Encoder(device='cpu', model_name='facebook/wav2vec2-base')

# 方法
__call__(wav_path)      # 输入: 音频文件路径 → 输出: 768维 numpy 向量
```

**特征**:
- 预训练模型: Wav2Vec2-Base
- 输出维度: 768
- 参数: 冻结

#### `AudioModel`
```python
# 初始化
model = AudioModel(use_mfcc=False, drop=0.2)

# 方法
forward(audio_features) # 输入: (B, 768) → 输出: (B, 2)
```

**特征**:
- 输入: Wav2Vec2 特征
- 网络: 4层全连接 (768→512→256→128→2)
- 可选: MFCC 特征融合

#### `mfcc_vector()`
```python
# 函数调用
features = mfcc_vector(wav_path, n_mfcc=40, sr=16_000)
# 输出: 40维 MFCC 特征向量
```

---

### plan.py

#### `Transpose`
```python
# 辅助模块，用于维度转置
transpose = Transpose(dim0=1, dim1=2)
output = transpose(x)  # x.transpose(1, 2)
```

#### `W` (动态权重模块)
```python
# 初始化
w_module = W(hidden_dim=256)

# 方法
forward(U_face, U_of, U_audio)
# 输入: 三个 (B, 256) 张量
# 输出: (B, 3) 权重向量 (softmax 归一化)
```

**功能**: 自动学习三个模态的重要性权重

#### `CrossAttentionBlock`
```python
# 初始化
attn = CrossAttentionBlock(dim_q=256, dim_kv=256, num_heads=4)

# 方法
forward(q, k, v)
# 输入: q (B, 256), k (B, T, 256), v (B, T, 256)
# 输出: (B, 256) 注意力融合特征
```

**功能**: 跨模态注意力机制

#### `UniEncoder`
```python
# 初始化
encoder = UniEncoder(
    d_in=1024,
    d_model=256,
    nhead=4,
    num_layers=2,
    ff_hidden=512,
    dropout=0.1
)

# 方法
forward(x)
# 输入: (B, T, d_in)
# 输出: (B, T, d_model)
```

**功能**: 单模态时序编码器（Transformer）

#### `FusionModel` (核心模型)
```python
# 初始化
model = FusionModel(
    device='cuda',
    num_classes=2,
    visual_hidden=256,
    fusion_hidden=128,
    dropout=0.3,
    freeze_backbones=True,
    uni_layers=2,
    uni_nhead=4,
    com_heads=4,
    face_checkpoint=None,
    openface_checkpoint=None,
    audio_checkpoint=None
)

# 方法
forward(faces, openfaces, audios)
# 输入:
#   faces: (B, T, 3, 160, 160)
#   openfaces: (B, T, 714)
#   audios: (B, 768)
# 输出: dict (见下方)
```

**输出结构**:
```python
{
    'logits': {
        'face': (B, 2),
        'openface': (B, 2),
        'audio': (B, 2),
        'fa_au': (B, 2),
        'fa_of': (B, 2),
    },
    'probs': {
        'face': (B, 2),
        'openface': (B, 2),
        'audio': (B, 2),
        'fa_au': (B, 2),
        'fa_of': (B, 2),
        'fused': (B, 2),  # ⭐ 最终预测
    },
    'weights': (B, 3),  # [w_face, w_of, w_audio]
    'cos': {
        'fa_fa_au': (B,),
        'fa_fa_of': (B,),
        'au_fa_au': (B,),
        'of_fa_of': (B,),
    }
}
```

**辅助方法**:
```python
matrix_sqrt(M)              # 矩阵平方根
fit_gmm(feats, K, max_iter) # GMM 拟合
sinkhorn(a, b, C, reg, num_iter)  # Sinkhorn 算法
compute_stats(feats)        # 统计量计算
mmd(x, y, sigma)            # 最大均值差异
```

---

## 🎯 常用代码片段

### 1. 加载模型
```python
import torch
from plan import FusionModel

model = FusionModel(device='cuda')
model.eval()
```

### 2. 推理单个样本
```python
# 准备数据
faces = torch.randn(1, 16, 3, 160, 160)
openfaces = torch.randn(1, 16, 714)
audios = torch.randn(1, 768)

# 推理
with torch.no_grad():
    output = model(faces, openfaces, audios)

# 获取结果
prob = output['probs']['fused'][0]  # [P(真话), P(谎言)]
pred = prob.argmax().item()         # 0 或 1
confidence = prob.max().item()      # 置信度

print(f"预测: {'谎言' if pred == 1 else '真话'}")
print(f"置信度: {confidence:.2%}")
```

### 3. 批量推理
```python
# 批量数据
batch_size = 8
faces = torch.randn(batch_size, 16, 3, 160, 160)
openfaces = torch.randn(batch_size, 16, 714)
audios = torch.randn(batch_size, 768)

# 推理
with torch.no_grad():
    output = model(faces, openfaces, audios)

# 批量结果
probs = output['probs']['fused']    # (8, 2)
preds = probs.argmax(dim=1)         # (8,)
weights = output['weights']         # (8, 3)

for i in range(batch_size):
    print(f"样本 {i}: 预测={preds[i]}, "
          f"权重=[{weights[i, 0]:.2f}, {weights[i, 1]:.2f}, {weights[i, 2]:.2f}]")
```

### 4. 训练循环
```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for faces, openfaces, audios, labels in dataloader:
        # 前向传播
        output = model(faces, openfaces, audios)
        
        # 计算损失
        loss = criterion(output['logits']['fused'], labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 5. 保存/加载检查点
```python
# 保存
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

### 6. 查看模态权重
```python
output = model(faces, openfaces, audios)
weights = output['weights']  # (B, 3)

# 平均权重
avg_weights = weights.mean(dim=0)
print(f"Face 权重: {avg_weights[0]:.3f}")
print(f"OpenFace 权重: {avg_weights[1]:.3f}")
print(f"Audio 权重: {avg_weights[2]:.3f}")
```

### 7. 分析模态一致性
```python
output = model(faces, openfaces, audios)
cos_sim = output['cos']

# 余弦相似度
print(f"Face-Audio 一致性: {cos_sim['fa_fa_au'].mean():.3f}")
print(f"Face-OpenFace 一致性: {cos_sim['fa_fa_of'].mean():.3f}")
```

---

## 🔧 调试技巧

### 1. 检查输入形状
```python
print(f"Faces: {faces.shape}")        # 应该是 (B, T, 3, 160, 160)
print(f"OpenFaces: {openfaces.shape}") # 应该是 (B, T, 714)
print(f"Audios: {audios.shape}")       # 应该是 (B, 768)
```

### 2. 检查输出范围
```python
output = model(faces, openfaces, audios)
probs = output['probs']['fused']

assert (probs >= 0).all() and (probs <= 1).all(), "概率超出范围"
assert torch.allclose(probs.sum(dim=1), torch.ones(probs.size(0))), "概率和不为1"
```

### 3. 检查梯度
```python
model.train()
output = model(faces, openfaces, audios)
loss = criterion(output['logits']['fused'], labels)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### 4. 可视化模态权重
```python
import matplotlib.pyplot as plt

weights = output['weights'].detach().cpu().numpy()  # (B, 3)

plt.figure(figsize=(10, 4))
plt.bar(['Face', 'OpenFace', 'Audio'], weights.mean(axis=0))
plt.ylabel('Average Weight')
plt.title('Modality Importance')
plt.show()
```

---

## ⚠️ 常见错误

### 错误 1: BatchNorm 单样本错误
```python
# ❌ 错误
model.train()
output = model(single_sample)  # batch_size=1

# ✅ 正确
model.eval()
with torch.no_grad():
    output = model(single_sample)
```

### 错误 2: 维度不匹配
```python
# ❌ 错误
faces = torch.randn(2, 16, 160, 160, 3)  # 通道在最后

# ✅ 正确
faces = torch.randn(2, 16, 3, 160, 160)  # 通道在前
# 或者代码会自动处理 (B, T, H, W, C) 格式
```

### 错误 3: 帧数不匹配
```python
# ❌ 错误
faces = torch.randn(2, 16, 3, 160, 160)
openfaces = torch.randn(2, 20, 714)  # T 不一致

# ✅ 正确
faces = torch.randn(2, 16, 3, 160, 160)
openfaces = torch.randn(2, 16, 714)  # T 一致
```

### 错误 4: 设备不匹配
```python
# ❌ 错误
model = model.cuda()
faces = torch.randn(2, 16, 3, 160, 160)  # CPU 张量

# ✅ 正确
model = model.cuda()
faces = torch.randn(2, 16, 3, 160, 160).cuda()
```

---

## 📊 性能优化

### 1. 使用混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for faces, openfaces, audios, labels in dataloader:
    with autocast():
        output = model(faces, openfaces, audios)
        loss = criterion(output['logits']['fused'], labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. 梯度累积
```python
accumulation_steps = 4

for i, (faces, openfaces, audios, labels) in enumerate(dataloader):
    output = model(faces, openfaces, audios)
    loss = criterion(output['logits']['fused'], labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 数据并行
```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## 📝 配置模板

### 训练配置
```python
config = {
    'model': {
        'device': 'cuda',
        'num_classes': 2,
        'visual_hidden': 256,
        'fusion_hidden': 128,
        'dropout': 0.3,
        'freeze_backbones': True,
    },
    'training': {
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'scheduler': 'cosine',
    },
    'data': {
        'num_frames': 16,
        'image_size': 160,
        'audio_sr': 16000,
    }
}
```

---

**版本**: v1.0  
**最后更新**: 2025-02-08
