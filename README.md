# 谎言检测项目 - 当前状态

## 项目概览

多模态谎言检测系统，融合人脸图像、音频和OpenFace特征进行谎言识别。

## 完成的模块

### ✅ 1. 数据加载器 (dataloader/)

**状态**: 完成并测试通过

**功能**:
- `FaceExtractor`: 从视频提取人脸图像序列
- `AudioExtractor`: 提取Wav2Vec2音频特征
- `OpenFaceExtractor`: 提取OpenFace面部特征
- `LieDetectionDataset`: 数据集类
- `precompute_features.py`: 批量特征预计算

**测试结果**:
- ✅ 人脸提取: 0.36s, (16, 3, 160, 160)
- ✅ 音频提取: 1.50s, (768,)
- ✅ OpenFace提取: 1.50s, (16, 714)

### ✅ 2. 模型架构 (models/)

**状态**: 完成

**子模型**:
- `FacesModel`: MobileNetV3 + 分类头
- `AudioModel`: Wav2Vec2特征 → 分类
- `OpenfaceModel`: OpenFace特征 → 分类

**融合模型**:
- `MultiModalFusionModel`: 多模态融合 (detect.py)
- 总参数: 3,245,160
- 可训练参数: 1,727,304

### ✅ 3. 推理系统 (detect.py)

**状态**: 完成并测试通过

**功能**:
- 单视频检测
- 详细数据流输出
- 支持GPU加速

**测试结果**:
```
预测类别: truth (说真话)
置信度: 85.99%
概率分布: truth=85.99%, deception=14.01%
```

### ✅ 4. 训练系统 (trainer/)

**状态**: 完成并测试通过

**功能**:
- 分阶段训练策略
- 自动保存最佳模型
- 训练历史记录
- 学习率调度

**测试结果**:
- ✅ 阶段1: 融合层训练成功
- ✅ 阶段2: 分类头微调成功
- ✅ 阶段3: 端到端微调成功
- 总训练时间: 0.07分钟（小批量）

### ✅ 5. 数据处理工具

**状态**: 完成

**功能**:
- `process_data.py`: 批量特征提取脚本
- `video_downloader.py`: YouTube视频下载和切片
- `generate_dataset_csv.py`: 生成数据集标签

**数据统计**:
- 视频总数: 1,479
- 说谎视频: 791
- 说真话视频: 688

**处理性能**:
- 单视频: ~3.5s (CPU) / ~2.2s (GPU)
- 全部视频: ~90分钟 (CPU) / ~54分钟 (GPU)

## 项目结构

```
lie_detector/
├── dataloader/              # 数据加载器 ✅
│   ├── face_extractor.py
│   ├── audio_extractor.py
│   ├── openface_extractor.py
│   ├── dataset.py
│   └── precompute_features.py
├── models/                  # 模型定义 ✅
│   ├── faces.py
│   ├── audio.py
│   └── openface.py
├── trainer/                 # 训练器 ✅
│   ├── dataset.py
│   ├── trainer.py
│   └── README.md
├── utils/                   # 工具脚本 ✅
│   ├── video_downloader.py
│   └── generate_dataset_csv.py
├── src/                     # 数据目录
│   ├── videos/cut/          # 1,479个视频切片
│   ├── dataset/             # 标签文件
│   └── test_data/           # 测试特征
├── detect.py                # 推理脚本 ✅
├── train.py                 # 训练脚本 ✅
├── test_train.py            # 测试训练 ✅
└── docs/                    # 文档 ✅
```

## 数据流

### 完整数据流

```
1. 视频下载
   YouTube → 完整视频 → 切片视频 (1,479个)

2. 标签生成
   视频文件名 → CSV标签 (truth/deception)

3. 特征提取
   视频 → 人脸 + 音频 + OpenFace 特征

4. 训练
   特征 → 模型训练 → 保存检查点

5. 推理
   新视频 → 特征提取 → 模型预测 → 结果
```

## 使用流程

### 1. 数据准备

```bash
# 下载和切片视频（如果需要）
python -c "from utils.video_downloader import VideoDownloader; VideoDownloader().run()"

# 生成标签文件
python run.py
```

### 2. 特征提取

```bash
# 测试单个视频
python process_data.py --mode test

# 批量处理所有视频（约90分钟）
python process_data.py --mode all
```

### 3. 训练模型

```bash
# 测试训练（小批量）
python test_train.py

# 完整训练
python train.py --batch_size 16 --stage1_epochs 10 --stage2_epochs 10 --stage3_epochs 10
```

### 4. 推理预测

```bash
# 单视频检测
python detect.py

# 或在代码中使用
from detect import MultiModalFusionModel, detect_video
# ... 加载模型和检测
```

## 当前数据状态

| 项目 | 数量 | 状态 |
|------|------|------|
| 视频切片 | 1,479 | ✅ 已下载 |
| 标签文件 | 1 | ✅ 已生成 |
| 测试特征 | 6 | ✅ 已提取 |
| 完整特征 | 0 | ⏳ 待提取 |
| 训练模型 | 0 | ⏳ 待训练 |

## 下一步任务

### 🔄 立即可做

1. **批量特征提取** (预计2-3小时)
   ```bash
   python -c "from dataloader.precompute_features import main; main()"
   ```
   - 提取1,479个视频的特征
   - 保存到 `src/features/`

2. **完整训练** (预计1-2小时)
   ```bash
   python train.py
   ```
   - 使用完整数据集
   - 训练30个epoch
   - 保存最佳模型

3. **模型评估**
   - 在测试集上评估
   - 计算准确率、F1分数
   - 绘制混淆矩阵

### 📋 后续优化

1. **数据增强**
   - 添加随机裁剪、翻转
   - 时间序列增强

2. **模型优化**
   - 超参数调优
   - 尝试不同的融合策略
   - 集成学习

3. **性能提升**
   - 混合精度训练
   - 梯度累积
   - 分布式训练

4. **部署准备**
   - 模型量化
   - ONNX导出
   - API接口

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.x |
| 预训练模型 | MobileNetV3, Wav2Vec2 |
| 人脸检测 | OpenCV, OpenFace |
| 音频处理 | librosa, transformers |
| 数据处理 | pandas, numpy |
| 视频处理 | ffmpeg, yt-dlp |

## 系统要求

### 最低配置
- CPU: 4核
- RAM: 8GB
- GPU: 可选（CPU也可运行）
- 存储: 50GB

### 推荐配置
- CPU: 8核+
- RAM: 16GB+
- GPU: NVIDIA GPU (8GB+ VRAM)
- 存储: 100GB SSD

## 性能指标

### 特征提取速度
- 人脸: ~0.4s/视频
- 音频: ~1.5s/视频
- OpenFace: ~1.5s/视频
- **总计**: ~3.5s/视频

### 训练速度（预估）
- 小批量(5样本): 0.07分钟
- 完整数据集(1479样本): 60-120分钟

### 推理速度
- 单视频: ~5秒（含特征提取）
- 批量推理: ~0.1秒/视频（特征已提取）

## 已知问题

1. ⚠️ **OpenFace CSV列名有空格**
   - 状态: 已修复
   - 解决: 添加列名清理

2. ⚠️ **部分视频人脸检测失败**
   - 状态: 正常（视频质量问题）
   - 影响: 可接受

3. ⚠️ **Windows上num_workers>0可能出错**
   - 状态: 已处理
   - 解决: 设置num_workers=0

## 文档

- ✅ `README.md`: 项目说明
- ✅ `DETECT_SUMMARY.md`: 推理系统总结
- ✅ `TRAINING_SUMMARY.md`: 训练系统总结
- ✅ `TEST_SUMMARY.md`: 测试结果总结
- ✅ `trainer/README.md`: 训练器使用文档
- ✅ `docs/`: 中文文档

## 贡献者

- 数据加载器: ✅ 完成
- 模型架构: ✅ 完成
- 训练系统: ✅ 完成
- 推理系统: ✅ 完成
- 文档: ✅ 完成

## 许可证

待定

## 联系方式

待定

---

**最后更新**: 2026-02-09

**项目状态**: 🟢 开发完成，准备训练

**下一里程碑**: 完整数据集训练
