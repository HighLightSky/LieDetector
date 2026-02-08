# 安装指南

## 📦 数据加载器额外依赖

数据加载器模块需要额外的依赖包。

### 必需依赖

```bash
# OpenCV (人脸检测)
pip install opencv-python

# 音频处理
pip install librosa soundfile

# 进度条
pip install tqdm
```

### 完整安装命令

```bash
# 基础依赖 (如果还没安装)
pip install torch torchvision torchaudio
pip install transformers timm pandas

# 数据加载器依赖
pip install opencv-python librosa soundfile tqdm
```

### 可选依赖

```bash
# 更好的人脸检测
pip install dlib
pip install facenet-pytorch
```

### 外部工具

#### 1. FFmpeg (必需)

用于从视频中提取音频。

**Windows**:
1. 下载: https://ffmpeg.org/download.html
2. 解压到 `C:\ffmpeg`
3. 添加 `C:\ffmpeg\bin` 到系统 PATH

**Linux**:

```bash
sudo apt-get install ffmpeg
```

**Mac**:
```bash
brew install ffmpeg
```

**验证安装**:
```bash
ffmpeg -version
```

#### 2. OpenFace (可选)

用于提取精细的面部动作单元特征。

**下载和安装**:
1. 访问: https://github.com/TadasBaltrusaitis/OpenFace
2. 按照官方文档编译
3. 添加到系统 PATH 或在代码中指定路径

**注意**: OpenFace 安装较复杂，如果遇到问题可以跳过，只使用 Face 和 Audio 模态。

## ✅ 验证安装

运行测试脚本:

```bash
python test_dataloader.py
```

预期输出:
```
✅ FaceExtractor 初始化成功
✅ AudioExtractor 初始化成功
✅ LieDetectionDataset 导入成功
🎉 所有测试通过!
```

## 🐛 常见问题

### Q: opencv-python 安装失败

**A**: 尝试安装 headless 版本:
```bash
pip install opencv-python-headless
```

### Q: librosa 安装失败

**A**: 确保安装了 soundfile:
```bash
pip install soundfile
pip install librosa
```

### Q: FFmpeg 未找到

**A**: 
1. 确认已安装 FFmpeg
2. 检查是否在 PATH 中: `ffmpeg -version`
3. 重启终端/IDE

### Q: OpenFace 安装太复杂

**A**: 可以跳过 OpenFace，只使用 Face 和 Audio 模态:
```python
# 不使用 OpenFace
from dataloader import FaceExtractor, AudioExtractor

face_extractor = FaceExtractor()
audio_extractor = AudioExtractor()
```

## 📝 完整依赖列表

创建 `requirements_dataloader.txt`:

```
opencv-python>=4.8.0
librosa>=0.10.0
soundfile>=0.12.0
tqdm>=4.65.0
pandas>=2.0.0
```

安装:
```bash
pip install -r requirements_dataloader.txt
```

---

**下一步**: 查看 [数据加载器指南](DATALOADER_GUIDE.md)
