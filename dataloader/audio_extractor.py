"""
音频特征提取器
使用 Wav2Vec2 或 MFCC 提取音频特征
"""

import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import warnings
import subprocess
import tempfile


class AudioExtractor:
    """从音频或视频中提取音频特征
    
    支持两种特征提取方法:
    - wav2vec2: 使用 Wav2Vec2 预训练模型 (推荐)
    - mfcc: 使用 MFCC 特征
    
    Args:
        method: 特征提取方法 ('wav2vec2' 或 'mfcc')
        model_name: Wav2Vec2 模型名称
        feature_dim: 特征维度 (wav2vec2: 768, mfcc: 40)
        sample_rate: 音频采样率
        device: 设备 ('cpu' 或 'cuda')
    """
    
    def __init__(
        self,
        method: str = 'wav2vec2',
        model_name: str = 'facebook/wav2vec2-base',
        feature_dim: Optional[int] = None,
        sample_rate: int = 16000,
        device: str = 'cpu'
    ):
        self.method = method
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        
        # 设置特征维度
        if feature_dim is None:
            self.feature_dim = 768 if method == 'wav2vec2' else 40
        else:
            self.feature_dim = feature_dim
        
        # 初始化特征提取器
        self._init_extractor()
    
    def _init_extractor(self):
        """初始化特征提取器"""
        if self.method == 'wav2vec2':
            try:
                from transformers import Wav2Vec2Processor, Wav2Vec2Model
                
                print(f"加载 Wav2Vec2 模型: {self.model_name}")
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                
                # 冻结参数
                for p in self.model.parameters():
                    p.requires_grad = False
                
                print("Wav2Vec2 模型加载成功")
            
            except ImportError:
                raise ImportError(
                    "请安装 transformers: pip install transformers"
                )
            except Exception as e:
                raise RuntimeError(f"加载 Wav2Vec2 模型失败: {e}")
        
        elif self.method == 'mfcc':
            # MFCC 不需要预加载模型
            pass
        
        else:
            raise ValueError(f"不支持的方法: {self.method}")
    
    def extract_audio_from_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            output_path: 输出音频文件路径，None 表示使用临时文件
        
        Returns:
            音频文件路径
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 确定输出路径
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
        else:
            output_path = Path(output_path)
        
        # 使用 ffmpeg 提取音频
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-vn',  # 不要视频
                    '-acodec', 'pcm_s16le',  # PCM 16-bit
                    '-ar', str(self.sample_rate),  # 采样率
                    '-ac', '1',  # 单声道
                    '-y',  # 覆盖输出文件
                    str(output_path)
                ],
                check=True,
                capture_output=True,
                timeout=60
            )
        except FileNotFoundError:
            raise RuntimeError(
                "未找到 ffmpeg。请安装: https://ffmpeg.org/download.html"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"音频提取失败: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("音频提取超时")
        
        return output_path
    
    def load_audio(
        self,
        audio_path: Union[str, Path]
    ) -> np.ndarray:
        """加载音频文件
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            音频波形 (numpy array)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 使用 librosa 加载
        try:
            waveform, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=True
            )
            return waveform
        except Exception as e:
            raise RuntimeError(f"加载音频失败: {e}")
    
    def extract_wav2vec2_features(
        self,
        waveform: np.ndarray
    ) -> np.ndarray:
        """使用 Wav2Vec2 提取特征
        
        Args:
            waveform: 音频波形
        
        Returns:
            特征向量 (768,)
        """
        # 预处理
        inputs = self.processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors='pt',
            padding=True
        )
        
        # 移到设备
        inputs = inputs.input_values.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(inputs)
            hidden_states = outputs.last_hidden_state  # (1, T, 768)
        
        # 时序平均池化
        features = hidden_states.mean(dim=1).squeeze(0)  # (768,)
        
        return features.cpu().numpy()
    
    def extract_mfcc_features(
        self,
        waveform: np.ndarray,
        n_mfcc: Optional[int] = None
    ) -> np.ndarray:
        """使用 MFCC 提取特征
        
        Args:
            waveform: 音频波形
            n_mfcc: MFCC 系数数量
        
        Returns:
            特征向量 (n_mfcc,)
        """
        if n_mfcc is None:
            n_mfcc = self.feature_dim
        
        # 提取 MFCC
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=n_mfcc
        )  # (n_mfcc, T)
        
        # 时序平均
        features = mfcc.mean(axis=1)  # (n_mfcc,)
        
        return features
    
    def extract_from_audio(
        self,
        audio_path: Union[str, Path],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """从音频文件中提取特征
        
        Args:
            audio_path: 音频文件路径
            return_tensor: 是否返回 PyTorch tensor
        
        Returns:
            音频特征向量
            - 如果 return_tensor=True: torch.Tensor (feature_dim,)
            - 如果 return_tensor=False: np.ndarray (feature_dim,)
        """
        # 加载音频
        waveform = self.load_audio(audio_path)
        
        # 提取特征
        if self.method == 'wav2vec2':
            features = self.extract_wav2vec2_features(waveform)
        elif self.method == 'mfcc':
            features = self.extract_mfcc_features(waveform)
        else:
            raise ValueError(f"不支持的方法: {self.method}")
        
        # 转换为 tensor
        if return_tensor:
            features = torch.from_numpy(features).float()
        
        return features
    
    def extract_from_video(
        self,
        video_path: Union[str, Path],
        return_tensor: bool = True,
        keep_audio: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """从视频文件中提取音频特征
        
        Args:
            video_path: 视频文件路径
            return_tensor: 是否返回 PyTorch tensor
            keep_audio: 是否保留提取的音频文件
        
        Returns:
            音频特征向量
        """
        # 提取音频
        audio_path = self.extract_audio_from_video(video_path)
        
        try:
            # 提取特征
            features = self.extract_from_audio(audio_path, return_tensor)
        finally:
            # 清理临时文件
            if not keep_audio and audio_path.exists():
                audio_path.unlink()
        
        return features
    
    def batch_extract(
        self,
        paths: List[Union[str, Path]],
        from_video: bool = False,
        return_tensor: bool = True
    ) -> List[Union[torch.Tensor, np.ndarray]]:
        """批量提取多个文件的特征
        
        Args:
            paths: 文件路径列表
            from_video: 是否从视频提取
            return_tensor: 是否返回 PyTorch tensor
        
        Returns:
            特征列表
        """
        features_list = []
        
        for path in paths:
            try:
                if from_video:
                    features = self.extract_from_video(path, return_tensor)
                else:
                    features = self.extract_from_audio(path, return_tensor)
                features_list.append(features)
            except Exception as e:
                warnings.warn(f"提取失败 {path}: {e}")
                # 返回零特征
                if return_tensor:
                    features_list.append(torch.zeros(self.feature_dim))
                else:
                    features_list.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        return features_list
    
    def __call__(
        self,
        source: Union[str, Path],
        from_video: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """便捷调用接口
        
        Args:
            source: 音频或视频文件路径
            from_video: 是否从视频提取
            **kwargs: 传递给提取方法
        
        Returns:
            音频特征向量
        """
        if from_video:
            return self.extract_from_video(source, **kwargs)
        else:
            return self.extract_from_audio(source, **kwargs)


# 示例用法
if __name__ == '__main__':
    # 创建提取器 (Wav2Vec2)
    extractor_w2v = AudioExtractor(
        method='wav2vec2',
        device='cpu'
    )
    
    # 从音频提取
    try:
        features = extractor_w2v('path/to/audio.wav')
        print(f"Wav2Vec2 特征形状: {features.shape}")  # (768,)
    except Exception as e:
        print(f"提取失败: {e}")
    
    # 从视频提取
    try:
        features = extractor_w2v('path/to/video.mp4', from_video=True)
        print(f"从视频提取的特征形状: {features.shape}")
    except Exception as e:
        print(f"提取失败: {e}")
    
    # 创建提取器 (MFCC)
    extractor_mfcc = AudioExtractor(
        method='mfcc',
        feature_dim=40
    )
    
    try:
        features = extractor_mfcc('path/to/audio.wav')
        print(f"MFCC 特征形状: {features.shape}")  # (40,)
    except Exception as e:
        print(f"提取失败: {e}")
