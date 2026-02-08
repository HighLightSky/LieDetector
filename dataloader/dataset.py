"""
多模态谎言检测数据集
整合面部、OpenFace 和音频特征
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import warnings

from .face_extractor import FaceExtractor
from .openface_extractor import OpenFaceExtractor
from .audio_extractor import AudioExtractor


class LieDetectionDataset(Dataset):
    """多模态谎言检测数据集
    
    支持两种模式:
    1. 实时提取: 从视频实时提取特征 (慢，但灵活)
    2. 预提取: 从预先提取的特征文件加载 (快，推荐用于训练)
    
    Args:
        annotation_file: 标注文件路径 (CSV 格式)
        video_dir: 视频文件目录
        mode: 'realtime' 或 'precomputed'
        face_extractor: 面部提取器实例
        openface_extractor: OpenFace 提取器实例
        audio_extractor: 音频提取器实例
        precomputed_dir: 预提取特征目录 (mode='precomputed' 时使用)
        transform: 数据增强
    
    标注文件格式 (CSV):
        video_id,label,split
        video_001,0,train
        video_002,1,train
        video_003,0,val
        ...
        
        - video_id: 视频文件名 (不含扩展名)
        - label: 0=真话, 1=谎言
        - split: train/val/test (可选)
    """
    
    def __init__(
        self,
        annotation_file: Union[str, Path],
        video_dir: Optional[Union[str, Path]] = None,
        mode: str = 'realtime',
        face_extractor: Optional[FaceExtractor] = None,
        openface_extractor: Optional[OpenFaceExtractor] = None,
        audio_extractor: Optional[AudioExtractor] = None,
        precomputed_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Any] = None,
        split: Optional[str] = None
    ):
        self.annotation_file = Path(annotation_file)
        self.video_dir = Path(video_dir) if video_dir else None
        self.mode = mode
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        self.transform = transform
        
        # 读取标注
        self.annotations = pd.read_csv(self.annotation_file)
        
        # 过滤 split
        if split is not None and 'split' in self.annotations.columns:
            self.annotations = self.annotations[self.annotations['split'] == split]
        
        # 验证
        if len(self.annotations) == 0:
            raise ValueError("标注文件为空或 split 过滤后无数据")
        
        # 初始化提取器
        if mode == 'realtime':
            if video_dir is None:
                raise ValueError("realtime 模式需要指定 video_dir")
            
            self.face_extractor = face_extractor or FaceExtractor()
            self.openface_extractor = openface_extractor or OpenFaceExtractor()
            self.audio_extractor = audio_extractor or AudioExtractor()
        
        elif mode == 'precomputed':
            if precomputed_dir is None:
                raise ValueError("precomputed 模式需要指定 precomputed_dir")
        
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本
        
        Returns:
            字典包含:
            - faces: (T, 3, H, W)
            - openfaces: (T, 714)
            - audios: (768,)
            - label: (,) 标量
            - video_id: 字符串
        """
        row = self.annotations.iloc[idx]
        video_id = row['video_id']
        label = int(row['label'])
        
        if self.mode == 'realtime':
            # 实时提取特征
            sample = self._extract_realtime(video_id, label)
        else:
            # 加载预提取特征
            sample = self._load_precomputed(video_id, label)
        
        # 数据增强
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _extract_realtime(self, video_id: str, label: int) -> Dict[str, torch.Tensor]:
        """实时提取特征"""
        # 查找视频文件
        video_path = self._find_video_file(video_id)
        
        try:
            # 提取面部图像
            faces = self.face_extractor(video_path)
            
            # 提取 OpenFace 特征
            openfaces = self.openface_extractor(video_path)
            
            # 提取音频特征
            audios = self.audio_extractor(video_path, from_video=True)
            
        except Exception as e:
            warnings.warn(f"提取失败 {video_id}: {e}，使用零特征")
            # 返回零特征
            faces = torch.zeros(16, 3, 160, 160)
            openfaces = torch.zeros(16, 714)
            audios = torch.zeros(768)
        
        return {
            'faces': faces,
            'openfaces': openfaces,
            'audios': audios,
            'label': torch.tensor(label, dtype=torch.long),
            'video_id': video_id
        }
    
    def _load_precomputed(self, video_id: str, label: int) -> Dict[str, torch.Tensor]:
        """加载预提取特征"""
        feature_dir = self.precomputed_dir / video_id
        
        try:
            # 加载特征
            faces = torch.load(feature_dir / 'faces.pt')
            openfaces = torch.load(feature_dir / 'openfaces.pt')
            audios = torch.load(feature_dir / 'audios.pt')
            
        except Exception as e:
            warnings.warn(f"加载失败 {video_id}: {e}，使用零特征")
            # 返回零特征
            faces = torch.zeros(16, 3, 160, 160)
            openfaces = torch.zeros(16, 714)
            audios = torch.zeros(768)
        
        return {
            'faces': faces,
            'openfaces': openfaces,
            'audios': audios,
            'label': torch.tensor(label, dtype=torch.long),
            'video_id': video_id
        }
    
    def _find_video_file(self, video_id: str) -> Path:
        """查找视频文件"""
        # 常见视频扩展名
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        
        for ext in extensions:
            video_path = self.video_dir / f"{video_id}{ext}"
            if video_path.exists():
                return video_path
        
        raise FileNotFoundError(f"未找到视频文件: {video_id}")
    
    @staticmethod
    def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
        """自定义 collate 函数
        
        处理批次数据，确保所有样本的帧数一致
        """
        # 提取各个字段
        faces = torch.stack([item['faces'] for item in batch])
        openfaces = torch.stack([item['openfaces'] for item in batch])
        audios = torch.stack([item['audios'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        video_ids = [item['video_id'] for item in batch]
        
        return {
            'faces': faces,          # (B, T, 3, H, W)
            'openfaces': openfaces,  # (B, T, 714)
            'audios': audios,        # (B, 768)
            'labels': labels,        # (B,)
            'video_ids': video_ids   # List[str]
        }


def create_dataloader(
    annotation_file: Union[str, Path],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """创建数据加载器的便捷函数
    
    Args:
        annotation_file: 标注文件路径
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        **dataset_kwargs: 传递给 LieDetectionDataset 的参数
    
    Returns:
        DataLoader 实例
    """
    dataset = LieDetectionDataset(annotation_file, **dataset_kwargs)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=LieDetectionDataset.collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


# 示例用法
if __name__ == '__main__':
    # 方式 1: 实时提取模式
    dataset_realtime = LieDetectionDataset(
        annotation_file='data/annotations.csv',
        video_dir='data/videos',
        mode='realtime',
        split='train'
    )
    
    # 方式 2: 预提取模式
    dataset_precomputed = LieDetectionDataset(
        annotation_file='data/annotations.csv',
        mode='precomputed',
        precomputed_dir='data/features',
        split='train'
    )
    
    # 创建 DataLoader
    dataloader = create_dataloader(
        annotation_file='data/annotations.csv',
        video_dir='data/videos',
        mode='realtime',
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    # 迭代
    for batch in dataloader:
        faces = batch['faces']          # (8, 16, 3, 160, 160)
        openfaces = batch['openfaces']  # (8, 16, 714)
        audios = batch['audios']        # (8, 768)
        labels = batch['labels']        # (8,)
        
        print(f"Batch shapes:")
        print(f"  faces: {faces.shape}")
        print(f"  openfaces: {openfaces.shape}")
        print(f"  audios: {audios.shape}")
        print(f"  labels: {labels.shape}")
        break
