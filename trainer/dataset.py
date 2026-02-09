"""
谎言检测数据集
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import warnings


class LieDetectionDataset(Dataset):
    """谎言检测数据集
    
    从预计算的特征文件中加载数据
    """
    
    def __init__(
        self,
        csv_path: str,
        features_dir: str,
        use_openface: bool = True,
        transform=None
    ):
        """
        Args:
            csv_path: 标签 CSV 文件路径 (video_name, label)
            features_dir: 预计算特征目录
            use_openface: 是否使用 OpenFace 特征
            transform: 数据增强（可选）
        """
        self.csv_path = Path(csv_path)
        self.features_dir = Path(features_dir)
        self.use_openface = use_openface
        self.transform = transform
        
        # 加载标签
        self.df = pd.read_csv(csv_path)
        
        # 标签映射: truth=0, deception=1
        self.label_map = {'truth': 0, 'deception': 1}
        
        # 过滤掉特征文件不存在的样本
        self._filter_valid_samples()
        
        print(f"数据集加载完成:")
        print(f"  - 总样本数: {len(self.df)}")
        print(f"  - truth: {(self.df['label'] == 'truth').sum()}")
        print(f"  - deception: {(self.df['label'] == 'deception').sum()}")
    
    def _filter_valid_samples(self):
        """过滤掉特征文件不存在的样本"""
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            video_name = row['video_name']
            
            # 检查必需的特征文件（支持两种命名方式）
            face_path = self.features_dir / f"{video_name}_faces.pt"
            audio_path = self.features_dir / f"{video_name}_audios.pt"
            
            # 兼容旧命名
            if not audio_path.exists():
                audio_path = self.features_dir / f"{video_name}_audio.pt"
            
            if not face_path.exists():
                continue
            if not audio_path.exists():
                continue
            
            # 检查 OpenFace 特征（如果需要）
            if self.use_openface:
                openface_path = self.features_dir / f"{video_name}_openfaces.pt"
                if not openface_path.exists():
                    openface_path = self.features_dir / f"{video_name}_openface.pt"
                if not openface_path.exists():
                    continue
            
            valid_indices.append(idx)
        
        # 过滤
        original_len = len(self.df)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        if len(self.df) < original_len:
            warnings.warn(
                f"过滤掉 {original_len - len(self.df)} 个样本（特征文件不存在）"
            )
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """获取单个样本
        
        Returns:
            dict: {
                'faces': (T, C, H, W),
                'audios': (768,),
                'openfaces': (T, 714) or None,
                'label': int,
                'video_name': str
            }
        """
        row = self.df.iloc[idx]
        video_name = row['video_name']
        label = self.label_map[row['label']]
        
        # 加载特征（支持两种命名方式）
        face_path = self.features_dir / f"{video_name}_faces.pt"
        audio_path = self.features_dir / f"{video_name}_audios.pt"
        
        # 兼容旧命名
        if not audio_path.exists():
            audio_path = self.features_dir / f"{video_name}_audio.pt"
        
        faces = torch.load(face_path)  # (T, C, H, W)
        audios = torch.load(audio_path)  # (768,)
        
        # 加载 OpenFace 特征（如果需要）
        openfaces = None
        if self.use_openface:
            openface_path = self.features_dir / f"{video_name}_openfaces.pt"
            if not openface_path.exists():
                openface_path = self.features_dir / f"{video_name}_openface.pt"
            if openface_path.exists():
                openfaces = torch.load(openface_path)  # (T, 714)
        
        # 数据增强（可选）
        if self.transform is not None:
            faces = self.transform(faces)
        
        return {
            'faces': faces,
            'audios': audios,
            'openfaces': openfaces,
            'label': torch.tensor(label, dtype=torch.long),
            'video_name': video_name
        }
    
    @staticmethod
    def collate_fn(batch):
        """批处理函数
        
        处理不同长度的序列，使用零填充
        """
        # 找到最大序列长度
        max_face_len = max(item['faces'].size(0) for item in batch)
        max_openface_len = max(
            item['openfaces'].size(0) if item['openfaces'] is not None else 0
            for item in batch
        )
        
        # 批处理
        batch_size = len(batch)
        faces_list = []
        audios_list = []
        openfaces_list = []
        labels_list = []
        video_names = []
        
        for item in batch:
            # 人脸特征填充
            faces = item['faces']  # (T, C, H, W)
            if faces.size(0) < max_face_len:
                padding = torch.zeros(
                    max_face_len - faces.size(0),
                    *faces.shape[1:]
                )
                faces = torch.cat([faces, padding], dim=0)
            faces_list.append(faces)
            
            # 音频特征
            audios_list.append(item['audios'])
            
            # OpenFace 特征填充
            if item['openfaces'] is not None:
                openfaces = item['openfaces']  # (T, 714)
                if openfaces.size(0) < max_openface_len:
                    padding = torch.zeros(
                        max_openface_len - openfaces.size(0),
                        openfaces.size(1)
                    )
                    openfaces = torch.cat([openfaces, padding], dim=0)
                openfaces_list.append(openfaces)
            else:
                # 如果没有 OpenFace 特征，用零填充
                openfaces_list.append(torch.zeros(max_openface_len, 714))
            
            # 标签
            labels_list.append(item['label'])
            video_names.append(item['video_name'])
        
        return {
            'faces': torch.stack(faces_list),  # (B, T, C, H, W)
            'audios': torch.stack(audios_list),  # (B, 768)
            'openfaces': torch.stack(openfaces_list) if openfaces_list else None,  # (B, T, 714)
            'labels': torch.stack(labels_list),  # (B,)
            'video_names': video_names
        }
