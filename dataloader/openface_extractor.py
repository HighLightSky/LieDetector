"""
OpenFace 特征提取器
使用 OpenFace 工具提取面部动作单元 (AU) 和关键点特征
"""

import subprocess
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, List
import warnings
import tempfile
import shutil


class OpenFaceExtractor:
    """使用 OpenFace 提取面部特征
    
    需要预先安装 OpenFace 工具:
    https://github.com/TadasBaltrusaitis/OpenFace
    
    Args:
        openface_path: OpenFace 可执行文件路径
        num_frames: 要采样的帧数，None 表示使用所有帧
        feature_dim: 特征维度 (默认 714)
        device: 设备 ('cpu' 或 'cuda')
    """
    
    def __init__(
        self,
        openface_path: Optional[str] = None,
        num_frames: Optional[int] = 16,
        feature_dim: int = 714,
        device: str = 'cpu'
    ):
        self.openface_path = openface_path or self._find_openface()
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.device = device
        
        # 验证 OpenFace 是否可用
        self._verify_openface()
    
    def _find_openface(self) -> str:
        """自动查找 OpenFace 可执行文件"""
        # 常见的 OpenFace 安装路径
        possible_paths = [
            'FeatureExtraction',  # 在 PATH 中
            'D:/soft/OpenFace/FeatureExtraction.exe',
            './OpenFace/build/bin/FeatureExtraction',
            '/usr/local/bin/FeatureExtraction',
        ]
        
        for path in possible_paths:
            if shutil.which(path):
                return path
        
        raise RuntimeError(
            "未找到 OpenFace 可执行文件。请:\n"
            "1. 安装 OpenFace: https://github.com/TadasBaltrusaitis/OpenFace\n"
            "2. 或指定 openface_path 参数"
        )
    
    def _verify_openface(self):
        """验证 OpenFace 是否可用"""
        try:
            result = subprocess.run(
                [self.openface_path, '-h'],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("OpenFace 执行失败")
        except Exception as e:
            raise RuntimeError(f"无法运行 OpenFace: {e}")
    
    def extract_from_video(
        self,
        video_path: Union[str, Path],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """从视频中提取 OpenFace 特征
        
        Args:
            video_path: 视频文件路径
            return_tensor: 是否返回 PyTorch tensor
        
        Returns:
            OpenFace 特征序列
            - 如果 return_tensor=True: torch.Tensor (T, 714)
            - 如果 return_tensor=False: np.ndarray (T, 714)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # 运行 OpenFace
            try:
                subprocess.run(
                    [
                        self.openface_path,
                        '-f', str(video_path),
                        '-out_dir', str(temp_dir),
                        '-2Dfp',  # 2D 面部关键点
                        '-3Dfp',  # 3D 面部关键点
                        '-pdmparams',  # PDM 参数
                        '-pose',  # 头部姿态
                        '-aus',  # 动作单元
                        '-gaze',  # 眼睛注视
                    ],
                    check=True,
                    capture_output=True,
                    timeout=300  # 5分钟超时
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"OpenFace 执行失败: {e.stderr.decode()}")
            except subprocess.TimeoutExpired:
                raise RuntimeError("OpenFace 执行超时")
            
            # 读取输出的 CSV 文件
            csv_files = list(temp_dir.glob('*.csv'))
            if len(csv_files) == 0:
                raise RuntimeError("OpenFace 未生成输出文件")
            
            csv_path = csv_files[0]
            df = pd.read_csv(csv_path)
            
            # 提取特征
            features = self._extract_features_from_df(df)
        
        # 采样帧
        if self.num_frames is not None and len(features) > self.num_frames:
            indices = np.linspace(0, len(features) - 1, self.num_frames, dtype=int)
            features = features[indices]
        
        if return_tensor:
            features = torch.from_numpy(features).float()
        
        return features
    
    def extract_from_csv(
        self,
        csv_path: Union[str, Path],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """从 OpenFace 输出的 CSV 文件中提取特征
        
        Args:
            csv_path: OpenFace 输出的 CSV 文件路径
            return_tensor: 是否返回 PyTorch tensor
        
        Returns:
            OpenFace 特征序列
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
        
        df = pd.read_csv(csv_path)
        features = self._extract_features_from_df(df)
        
        # 采样帧
        if self.num_frames is not None and len(features) > self.num_frames:
            indices = np.linspace(0, len(features) - 1, self.num_frames, dtype=int)
            features = features[indices]
        
        if return_tensor:
            features = torch.from_numpy(features).float()
        
        return features
    
    def _extract_features_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """从 DataFrame 中提取特征向量
        
        OpenFace 输出包含:
        - 68 个 2D 面部关键点 (x_0 到 x_67, y_0 到 y_67): 136 维
        - 68 个 3D 面部关键点 (X_0 到 X_67, Y_0 到 Y_67, Z_0 到 Z_67): 204 维
        - 头部姿态 (pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz): 6 维
        - 眼睛注视 (gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z): 6 维
        - 动作单元强度 (AU01_r 到 AU45_r): ~17 维
        - 动作单元存在 (AU01_c 到 AU45_c): ~17 维
        
        总计约 714 维
        """
        feature_columns = []
        
        # 1. 2D 面部关键点 (136 维)
        for i in range(68):
            feature_columns.extend([f'x_{i}', f'y_{i}'])
        
        # 2. 3D 面部关键点 (204 维)
        for i in range(68):
            feature_columns.extend([f'X_{i}', f'Y_{i}', f'Z_{i}'])
        
        # 3. 头部姿态 (6 维)
        feature_columns.extend([
            'pose_Tx', 'pose_Ty', 'pose_Tz',
            'pose_Rx', 'pose_Ry', 'pose_Rz'
        ])
        
        # 4. 眼睛注视 (6 维)
        feature_columns.extend([
            'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
            'gaze_1_x', 'gaze_1_y', 'gaze_1_z'
        ])
        
        # 5. 动作单元强度 (AU_r)
        au_intensity_cols = [col for col in df.columns if col.endswith('_r') and col.startswith('AU')]
        feature_columns.extend(sorted(au_intensity_cols))
        
        # 6. 动作单元存在 (AU_c)
        au_presence_cols = [col for col in df.columns if col.endswith('_c') and col.startswith('AU')]
        feature_columns.extend(sorted(au_presence_cols))
        
        # 过滤掉不存在的列
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if len(available_columns) == 0:
            raise RuntimeError("CSV 文件中未找到有效的特征列")
        
        # 提取特征
        features = df[available_columns].values.astype(np.float32)
        
        # 处理缺失值
        if np.isnan(features).any():
            warnings.warn("特征中包含 NaN 值，将使用 0 填充")
            features = np.nan_to_num(features, nan=0.0)
        
        # 如果特征维度不足 714，用 0 填充
        if features.shape[1] < self.feature_dim:
            padding = np.zeros((features.shape[0], self.feature_dim - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > self.feature_dim:
            # 如果超过，截断
            features = features[:, :self.feature_dim]
        
        return features
    
    def batch_extract(
        self,
        video_paths: List[Union[str, Path]],
        return_tensor: bool = True
    ) -> List[Union[torch.Tensor, np.ndarray]]:
        """批量提取多个视频的特征
        
        Args:
            video_paths: 视频文件路径列表
            return_tensor: 是否返回 PyTorch tensor
        
        Returns:
            特征列表
        """
        features_list = []
        
        for video_path in video_paths:
            try:
                features = self.extract_from_video(video_path, return_tensor)
                features_list.append(features)
            except Exception as e:
                warnings.warn(f"提取失败 {video_path}: {e}")
                # 返回零特征
                if return_tensor:
                    features_list.append(torch.zeros(self.num_frames or 1, self.feature_dim))
                else:
                    features_list.append(np.zeros((self.num_frames or 1, self.feature_dim), dtype=np.float32))
        
        return features_list
    
    def __call__(
        self,
        source: Union[str, Path],
        **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """便捷调用接口
        
        Args:
            source: 视频文件路径或 CSV 文件路径
            **kwargs: 传递给提取方法
        
        Returns:
            OpenFace 特征序列
        """
        source = Path(source)
        
        if source.suffix.lower() == '.csv':
            return self.extract_from_csv(source, **kwargs)
        else:
            return self.extract_from_video(source, **kwargs)


# 示例用法
if __name__ == '__main__':
    # 创建提取器
    extractor = OpenFaceExtractor(
        openface_path='FeatureExtraction',  # 或指定完整路径
        num_frames=16
    )
    
    # 从视频提取
    try:
        features = extractor('path/to/video.mp4')
        print(f"提取的 OpenFace 特征形状: {features.shape}")  # (16, 714)
    except Exception as e:
        print(f"提取失败: {e}")
    
    # 从 CSV 提取
    try:
        features = extractor.extract_from_csv('path/to/openface_output.csv')
        print(f"从 CSV 提取的特征形状: {features.shape}")
    except Exception as e:
        print(f"提取失败: {e}")
