"""
谎言检测器
封装模型推理逻辑
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from models.fusion import FusionModel
from dataloader import FaceExtractor, AudioExtractor, OpenFaceExtractor


class LieDetector:
    """谎言检测器
    
    封装完整的推理流程：
    1. 特征提取
    2. 模型推理
    3. 结果解析
    """
    
    def __init__(
        self,
        model: FusionModel,
        face_extractor: FaceExtractor,
        audio_extractor: AudioExtractor,
        openface_extractor: Optional[OpenFaceExtractor] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 融合模型
            face_extractor: 人脸特征提取器
            audio_extractor: 音频特征提取器
            openface_extractor: OpenFace特征提取器（可选）
            device: 设备
        """
        self.model = model.to(device)
        self.face_extractor = face_extractor
        self.audio_extractor = audio_extractor
        self.openface_extractor = openface_extractor
        self.device = device
        
        # 标签映射
        self.label_map = {0: 'truth', 1: 'deception'}
        self.label_map_cn = {0: '说真话', 1: '说谎'}
    
    def extract_features(
        self,
        video_path: Path,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """从视频中提取特征
        
        Args:
            video_path: 视频文件路径
            verbose: 是否打印详细信息
        
        Returns:
            (faces, audios, openfaces)
        """
        if verbose:
            print("\n[特征提取阶段]")
        
        # 人脸特征
        if verbose:
            print("  提取人脸特征...")
        faces = self.face_extractor.extract_from_video(video_path, return_tensor=True)  # (T, C, H, W)
        faces = faces.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        if verbose:
            print(f"    ✓ 人脸特征: {faces.shape}")
        
        # 音频特征
        if verbose:
            print("  提取音频特征...")
        audios = self.audio_extractor.extract_from_video(video_path, return_tensor=True)  # (768,)
        audios = audios.unsqueeze(0).to(self.device)  # (1, 768)
        if verbose:
            print(f"    ✓ 音频特征: {audios.shape}")
        
        # OpenFace特征（可选）
        openfaces = None
        if self.openface_extractor is not None:
            try:
                if verbose:
                    print("  提取 OpenFace 特征...")
                openfaces = self.openface_extractor.extract_from_video(video_path, return_tensor=True)  # (T, 714)
                openfaces = openfaces.unsqueeze(0).to(self.device)  # (1, T, 714)
                if verbose:
                    print(f"    ✓ OpenFace 特征: {openfaces.shape}")
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  OpenFace 提取失败: {e}")
                openfaces = None
        
        return faces, audios, openfaces
    
    def predict(
        self,
        video_path: Path,
        verbose: bool = True
    ) -> Dict:
        """检测单个视频
        
        Args:
            video_path: 视频文件路径
            verbose: 是否打印详细信息
        
        Returns:
            dict: {
                'prediction': int (0: truth, 1: deception),
                'label': str ('truth' or 'deception'),
                'label_cn': str ('说真话' or '说谎'),
                'confidence': float,
                'probs': dict,
                'weights': dict,
                'details': dict
            }
        """
        video_path = Path(video_path)
        
        if verbose:
            print("\n" + "="*60)
            print(f"检测视频: {video_path.name}")
            print("="*60)
        
        # 1. 提取特征
        faces, audios, openfaces = self.extract_features(video_path, verbose=verbose)
        
        # 2. 模型推理
        if verbose:
            print("\n[模型推理阶段]")
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(faces, openfaces, audios)
        
        # 3. 解析结果
        if verbose:
            print("\n[结果解析]")
        
        # 融合概率
        fused_probs = output['probs']['fused'].cpu().numpy()[0]  # (2,)
        pred = int(np.argmax(fused_probs))
        confidence = float(fused_probs[pred])
        
        # 各模态概率
        probs_dict = {
            'face': output['probs']['face'].cpu().numpy()[0].tolist(),
            'openface': output['probs']['openface'].cpu().numpy()[0].tolist(),
            'audio': output['probs']['audio'].cpu().numpy()[0].tolist(),
            'fa_au': output['probs']['fa_au'].cpu().numpy()[0].tolist(),
            'fa_of': output['probs']['fa_of'].cpu().numpy()[0].tolist(),
            'fused': fused_probs.tolist()
        }
        
        # 模态权重
        weights = output['weights'].cpu().numpy()[0]  # (3,)
        weights_dict = {
            'face': float(weights[0]),
            'openface': float(weights[1]),
            'audio': float(weights[2])
        }
        
        # 余弦相似度
        cos_dict = {
            'fa_fa_au': float(output['cos']['fa_fa_au'].cpu().numpy()[0]),
            'fa_fa_of': float(output['cos']['fa_fa_of'].cpu().numpy()[0]),
            'au_fa_au': float(output['cos']['au_fa_au'].cpu().numpy()[0]),
            'of_fa_of': float(output['cos']['of_fa_of'].cpu().numpy()[0])
        }
        
        # 打印结果
        if verbose:
            print(f"\n  预测类别: {self.label_map_cn[pred]} ({self.label_map[pred]})")
            print(f"  置信度: {confidence:.2%}")
            print(f"\n  概率分布:")
            print(f"    Truth: {fused_probs[0]:.2%}")
            print(f"    Deception: {fused_probs[1]:.2%}")
            print(f"\n  模态权重:")
            print(f"    Face: {weights_dict['face']:.3f}")
            print(f"    OpenFace: {weights_dict['openface']:.3f}")
            print(f"    Audio: {weights_dict['audio']:.3f}")
            print("="*60 + "\n")
        
        return {
            'prediction': pred,
            'label': self.label_map[pred],
            'label_cn': self.label_map_cn[pred],
            'confidence': confidence,
            'probs': probs_dict,
            'weights': weights_dict,
            'details': {
                'cosine_similarities': cos_dict,
                'video_path': str(video_path)
            }
        }
    
    def predict_batch(
        self,
        video_paths: list,
        verbose: bool = False
    ) -> list:
        """批量检测视频
        
        Args:
            video_paths: 视频文件路径列表
            verbose: 是否打印详细信息
        
        Returns:
            list: 预测结果列表
        """
        results = []
        
        print(f"\n批量检测 {len(video_paths)} 个视频...")
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n[{i}/{len(video_paths)}] {Path(video_path).name}")
            try:
                result = self.predict(video_path, verbose=verbose)
                results.append(result)
                print(f"  ✓ {result['label_cn']} (置信度: {result['confidence']:.2%})")
            except Exception as e:
                print(f"  ✗ 检测失败: {e}")
                results.append({
                    'prediction': -1,
                    'label': 'error',
                    'label_cn': '错误',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        **kwargs
    ):
        """从检查点加载检测器
        
        Args:
            checkpoint_path: 检查点文件路径
            device: 设备
            **kwargs: 其他参数
        
        Returns:
            LieDetector实例
        """
        # 创建模型
        model = FusionModel(device=device, **kwargs)
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"✓ 从检查点加载模型: {checkpoint_path}")
        
        # 创建特征提取器
        face_extractor = FaceExtractor(
            method='opencv',
            target_size=(160, 160),
            num_frames=16,
            device='cpu'
        )
        
        audio_extractor = AudioExtractor(
            method='wav2vec2',
            device=device
        )
        
        try:
            openface_extractor = OpenFaceExtractor(
                openface_path=None,
                num_frames=16,
                device='cpu'
            )
        except Exception:
            openface_extractor = None
            print("⚠️  OpenFace 不可用")
        
        return cls(
            model=model,
            face_extractor=face_extractor,
            audio_extractor=audio_extractor,
            openface_extractor=openface_extractor,
            device=device
        )
