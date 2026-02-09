"""
谎言检测推理脚本
使用未训练的模型测试数据流，输出每层的形状用于调试
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from models.faces import FacesModel
from models.audio import AudioModel
from models.openface import OpenfaceModel
from dataloader import FaceExtractor, AudioExtractor, OpenFaceExtractor


class MultiModalFusionModel(nn.Module):
    """多模态融合模型
    
    整合人脸图像、音频和 OpenFace 特征进行谎言检测
    """
    def __init__(self, device='cpu', use_openface=True):
        super().__init__()
        self.device = device
        self.use_openface = use_openface
        
        # 三个子模型
        self.face_model = FacesModel(device=device)
        self.audio_model = AudioModel(use_mfcc=False)
        self.openface_model = OpenfaceModel(input_dim=714, num_classes=2) if use_openface else None
        
        # 融合层
        # FacesModel 输出: (B, 2)
        # AudioModel 输出: (B, 2)
        # OpenfaceModel 输出: (B*T, 2) -> 需要平均池化 -> (B, 2)
        fusion_input_dim = 2 + 2  # faces + audio
        if use_openface:
            fusion_input_dim += 2  # + openface
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 2)
        )
        
        self.to(device)
    
    def forward(self, faces, audios, openfaces=None, verbose=False):
        """
        Args:
            faces: 人脸图像特征 (B, T, C, H, W) 或 (B, C, H, W)
            audios: 音频特征 (B, 768)
            openfaces: OpenFace 特征 (B, T, 714) 或 None
            verbose: 是否打印每层的形状
        
        Returns:
            logits: 分类 logits (B, 2)
        """
        batch_size = faces.size(0)
        
        if verbose:
            print("\n" + "="*60)
            print("多模态融合模型 - 前向传播")
            print("="*60)
        
        # 1. 处理人脸特征
        if verbose:
            print(f"\n[1] 人脸特征输入: {faces.shape}")
        
        # 如果是视频序列 (B, T, C, H, W)，需要展平
        if faces.dim() == 5:
            B, T, C, H, W = faces.shape
            faces = faces.view(B * T, C, H, W)
            if verbose:
                print(f"    展平为: {faces.shape}")
        
        # 通过人脸模型
        face_logits = self.face_model(faces)  # (B*T, 2) 或 (B, 2)
        if verbose:
            print(f"    人脸模型输出: {face_logits.shape}")
        
        # 如果是序列，平均池化
        if face_logits.size(0) != batch_size:
            face_logits = face_logits.view(batch_size, -1, 2).mean(dim=1)  # (B, 2)
            if verbose:
                print(f"    平均池化后: {face_logits.shape}")
        
        # 2. 处理音频特征
        if verbose:
            print(f"\n[2] 音频特征输入: {audios.shape}")
        
        audio_logits = self.audio_model(audios)  # (B, 2)
        if verbose:
            print(f"    音频模型输出: {audio_logits.shape}")
        
        # 3. 处理 OpenFace 特征（如果有）
        if self.use_openface and openfaces is not None:
            if verbose:
                print(f"\n[3] OpenFace 特征输入: {openfaces.shape}")
            
            # 如果是序列 (B, T, 714)，需要展平
            if openfaces.dim() == 3:
                B, T, D = openfaces.shape
                openfaces = openfaces.view(B * T, D)
                if verbose:
                    print(f"    展平为: {openfaces.shape}")
            
            openface_logits = self.openface_model(openfaces)  # (B*T, 2)
            if verbose:
                print(f"    OpenFace 模型输出: {openface_logits.shape}")
            
            # 平均池化
            if openface_logits.size(0) != batch_size:
                openface_logits = openface_logits.view(batch_size, -1, 2).mean(dim=1)  # (B, 2)
                if verbose:
                    print(f"    平均池化后: {openface_logits.shape}")
            
            # 拼接所有特征
            fused = torch.cat([face_logits, audio_logits, openface_logits], dim=1)  # (B, 6)
        else:
            # 只使用人脸和音频
            fused = torch.cat([face_logits, audio_logits], dim=1)  # (B, 4)
        
        if verbose:
            print(f"\n[4] 特征融合: {fused.shape}")
        
        # 4. 融合层
        output = self.fusion(fused)  # (B, 2)
        if verbose:
            print(f"    融合层输出: {output.shape}")
            print("="*60 + "\n")
        
        return output


def detect_video(
    video_path: str,
    model: MultiModalFusionModel,
    face_extractor: FaceExtractor,
    audio_extractor: AudioExtractor,
    openface_extractor: OpenFaceExtractor = None,
    device: str = 'cpu',
    verbose: bool = True
):
    """检测单个视频
    
    Args:
        video_path: 视频文件路径
        model: 融合模型
        face_extractor: 人脸特征提取器
        audio_extractor: 音频特征提取器
        openface_extractor: OpenFace 特征提取器（可选）
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        prediction: 预测结果 (0: truth, 1: deception)
        probability: 预测概率
    """
    video_path = Path(video_path)
    
    if verbose:
        print("\n" + "="*60)
        print(f"检测视频: {video_path.name}")
        print("="*60)
    
    # 1. 提取特征
    if verbose:
        print("\n[特征提取阶段]")
    
    # 人脸特征
    if verbose:
        print("  提取人脸特征...")
    faces = face_extractor.extract_from_video(video_path, return_tensor=True)  # (T, C, H, W)
    faces = faces.unsqueeze(0).to(device)  # (1, T, C, H, W)
    if verbose:
        print(f"    ✓ 人脸特征: {faces.shape}")
    
    # 音频特征
    if verbose:
        print("  提取音频特征...")
    audios = audio_extractor.extract_from_video(video_path, return_tensor=True)  # (768,)
    audios = audios.unsqueeze(0).to(device)  # (1, 768)
    if verbose:
        print(f"    ✓ 音频特征: {audios.shape}")
    
    # OpenFace 特征（可选）
    openfaces = None
    if openface_extractor is not None:
        try:
            if verbose:
                print("  提取 OpenFace 特征...")
            openfaces = openface_extractor.extract_from_video(video_path, return_tensor=True)  # (T, 714)
            openfaces = openfaces.unsqueeze(0).to(device)  # (1, T, 714)
            if verbose:
                print(f"    ✓ OpenFace 特征: {openfaces.shape}")
        except Exception as e:
            if verbose:
                print(f"    ⚠️  OpenFace 提取失败: {e}")
            openfaces = None
    
    # 2. 模型推理
    if verbose:
        print("\n[模型推理阶段]")
    
    model.eval()
    with torch.no_grad():
        logits = model(faces, audios, openfaces, verbose=verbose)  # (1, 2)
    
    # 3. 结果
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (2,)
    pred = int(np.argmax(probs))
    
    if verbose:
        print("\n[预测结果]")
        print(f"  预测类别: {'deception (说谎)' if pred == 1 else 'truth (说真话)'}")
        print(f"  置信度: {probs[pred]:.2%}")
        print(f"  概率分布: truth={probs[0]:.2%}, deception={probs[1]:.2%}")
        print("="*60 + "\n")
    
    return pred, probs


def main():
    """主函数：测试数据流"""
    print("\n" + "="*60)
    print("谎言检测系统 - 数据流测试")
    print("="*60)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 1. 初始化提取器
    print("\n[1] 初始化特征提取器...")
    face_extractor = FaceExtractor(
        method='opencv',
        target_size=(160, 160),
        num_frames=16,
        device='cpu'
    )
    print("  ✓ 人脸提取器")
    
    audio_extractor = AudioExtractor(
        method='wav2vec2',
        device=device
    )
    print("  ✓ 音频提取器")
    
    # OpenFace 提取器（可选）
    try:
        openface_extractor = OpenFaceExtractor(
            openface_path=None,  # 自动查找
            num_frames=16,
            device='cpu'
        )
        print("  ✓ OpenFace 提取器")
        use_openface = True
    except Exception as e:
        print(f"  ⚠️  OpenFace 不可用: {e}")
        openface_extractor = None
        use_openface = False
    
    # 2. 初始化模型（未训练）
    print("\n[2] 初始化融合模型（未训练）...")
    model = MultiModalFusionModel(device=device, use_openface=use_openface)
    print(f"  ✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 选择测试视频
    print("\n[3] 选择测试视频...")
    video_dir = Path('src/videos/cut')
    video_files = list(video_dir.glob('*.mp4'))
    
    if not video_files:
        print("  ✗ 未找到测试视频")
        return
    
    # 选择第一个视频
    test_video = video_files[0]
    print(f"  ✓ 测试视频: {test_video.name}")
    
    # 4. 运行检测
    print("\n[4] 运行检测...")
    pred, probs = detect_video(
        video_path=test_video,
        model=model,
        face_extractor=face_extractor,
        audio_extractor=audio_extractor,
        openface_extractor=openface_extractor,
        device=device,
        verbose=True
    )
    
    print("\n✨ 数据流测试完成！")
    print("\n注意: 当前使用的是未训练的模型，预测结果是随机的。")
    print("      需要训练模型后才能获得准确的预测结果。")


if __name__ == '__main__':
    main()
