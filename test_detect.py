"""
快速测试检测器
验证推理流程是否正常工作
"""

import torch
from pathlib import Path

from models.fusion import FusionModel
from detector import LieDetector
from dataloader import FaceExtractor, AudioExtractor, OpenFaceExtractor


def test_detector_with_dummy_data():
    """使用虚拟数据测试检测器"""
    print("\n" + "="*60)
    print("测试检测器 - 虚拟数据")
    print("="*60)
    
    device = 'cpu'
    
    # 1. 创建模型
    print("\n[1] 创建模型...")
    model = FusionModel(
        device=device,
        freeze_backbones=True,
        num_classes=2,
        visual_hidden=256,
        fusion_hidden=128,
        dropout=0.3
    )
    print("  ✓ 模型创建成功")
    
    # 2. 准备虚拟数据
    print("\n[2] 准备虚拟数据...")
    B, T = 1, 8
    faces = torch.randn(B, T, 3, 160, 160)
    openfaces = torch.randn(B, T, 714)
    audios = torch.randn(B, 768)
    print(f"  Faces: {faces.shape}")
    print(f"  OpenFaces: {openfaces.shape}")
    print(f"  Audios: {audios.shape}")
    
    # 3. 推理
    print("\n[3] 模型推理...")
    model.eval()
    with torch.no_grad():
        output = model(faces, openfaces, audios)
    
    # 4. 检查输出
    print("\n[4] 检查输出...")
    print(f"  输出键: {list(output.keys())}")
    
    # 融合概率
    fused_probs = output['probs']['fused'].cpu().numpy()[0]
    pred = int(fused_probs.argmax())
    confidence = float(fused_probs[pred])
    
    print(f"\n  预测: {'说谎' if pred == 1 else '说真话'}")
    print(f"  置信度: {confidence:.2%}")
    print(f"  概率: Truth={fused_probs[0]:.2%}, Deception={fused_probs[1]:.2%}")
    
    # 模态权重
    weights = output['weights'].cpu().numpy()[0]
    print(f"\n  模态权重:")
    print(f"    Face: {weights[0]:.3f}")
    print(f"    OpenFace: {weights[1]:.3f}")
    print(f"    Audio: {weights[2]:.3f}")
    
    print("\n  ✓ 检测器测试通过（虚拟数据）")


def test_detector_with_real_video():
    """使用真实视频测试检测器"""
    print("\n" + "="*60)
    print("测试检测器 - 真实视频")
    print("="*60)
    
    # 查找测试视频
    video_dir = Path('src/videos/cut')
    if not video_dir.exists():
        print("  ⚠️  未找到视频目录，跳过真实视频测试")
        return
    
    video_files = list(video_dir.glob('*.mp4'))
    if not video_files:
        print("  ⚠️  未找到视频文件，跳过真实视频测试")
        return
    
    test_video = video_files[0]
    print(f"\n  测试视频: {test_video.name}")
    
    device = 'cpu'
    
    # 1. 创建特征提取器
    print("\n[1] 创建特征提取器...")
    face_extractor = FaceExtractor(
        method='opencv',
        target_size=(160, 160),
        num_frames=8,  # 减少帧数加快测试
        device='cpu'
    )
    print("  ✓ 人脸提取器")
    
    audio_extractor = AudioExtractor(
        method='wav2vec2',
        device=device
    )
    print("  ✓ 音频提取器")
    
    try:
        openface_extractor = OpenFaceExtractor(
            openface_path=None,
            num_frames=8,
            device='cpu'
        )
        print("  ✓ OpenFace 提取器")
    except Exception as e:
        print(f"  ⚠️  OpenFace 不可用: {e}")
        openface_extractor = None
    
    # 2. 创建模型
    print("\n[2] 创建模型...")
    model = FusionModel(
        device=device,
        freeze_backbones=True,
        num_classes=2,
        visual_hidden=256,
        fusion_hidden=128,
        dropout=0.3
    )
    print("  ✓ 模型创建成功")
    
    # 3. 创建检测器
    print("\n[3] 创建检测器...")
    detector = LieDetector(
        model=model,
        face_extractor=face_extractor,
        audio_extractor=audio_extractor,
        openface_extractor=openface_extractor,
        device=device
    )
    print("  ✓ 检测器创建成功")
    
    # 4. 检测视频
    print("\n[4] 检测视频...")
    try:
        result = detector.predict(test_video, verbose=True)
        print("\n  ✓ 检测器测试通过（真实视频）")
    except Exception as e:
        print(f"\n  ✗ 检测失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("检测器测试脚本")
    print("="*60)
    
    try:
        # 测试1：虚拟数据
        test_detector_with_dummy_data()
        
        # 测试2：真实视频
        test_detector_with_real_video()
        
        print("\n" + "="*60)
        print("✨ 所有测试完成！")
        print("="*60)
        print("\n注意: 当前使用的是未训练的模型，预测结果是随机的。")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 测试失败: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
