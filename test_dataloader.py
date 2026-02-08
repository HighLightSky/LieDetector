"""
测试数据加载器
验证特征提取器是否正常工作
"""

import torch
import numpy as np
from pathlib import Path


def test_face_extractor():
    """测试面部图像提取器"""
    print("=" * 60)
    print("测试 FaceExtractor...")
    print("=" * 60)
    
    try:
        from dataloader import FaceExtractor
        
        # 创建提取器
        extractor = FaceExtractor(
            method='opencv',
            target_size=(160, 160),
            num_frames=16,
            device='cpu'
        )
        
        print("✅ FaceExtractor 初始化成功")
        print(f"   检测方法: {extractor.method}")
        print(f"   目标大小: {extractor.target_size}")
        print(f"   帧数: {extractor.num_frames}")
        
        # 测试人脸检测
        print("\n测试人脸检测功能...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = extractor.detect_face(dummy_frame)
        print(f"   检测结果: {bbox if bbox else '未检测到人脸（正常，这是随机图像）'}")
        
        return True
        
    except Exception as e:
        print(f"❌ FaceExtractor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openface_extractor():
    """测试 OpenFace 特征提取器"""
    print("\n" + "=" * 60)
    print("测试 OpenFaceExtractor...")
    print("=" * 60)
    
    try:
        from dataloader import OpenFaceExtractor
        
        # 创建提取器
        extractor = OpenFaceExtractor(
            openface_path=None,  # 自动查找
            num_frames=16,
            feature_dim=714,
            device='cpu'
        )
        
        print("✅ OpenFaceExtractor 初始化成功")
        print(f"   OpenFace 路径: {extractor.openface_path}")
        print(f"   帧数: {extractor.num_frames}")
        print(f"   特征维度: {extractor.feature_dim}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  OpenFaceExtractor 初始化失败: {e}")
        print("   这是正常的，如果未安装 OpenFace")
        return False


def test_audio_extractor():
    """测试音频特征提取器"""
    print("\n" + "=" * 60)
    print("测试 AudioExtractor...")
    print("=" * 60)
    
    try:
        from dataloader import AudioExtractor
        
        # 创建提取器 (Wav2Vec2)
        print("测试 Wav2Vec2 提取器...")
        extractor_w2v = AudioExtractor(
            method='wav2vec2',
            device='cpu'
        )
        
        print("✅ Wav2Vec2 AudioExtractor 初始化成功")
        print(f"   方法: {extractor_w2v.method}")
        print(f"   模型: {extractor_w2v.model_name}")
        print(f"   特征维度: {extractor_w2v.feature_dim}")
        
        # 创建提取器 (MFCC)
        print("\n测试 MFCC 提取器...")
        extractor_mfcc = AudioExtractor(
            method='mfcc',
            feature_dim=40
        )
        
        print("✅ MFCC AudioExtractor 初始化成功")
        print(f"   方法: {extractor_mfcc.method}")
        print(f"   特征维度: {extractor_mfcc.feature_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ AudioExtractor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """测试数据集类"""
    print("\n" + "=" * 60)
    print("测试 LieDetectionDataset...")
    print("=" * 60)
    
    try:
        from dataloader import LieDetectionDataset
        
        print("✅ LieDetectionDataset 导入成功")
        print("   注意: 需要准备标注文件和视频才能完整测试")
        
        # 测试 collate_fn
        print("\n测试 collate_fn...")
        batch = [
            {
                'faces': torch.randn(16, 3, 160, 160),
                'openfaces': torch.randn(16, 714),
                'audios': torch.randn(768),
                'label': torch.tensor(0),
                'video_id': 'video_001'
            },
            {
                'faces': torch.randn(16, 3, 160, 160),
                'openfaces': torch.randn(16, 714),
                'audios': torch.randn(768),
                'label': torch.tensor(1),
                'video_id': 'video_002'
            }
        ]
        
        collated = LieDetectionDataset.collate_fn(batch)
        
        print(f"   faces shape: {collated['faces'].shape}")
        print(f"   openfaces shape: {collated['openfaces'].shape}")
        print(f"   audios shape: {collated['audios'].shape}")
        print(f"   labels shape: {collated['labels'].shape}")
        print(f"   video_ids: {collated['video_ids']}")
        
        assert collated['faces'].shape == (2, 16, 3, 160, 160)
        assert collated['openfaces'].shape == (2, 16, 714)
        assert collated['audios'].shape == (2, 768)
        assert collated['labels'].shape == (2,)
        
        print("✅ collate_fn 测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ LieDetectionDataset 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_dataloader():
    """测试 create_dataloader 函数"""
    print("\n" + "=" * 60)
    print("测试 create_dataloader...")
    print("=" * 60)
    
    try:
        from dataloader import create_dataloader
        
        print("✅ create_dataloader 导入成功")
        print("   注意: 需要准备标注文件和视频才能完整测试")
        
        return True
        
    except Exception as e:
        print(f"❌ create_dataloader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n🔍 开始测试数据加载器模块...\n")
    
    results = {
        'FaceExtractor': test_face_extractor(),
        'OpenFaceExtractor': test_openface_extractor(),
        'AudioExtractor': test_audio_extractor(),
        'LieDetectionDataset': test_dataset(),
        'create_dataloader': test_create_dataloader()
    }
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:25s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "=" * 60)
    if passed == total:
        print(f"🎉 所有测试通过! ({passed}/{total})")
    else:
        print(f"⚠️  部分测试失败 ({passed}/{total})")
        print("\n注意:")
        print("- OpenFaceExtractor 失败是正常的，如果未安装 OpenFace")
        print("- 完整测试需要准备标注文件和视频数据")
    print("=" * 60)


if __name__ == '__main__':
    main()
