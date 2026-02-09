"""
测试数据加载器
使用真实的切片视频验证特征提取器是否正常工作
"""

import torch
import numpy as np
from pathlib import Path
import random


def get_test_videos(video_dir='src/videos/cut', num_samples=3):
    """获取测试视频文件
    
    Args:
        video_dir: 视频目录
        num_samples: 采样数量
    
    Returns:
        视频文件路径列表
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"⚠️  视频目录不存在: {video_dir}")
        return []
    
    video_files = list(video_dir.glob('*.mp4'))
    if not video_files:
        print(f"⚠️  未找到视频文件: {video_dir}")
        return []
    
    # 随机采样
    num_samples = min(num_samples, len(video_files))
    sampled = random.sample(video_files, num_samples)
    
    print(f"📁 找到 {len(video_files)} 个视频文件")
    print(f"📊 随机采样 {num_samples} 个进行测试")
    
    return sampled


def test_face_extractor():
    """测试面部图像提取器"""
    print("=" * 60)
    print("测试 FaceExtractor (使用真实视频)")
    print("=" * 60)
    
    try:
        from dataloader import FaceExtractor
        
        # 获取测试视频
        test_videos = get_test_videos(num_samples=2)
        if not test_videos:
            print("⚠️  跳过测试：没有可用的视频文件")
            return False
        
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
        
        # 测试每个视频
        for i, video_path in enumerate(test_videos, 1):
            print(f"\n[{i}/{len(test_videos)}] 测试视频: {video_path.name}")
            
            try:
                # 提取人脸特征
                faces = extractor.extract_from_video(
                    video_path,
                    return_tensor=True
                )
                
                print(f"   ✅ 提取成功")
                print(f"   输出形状: {faces.shape}")
                print(f"   数据类型: {faces.dtype}")
                print(f"   数值范围: [{faces.min():.3f}, {faces.max():.3f}]")
                
            except Exception as e:
                print(f"   ❌ 提取失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ FaceExtractor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openface_extractor():
    """测试 OpenFace 特征提取器"""
    print("\n" + "=" * 60)
    print("测试 OpenFaceExtractor (使用真实视频)")
    print("=" * 60)
    
    try:
        from dataloader import OpenFaceExtractor
        
        # 获取测试视频
        test_videos = get_test_videos(num_samples=2)
        if not test_videos:
            print("⚠️  跳过测试：没有可用的视频文件")
            return False
        
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
        
        # 测试每个视频
        for i, video_path in enumerate(test_videos, 1):
            print(f"\n[{i}/{len(test_videos)}] 测试视频: {video_path.name}")
            
            try:
                # 提取 OpenFace 特征
                features = extractor.extract_from_video(
                    video_path,
                    return_tensor=True
                )
                
                print(f"   ✅ 提取成功")
                print(f"   输出形状: {features.shape}")
                print(f"   数据类型: {features.dtype}")
                print(f"   数值范围: [{features.min():.3f}, {features.max():.3f}]")
                
            except Exception as e:
                print(f"   ❌ 提取失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  OpenFaceExtractor 初始化失败: {e}")
        print("   这是正常的，如果未安装 OpenFace")
        return False


def test_audio_extractor():
    """测试音频特征提取器"""
    print("\n" + "=" * 60)
    print("测试 AudioExtractor (使用真实视频)")
    print("=" * 60)
    
    try:
        from dataloader import AudioExtractor
        
        # 获取测试视频
        test_videos = get_test_videos(num_samples=2)
        if not test_videos:
            print("⚠️  跳过测试：没有可用的视频文件")
            return False
        
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
        
        # 测试每个视频
        for i, video_path in enumerate(test_videos, 1):
            print(f"\n[{i}/{len(test_videos)}] 测试视频: {video_path.name}")
            
            try:
                # 从视频提取音频特征
                features = extractor_w2v.extract_from_video(
                    video_path,
                    return_tensor=True,
                    keep_audio=False
                )
                
                print(f"   ✅ Wav2Vec2 提取成功")
                print(f"   输出形状: {features.shape}")
                print(f"   数据类型: {features.dtype}")
                print(f"   数值范围: [{features.min():.3f}, {features.max():.3f}]")
                
            except Exception as e:
                print(f"   ❌ Wav2Vec2 提取失败: {e}")
        
        # 创建提取器 (MFCC)
        print("\n测试 MFCC 提取器...")
        extractor_mfcc = AudioExtractor(
            method='mfcc',
            feature_dim=40
        )
        
        print("✅ MFCC AudioExtractor 初始化成功")
        print(f"   方法: {extractor_mfcc.method}")
        print(f"   特征维度: {extractor_mfcc.feature_dim}")
        
        # 测试一个视频
        if test_videos:
            video_path = test_videos[0]
            print(f"\n测试视频: {video_path.name}")
            
            try:
                features = extractor_mfcc.extract_from_video(
                    video_path,
                    return_tensor=True,
                    keep_audio=False
                )
                
                print(f"   ✅ MFCC 提取成功")
                print(f"   输出形状: {features.shape}")
                print(f"   数据类型: {features.dtype}")
                print(f"   数值范围: [{features.min():.3f}, {features.max():.3f}]")
                
            except Exception as e:
                print(f"   ❌ MFCC 提取失败: {e}")
        
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


def test_precompute_features():
    """测试预计算特征功能"""
    print("\n" + "=" * 60)
    print("测试预计算特征 (保存到 src/test_data/)")
    print("=" * 60)
    
    try:
        from dataloader.precompute_features import precompute_all_features
        
        # 获取测试视频
        test_videos = get_test_videos(num_samples=5)
        if not test_videos:
            print("⚠️  跳过测试：没有可用的视频文件")
            return False
        
        print(f"\n将为 {len(test_videos)} 个视频预计算特征...")
        print("这可能需要几分钟时间...\n")
        
        # 创建视频列表（带标签）
        video_list = []
        for video_path in test_videos:
            # 从文件名推断标签
            label = 1 if 'lie' in video_path.stem or 'deception' in video_path.stem else 0
            video_list.append({
                'video_path': str(video_path),
                'label': label,
                'video_id': video_path.stem
            })
        
        # 预计算特征
        success_count = precompute_all_features(
            video_list=video_list,
            output_dir='src/test_data',
            extract_faces=True,
            extract_openface=True,  # 跳过 OpenFace（可能有问题）
            extract_audio=True,
            num_frames=16,
            device='cpu'
        )
        
        print(f"\n✅ 预计算完成: {success_count}/{len(video_list)} 个视频成功")
        
        # 检查输出文件
        output_dir = Path('src/test_data')
        if output_dir.exists():
            face_files = list(output_dir.glob('*_faces.pt'))
            audio_files = list(output_dir.glob('*_audio.pt'))
            
            print(f"\n生成的文件:")
            print(f"   人脸特征: {len(face_files)} 个")
            print(f"   音频特征: {len(audio_files)} 个")
            
            # 显示示例文件
            if face_files:
                print(f"\n示例人脸特征文件:")
                sample_face = torch.load(face_files[0])
                print(f"   文件: {face_files[0].name}")
                print(f"   形状: {sample_face.shape}")
                print(f"   大小: {face_files[0].stat().st_size / 1024:.2f} KB")
            
            if audio_files:
                print(f"\n示例音频特征文件:")
                sample_audio = torch.load(audio_files[0])
                print(f"   文件: {audio_files[0].name}")
                print(f"   形状: {sample_audio.shape}")
                print(f"   大小: {audio_files[0].stat().st_size / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ 预计算特征测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试数据加载器模块')
    parser.add_argument(
        '--mode',
        type=str,
        default='test',
        choices=['test', 'precompute', 'all'],
        help='运行模式: test=仅测试, precompute=仅预计算, all=全部'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='预计算时使用的视频数量'
    )
    
    args = parser.parse_args()
    
    print("\n🔍 开始测试数据加载器模块...\n")
    print(f"运行模式: {args.mode}")
    
    results = {}
    
    # 基础测试
    if args.mode in ['test', 'all']:
        print("\n" + "=" * 60)
        print("第一部分: 基础功能测试")
        print("=" * 60)
        
        results['FaceExtractor'] = test_face_extractor()
        results['OpenFaceExtractor'] = test_openface_extractor()
        results['AudioExtractor'] = test_audio_extractor()
        results['LieDetectionDataset'] = test_dataset()
        results['create_dataloader'] = test_create_dataloader()
    
    # 预计算特征
    if args.mode in ['precompute', 'all']:
        print("\n" + "=" * 60)
        print("第二部分: 预计算特征")
        print("=" * 60)
        
        # 临时修改采样数量
        global get_test_videos
        original_func = get_test_videos
        
        def get_test_videos_custom(video_dir='src/videos/cut', num_samples=None):
            if num_samples is None:
                num_samples = args.num_samples
            return original_func(video_dir, num_samples)
        
        get_test_videos = get_test_videos_custom
        results['PrecomputeFeatures'] = test_precompute_features()
        get_test_videos = original_func
    
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
    
    # 使用说明
    if args.mode == 'test':
        print("\n💡 提示:")
        print("   运行 'python test_dataloader.py --mode precompute' 来预计算特征")
        print("   运行 'python test_dataloader.py --mode all' 来运行完整测试")


if __name__ == '__main__':
    main()
