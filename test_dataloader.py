"""
数据加载器简单测试
测试单个视频的特征提取并保存结果
"""

import torch
from pathlib import Path
import time


def test_single_video():
    """测试单个视频的特征提取"""
    
    # 查找一个测试视频
    video_dir = Path('src/videos/cut')
    if not video_dir.exists():
        print("❌ 视频目录不存在: src/videos/cut")
        return
    
    video_files = list(video_dir.glob('*.mp4'))
    if not video_files:
        print("❌ 未找到视频文件")
        return
    
    # 使用第一个视频
    test_video = video_files[0]
    print(f"📹 测试视频: {test_video.name}\n")
    
    # 创建输出目录
    output_dir = Path('src/test_data')
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # ========== 测试 1: 人脸提取 ==========
    print("=" * 60)
    print("测试 1: 人脸特征提取")
    print("=" * 60)
    
    try:
        from dataloader import FaceExtractor
        
        extractor = FaceExtractor(
            method='opencv',
            target_size=(160, 160),
            num_frames=16,
            device='cpu'
        )
        
        print("开始提取人脸特征...")
        start_time = time.time()
        
        faces = extractor.extract_from_video(test_video, return_tensor=True)
        
        elapsed = time.time() - start_time
        
        # 保存结果
        output_path = output_dir / f'{test_video.name}_faces.pt'
        torch.save(faces, output_path)
        
        print(f"✅ 提取成功 (耗时: {elapsed:.2f}秒)")
        print(f"   形状: {faces.shape}")
        print(f"   数值范围: [{faces.min():.3f}, {faces.max():.3f}]")
        print(f"   保存到: {output_path}")
        
        results['faces'] = True
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        results['faces'] = False
        
        # 分析失败原因
        if "未检测到任何人脸" in str(e):
            print("\n💡 人脸检测失败原因分析:")
            print("   1. 视频质量问题（模糊、光照不足）")
            print("   2. 人脸角度问题（侧脸、低头）")
            print("   3. OpenCV 检测器精度有限")
            print("   建议: 使用 MTCNN 或 RetinaFace 检测器")
    
    # ========== 测试 2: 音频提取 ==========
    print("\n" + "=" * 60)
    print("测试 2: 音频特征提取 (Wav2Vec2)")
    print("=" * 60)
    
    try:
        from dataloader import AudioExtractor
        
        extractor = AudioExtractor(
            method='wav2vec2',
            device='cpu'
        )
        
        print("开始提取音频特征...")
        start_time = time.time()
        
        audios = extractor.extract_from_video(
            test_video,
            return_tensor=True,
            keep_audio=False
        )
        
        elapsed = time.time() - start_time
        
        # 保存结果
        output_path = output_dir / f'{test_video.name}_audios.pt'
        torch.save(audios, output_path)
        
        print(f"✅ 提取成功 (耗时: {elapsed:.2f}秒)")
        print(f"   形状: {audios.shape}")
        print(f"   数值范围: [{audios.min():.3f}, {audios.max():.3f}]")
        print(f"   保存到: {output_path}")
        
        results['audios'] = True
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        results['audios'] = False
    
    # ========== 测试 3: OpenFace 提取 ==========
    print("\n" + "=" * 60)
    print("测试 3: OpenFace 特征提取")
    print("=" * 60)
    
    try:
        from dataloader import OpenFaceExtractor
        
        extractor = OpenFaceExtractor(
            openface_path=None,
            num_frames=16,
            feature_dim=714
        )
        
        print("开始提取 OpenFace 特征...")
        start_time = time.time()
        
        openfaces = extractor.extract_from_video(test_video, return_tensor=True)
        
        elapsed = time.time() - start_time
        
        # 保存结果
        output_path = output_dir / f'{test_video.name}_openfaces.pt'
        torch.save(openfaces, output_path)
        
        print(f"✅ 提取成功 (耗时: {elapsed:.2f}秒)")
        print(f"   形状: {openfaces.shape}")
        print(f"   数值范围: [{openfaces.min():.3f}, {openfaces.max():.3f}]")
        print(f"   保存到: {output_path}")
        
        results['openfaces'] = True
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        results['openfaces'] = False
        
        # 分析失败原因
        print("\n💡 OpenFace 失败原因分析:")
        print("   1. OpenFace 未正确安装")
        print("   2. 路径配置错误")
        print("   3. 视频格式不兼容")
        print("   4. OpenFace 执行超时")
        print("\n   常见问题:")
        print("   - Windows: 需要下载预编译版本")
        print("   - 路径: 确保 FeatureExtraction.exe 可访问")
        print("   - 依赖: 需要 Visual C++ Redistributable")
        print("\n   解决方案:")
        print("   - 跳过 OpenFace，仅使用人脸图像和音频特征")
        print("   - 或手动指定 OpenFace 路径")
    
    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    print(f"输出目录: {output_dir.absolute()}")


if __name__ == '__main__':
    print("\n🔍 数据加载器简单测试\n")
    test_single_video()
    print("\n✨ 测试完成\n")

