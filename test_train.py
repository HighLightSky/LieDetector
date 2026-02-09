"""
测试训练脚本 - 小批量训练
用于验证训练流程是否正常工作
"""

import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys

from detect import MultiModalFusionModel
from trainer import MultiModalTrainer, LieDetectionDataset


def test_train():
    """测试小批量训练"""
    print("\n" + "="*60)
    print("测试训练流程 - 小批量数据")
    print("="*60)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 1. 检查数据
    print("\n[1] 检查数据...")
    features_dir = Path('src/test_data')
    csv_path = Path('src/dataset/video_labels.csv')
    
    if not csv_path.exists():
        print(f"✗ 标签文件不存在: {csv_path}")
        print("  请先生成标签文件:")
        print("  python run.py")
        return
    
    # 检查特征文件数量
    face_files = list(features_dir.glob('*_faces.pt'))
    audio_files = list(features_dir.glob('*_audios.pt'))
    openface_files = list(features_dir.glob('*_openfaces.pt'))
    
    print(f"  ✓ 特征目录: {features_dir}")
    print(f"  ✓ 人脸特征: {len(face_files)} 个")
    print(f"  ✓ 音频特征: {len(audio_files)} 个")
    print(f"  ✓ OpenFace 特征: {len(openface_files)} 个")
    
    # 如果特征文件不足，提取一些测试特征
    if len(face_files) < 5 or len(audio_files) < 5:
        print("\n  特征文件不足，正在提取测试特征...")
        from dataloader import FaceExtractor, AudioExtractor, OpenFaceExtractor
        
        # 创建提取器
        face_extractor = FaceExtractor(num_frames=16, device='cpu')
        audio_extractor = AudioExtractor(device=device)
        try:
            openface_extractor = OpenFaceExtractor(num_frames=16, device='cpu')
            use_openface = True
        except:
            openface_extractor = None
            use_openface = False
        
        # 获取前5个视频
        video_dir = Path('src/videos/cut')
        video_files = list(video_dir.glob('*.mp4'))[:5]
        
        if len(video_files) == 0:
            print("  ✗ 未找到视频文件")
            return
        
        features_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  提取 {len(video_files)} 个视频的特征...")
        for i, video_path in enumerate(video_files, 1):
            video_name = video_path.stem
            print(f"    [{i}/{len(video_files)}] {video_name}")
            
            try:
                # 人脸特征
                faces = face_extractor.extract_from_video(video_path, return_tensor=True)
                torch.save(faces, features_dir / f"{video_name}_faces.pt")
                
                # 音频特征
                audios = audio_extractor.extract_from_video(video_path, return_tensor=True)
                torch.save(audios, features_dir / f"{video_name}_audios.pt")
                
                # OpenFace 特征
                if openface_extractor is not None:
                    try:
                        openfaces = openface_extractor.extract_from_video(video_path, return_tensor=True, save_csv=False)
                        torch.save(openfaces, features_dir / f"{video_name}_openfaces.pt")
                    except:
                        pass
                
                print(f"      ✓ 完成")
            except Exception as e:
                print(f"      ✗ 失败: {e}")
        
        # 重新统计
        face_files = list(features_dir.glob('*_faces.pt'))
        audio_files = list(features_dir.glob('*_audios.pt'))
        openface_files = list(features_dir.glob('*_openfaces.pt'))
        
        print(f"\n  提取完成:")
        print(f"    人脸特征: {len(face_files)} 个")
        print(f"    音频特征: {len(audio_files)} 个")
        print(f"    OpenFace 特征: {len(openface_files)} 个")
    
    if len(face_files) == 0 or len(audio_files) == 0:
        print("\n✗ 没有足够的特征文件进行训练")
        return
    
    # 2. 创建数据集
    print("\n[2] 创建数据集...")
    use_openface = len(openface_files) > 0
    
    try:
        dataset = LieDetectionDataset(
            csv_path=csv_path,
            features_dir=features_dir,
            use_openface=use_openface
        )
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        return
    
    if len(dataset) == 0:
        print("✗ 数据集为空")
        return
    
    # 3. 创建小批量数据集（用于测试）
    print("\n[3] 创建小批量数据集...")
    
    # 只使用前 10 个样本（如果有的话）
    num_samples = min(10, len(dataset))
    indices = list(range(num_samples))
    
    # 划分训练集和验证集 (8:2)
    split_idx = int(0.8 * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    if len(train_indices) == 0 or len(val_indices) == 0:
        print(f"✗ 样本数量不足 (总共 {num_samples} 个)")
        print("  至少需要 2 个样本（1个训练，1个验证）")
        return
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"  ✓ 训练集: {len(train_dataset)} 个样本")
    print(f"  ✓ 验证集: {len(val_dataset)} 个样本")
    
    # 4. 创建数据加载器
    print("\n[4] 创建数据加载器...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # 小批量
        shuffle=True,
        collate_fn=LieDetectionDataset.collate_fn,
        num_workers=0  # Windows 上设置为 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=LieDetectionDataset.collate_fn,
        num_workers=0
    )
    
    print(f"  ✓ 训练批次: {len(train_loader)}")
    print(f"  ✓ 验证批次: {len(val_loader)}")
    
    # 5. 创建模型
    print("\n[5] 创建模型...")
    
    model = MultiModalFusionModel(
        device=device,
        use_openface=use_openface
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✓ 模型创建成功")
    print(f"  ✓ 总参数: {total_params:,}")
    print(f"  ✓ 可训练参数: {trainable_params:,}")
    
    # 6. 创建训练器
    print("\n[6] 创建训练器...")
    
    trainer = MultiModalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir='checkpoints/test'
    )
    
    print("  ✓ 训练器创建成功")
    
    # 7. 测试训练（每个阶段只训练 2 个 epoch）
    print("\n[7] 开始测试训练...")
    print("\n注意: 这是小批量测试，只训练少量 epoch")
    print("      目的是验证训练流程是否正常工作")
    
    try:
        trainer.train_all_stages(
            stage1_epochs=2,  # 阶段1: 2 个 epoch
            stage2_epochs=2,  # 阶段2: 2 个 epoch
            stage3_epochs=2,  # 阶段3: 2 个 epoch
            stage1_lr=1e-3,
            stage2_lr=5e-4,
            stage3_lr=1e-4,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("✨ 测试训练完成!")
        print("="*60)
        print("\n训练流程验证成功，可以开始完整训练。")
        print("\n下一步:")
        print("  1. 预计算所有视频的特征")
        print("  2. 使用完整数据集训练模型")
        print("  3. 调整超参数以获得更好的性能")
        
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 8. 测试加载模型
    print("\n[8] 测试加载模型...")
    
    try:
        trainer.load_checkpoint('best_stage3.pth')
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ⚠️  模型加载失败: {e}")


def main():
    """主函数"""
    test_train()


if __name__ == '__main__':
    main()
