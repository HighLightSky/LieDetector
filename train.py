"""
训练脚本 - 多模态谎言检测模型

使用FusionModel和FusionModelTrainer进行训练
支持从预计算的特征文件加载数据
"""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import sys

from models.fusion import FusionModel
from trainer import FusionModelTrainer, LieDetectionDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练多模态谎言检测模型')
    
    # 数据参数
    parser.add_argument('--csv_path', type=str, default='src/dataset/video_labels.csv',
                        help='标签CSV文件路径')
    parser.add_argument('--features_dir', type=str, default='src/features',
                        help='预计算特征目录')
    parser.add_argument('--use_openface', action='store_true', default=True,
                        help='是否使用OpenFace特征')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='学习率（非常低以确保稳定性）')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    
    # 模型参数
    parser.add_argument('--visual_hidden', type=int, default=256,
                        help='视觉特征隐藏层维度')
    parser.add_argument('--fusion_hidden', type=int, default=128,
                        help='融合层隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比例')
    parser.add_argument('--uni_layers', type=int, default=2,
                        help='单模态Transformer层数')
    parser.add_argument('--uni_nhead', type=int, default=4,
                        help='单模态Transformer注意力头数')
    parser.add_argument('--com_heads', type=int, default=4,
                        help='跨模态注意力头数')
    parser.add_argument('--freeze_backbones', action='store_true', default=True,
                        help='是否冻结预训练backbone')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader工作进程数 (Windows建议设为0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子（更严格的控制）"""
    import os
    import numpy as np
    import random
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"\n随机种子: {args.seed}")
    
    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA不可用，使用CPU")
        args.device = 'cpu'
    print(f"使用设备: {args.device}")
    
    # 1. 检查数据
    print("\n[1] 检查数据...")
    csv_path = Path(args.csv_path)
    features_dir = Path(args.features_dir)
    
    if not csv_path.exists():
        print(f"标签文件不存在: {csv_path}")
        print("\n请先生成标签文件和提取特征:")
        print("  1. 下载和切片视频:")
        print("     python -c \"from dataloader.video_downloader import VideoDownloader; VideoDownloader().run()\"")
        print("  2. 提取特征:")
        print("     python dataloader/precompute_features.py")
        sys.exit(1)
    
    if not features_dir.exists():
        print(f"特征目录不存在: {features_dir}")
        print("\n请先提取特征:")
        print("  python dataloader/precompute_features.py")
        sys.exit(1)
    
    # 检查特征文件数量
    face_files = list(features_dir.glob('*_faces.pt'))
    audio_files = list(features_dir.glob('*_audios.pt'))
    openface_files = list(features_dir.glob('*_openfaces.pt'))
    
    print(f"  标签文件: {csv_path}")
    print(f"  特征目录: {features_dir}")
    print(f"  人脸特征: {len(face_files)} 个")
    print(f"  音频特征: {len(audio_files)} 个")
    print(f"  OpenFace特征: {len(openface_files)} 个")
    
    if len(face_files) == 0 or len(audio_files) == 0:
        print("\n没有足够的特征文件")
        print("请先提取特征: python dataloader/precompute_features.py")
        sys.exit(1)
    
    # 2. 创建数据集
    print("\n[2] 创建数据集...")
    
    try:
        dataset = LieDetectionDataset(
            csv_path=csv_path,
            features_dir=features_dir,
            use_openface=args.use_openface
        )
    except Exception as e:
        print(f"数据集创建失败: {e}")
        sys.exit(1)
    
    if len(dataset) == 0:
        print("数据集为空")
        sys.exit(1)
    
    # 3. 划分训练集和验证集
    print("\n[3] 划分训练集和验证集...")
    
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    if train_size == 0 or val_size == 0:
        print(f"数据集太小 (总共 {len(dataset)} 个样本)")
        print("  至少需要 5 个样本")
        sys.exit(1)
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  验证集: {len(val_dataset)} 个样本")
    
    # 4. 创建数据加载器
    print("\n[4] 创建数据加载器...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=LieDetectionDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=LieDetectionDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    
    # 5. 创建模型
    print("\n[5] 创建模型...")
    
    model = FusionModel(
        device=args.device,
        freeze_backbones=args.freeze_backbones,
        num_classes=2,
        visual_hidden=args.visual_hidden,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout,
        uni_layers=args.uni_layers,
        uni_nhead=args.uni_nhead,
        com_heads=args.com_heads
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  模型创建成功")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结backbone: {args.freeze_backbones}")
    
    # 6. 创建训练器
    print("\n[6] 创建训练器...")
    
    # 损失权重配置
    loss_weights = {
        'face': 0.2,        # 人脸分类损失
        'openface': 0.2,    # OpenFace分类损失
        'audio': 0.2,       # 音频分类损失
        'fa_au': 0.15,      # Face-Audio融合损失
        'fa_of': 0.15,      # Face-OpenFace融合损失
        'fused': 1.0        # 最终融合损失（权重最高）
    }
    
    trainer = FusionModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        save_dir=args.save_dir,
        loss_weights=loss_weights
    )

    # 启用正则化
    trainer.loss_computer.ent_reg_weight = 0.2  # 熵正则化
    trainer.loss_computer.w_mse_weight = 0.05   # 权重一致性
    
    print("  训练器创建成功")
    print("  损失权重配置:")
    for key, weight in loss_weights.items():
        print(f"      {key}: {weight}")
    
    # 7. 从检查点恢复（如果指定）
    if args.resume:
        print(f"\n[7] 从检查点恢复训练...")
        try:
            trainer.load_checkpoint(args.resume)
        except Exception as e:
            print(f"加载检查点失败: {e}")
            sys.exit(1)
    
    # 8. 开始训练
    print("\n[8] 开始训练...")
    print(f"\n训练配置:")
    print(f"  - Epochs: {args.num_epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Weight Decay: {args.weight_decay}")
    print(f"  - Patience: {args.patience}")
    print(f"  - Device: {args.device}")
    
    try:
        trainer.train(
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("[SUCCESS] 训练完成!")
        print("="*60)
        print(f"\n最佳模型保存在: {Path(args.save_dir) / 'best_model.pth'}")
        print(f"训练历史保存在: {Path(args.save_dir) / 'training_history.json'}")
        
        print("\n下一步:")
        print("  1. 使用最佳模型进行推理:")
        print(f"     python detect.py --checkpoint {args.save_dir}/best_model.pth --video path/to/video.mp4")
        print("  2. 或使用LieDetector API:")
        print("     from detector import LieDetector")
        print(f"     detector = LieDetector.from_checkpoint('{args.save_dir}/best_model.pth')")
        print("     result = detector.predict('video.mp4')")
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] 训练被用户中断")
        print(f"\n最新模型保存在: {Path(args.save_dir) / 'latest_model.pth'}")
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
