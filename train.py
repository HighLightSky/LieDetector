"""
完整训练脚本
使用完整数据集训练谎言检测模型
"""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse

from detect import MultiModalFusionModel
from trainer import MultiModalTrainer, LieDetectionDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练谎言检测模型')
    
    # 数据参数
    parser.add_argument('--csv_path', type=str, default='src/dataset/video_labels.csv',
                        help='标签 CSV 文件路径')
    parser.add_argument('--features_dir', type=str, default='src/features',
                        help='预计算特征目录')
    parser.add_argument('--use_openface', action='store_true', default=True,
                        help='是否使用 OpenFace 特征')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    
    # 阶段训练参数
    parser.add_argument('--stage1_epochs', type=int, default=10,
                        help='阶段1训练轮数')
    parser.add_argument('--stage2_epochs', type=int, default=10,
                        help='阶段2训练轮数')
    parser.add_argument('--stage3_epochs', type=int, default=10,
                        help='阶段3训练轮数')
    
    parser.add_argument('--stage1_lr', type=float, default=1e-3,
                        help='阶段1学习率')
    parser.add_argument('--stage2_lr', type=float, default=5e-4,
                        help='阶段2学习率')
    parser.add_argument('--stage3_lr', type=float, default=1e-4,
                        help='阶段3学习率')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "="*60)
    print("谎言检测模型训练")
    print("="*60)
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"\n随机种子: {args.seed}")
    
    # 设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 加载数据集
    print("\n[1] 加载数据集...")
    print(f"  CSV 文件: {args.csv_path}")
    print(f"  特征目录: {args.features_dir}")
    
    dataset = LieDetectionDataset(
        csv_path=args.csv_path,
        features_dir=args.features_dir,
        use_openface=args.use_openface
    )
    
    if len(dataset) == 0:
        print("✗ 数据集为空，请先提取特征")
        return
    
    # 2. 划分训练集和验证集
    print(f"\n[2] 划分数据集 (验证集比例: {args.val_split})...")
    
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  验证集: {len(val_dataset)} 个样本")
    
    # 3. 创建数据加载器
    print(f"\n[3] 创建数据加载器 (batch_size={args.batch_size})...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=LieDetectionDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=LieDetectionDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    
    # 4. 创建模型
    print("\n[4] 创建模型...")
    
    model = MultiModalFusionModel(
        device=device,
        use_openface=args.use_openface
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 5. 创建训练器
    print(f"\n[5] 创建训练器 (保存目录: {args.save_dir})...")
    
    trainer = MultiModalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir
    )
    
    # 6. 开始训练
    print("\n[6] 开始分阶段训练...")
    print(f"  阶段1: {args.stage1_epochs} epochs, lr={args.stage1_lr}")
    print(f"  阶段2: {args.stage2_epochs} epochs, lr={args.stage2_lr}")
    print(f"  阶段3: {args.stage3_epochs} epochs, lr={args.stage3_lr}")
    
    trainer.train_all_stages(
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        stage3_lr=args.stage3_lr,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("✨ 训练完成!")
    print("="*60)


if __name__ == '__main__':
    main()
