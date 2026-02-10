"""
多模态谎言检测训练器（模块化版本）
基于FusionModel的训练策略

模块化设计：
- LossComputer: 损失计算
- MetricsEvaluator: 指标评估
- TrainingLoop: 训练循环
- ValidationLoop: 验证循环
- CheckpointManager: 检查点管理
- HistoryTracker: 历史记录
"""

import torch
from torch.utils.data import DataLoader
import time
from typing import Dict, Optional

from models.fusion import FusionModel
from .loss_computer import LossComputer
from .training_loop import TrainingLoop
from .validation_loop import ValidationLoop
from .checkpoint_manager import CheckpointManager
from .history_tracker import HistoryTracker


class FusionModelTrainer:
    """融合模型训练器（模块化版本）
    
    训练策略：
    1. 冻结预训练backbone（MobileNetV3）
    2. 训练融合层、注意力模块、分类头
    3. 支持多任务学习（各模态+融合）
    
    模块化组件：
    - loss_computer: 损失计算
    - training_loop: 训练循环
    - validation_loop: 验证循环
    - checkpoint_manager: 检查点管理
    - history_tracker: 历史记录
    """
    
    def __init__(
        self,
        model: FusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        save_dir: str = 'checkpoints',
        loss_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 初始化各个模块
        self.loss_computer = LossComputer(loss_weights)
        self.training_loop = TrainingLoop(self.model, self.loss_computer, device)
        self.validation_loop = ValidationLoop(self.model, self.loss_computer, device)
        self.checkpoint_manager = CheckpointManager(save_dir)
        self.history_tracker = HistoryTracker()
    
    
    
    
    def train(
        self,
        num_epochs: int = 30,
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        patience: int = 5,
        verbose: bool = True
    ):
        """训练模型
        
        Args:
            num_epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            verbose: 是否显示详细信息
        """
        print(f"\n{'='*60}")
        print(f"开始训练融合模型（模块化版本）")
        print(f"{'='*60}")
        
        # 统计可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")
        
        # 优化器
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=lr * 0.01
        )
        
        # 训练循环
        print(f"\n开始训练 {num_epochs} 个 epoch...")
        start_time = time.time()
        
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # 训练阶段（使用TrainingLoop模块）
            train_metrics = self.training_loop.run_epoch(
                self.train_loader,
                optimizer,
                verbose=verbose
            )
            print(f"训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            
            if verbose and train_metrics['loss_detail']:
                print("  详细损失:")
                for key, val in train_metrics['loss_detail'].items():
                    print(f"    {key}: {val:.4f}")
            
            # 验证阶段（使用ValidationLoop模块）
            val_metrics = self.validation_loop.run_validation(
                self.val_loader,
                verbose=verbose
            )
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            if verbose and val_metrics['loss_detail']:
                print("  详细损失:")
                for key, val in val_metrics['loss_detail'].items():
                    print(f"    {key}: {val:.4f}")
            
            # 记录历史（使用HistoryTracker模块）
            self.history_tracker.record_epoch(train_metrics, val_metrics)
            
            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")
            
            # 保存最佳模型（使用CheckpointManager模块）
            if self.checkpoint_manager.update_best(val_metrics['accuracy'], epoch + 1):
                self.checkpoint_manager.save(
                    'best_model.pth',
                    self.model.state_dict(),
                    epoch + 1,
                    history=self.history_tracker.get_history(),
                    extra_info={'loss_weights': self.loss_computer.get_weights()}
                )
                best_info = self.checkpoint_manager.get_best_info()
                print(f"[OK] 保存最佳模型 (验证准确率: {best_info['best_val_acc']:.2f}%)")
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 保存最新模型
            self.checkpoint_manager.save(
                'latest_model.pth',
                self.model.state_dict(),
                epoch + 1,
                history=self.history_tracker.get_history(),
                extra_info={'loss_weights': self.loss_computer.get_weights()}
            )
            
            # 早停
            if no_improve_count >= patience * 2:
                print(f"\n早停: {patience * 2} 个epoch没有改进")
                break
        
        # 总结
        elapsed_time = time.time() - start_time
        best_info = self.checkpoint_manager.get_best_info()
        
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"{'='*60}")
        print(f"总训练时间: {elapsed_time / 60:.2f} 分钟")
        print(f"最佳验证准确率: {best_info['best_val_acc']:.2f}% (Epoch {best_info['best_epoch']})")
        print(f"模型保存在: {self.checkpoint_manager.get_save_dir()}")
        
        # 保存训练历史
        self.save_history()
    
    
    def save_checkpoint(self, filename: str, epoch: int):
        """保存检查点（兼容旧接口）
        
        Args:
            filename: 文件名
            epoch: 当前epoch
        """
        self.checkpoint_manager.save(
            filename,
            self.model.state_dict(),
            epoch,
            history=self.history_tracker.get_history(),
            extra_info={'loss_weights': self.loss_computer.get_weights()}
        )
    
    def load_checkpoint(self, filename: str):
        """加载检查点（兼容旧接口）
        
        Args:
            filename: 文件名
        """
        checkpoint = self.checkpoint_manager.load(filename, self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'history' in checkpoint:
            self.history_tracker.history = checkpoint['history']
        
        if 'loss_weights' in checkpoint:
            self.loss_computer.update_weights(checkpoint['loss_weights'])
        
        best_info = self.checkpoint_manager.get_best_info()
        print(f"[OK] 加载检查点: {filename}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  最佳验证准确率: {best_info['best_val_acc']:.2f}%")
    
    def save_history(self):
        """保存训练历史（兼容旧接口）"""
        history_path = self.checkpoint_manager.get_save_dir() / 'training_history.json'
        self.history_tracker.save(history_path)
        print(f"[OK] 训练历史保存到: {history_path}")

