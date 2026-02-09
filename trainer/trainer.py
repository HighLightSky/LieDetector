"""
多模态谎言检测训练器
采用分阶段训练策略
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from typing import Dict, Optional
import json
from tqdm import tqdm

from detect import MultiModalFusionModel


class MultiModalTrainer:
    """多模态融合模型训练器
    
    分阶段训练策略:
    1. 阶段1: 只训练融合层，冻结所有子模型
    2. 阶段2: 解冻并微调所有子模型的分类头
    3. 阶段3: 端到端微调整个模型
    """
    
    def __init__(
        self,
        model: MultiModalFusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'stage': []
        }
        
        # 最佳模型
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def _freeze_all_submodels(self):
        """冻结所有子模型"""
        print("  冻结所有子模型...")
        for param in self.model.face_model.parameters():
            param.requires_grad = False
        for param in self.model.audio_model.parameters():
            param.requires_grad = False
        if self.model.openface_model is not None:
            for param in self.model.openface_model.parameters():
                param.requires_grad = False
    
    def _unfreeze_classification_heads(self):
        """解冻子模型的分类头"""
        print("  解冻子模型分类头...")
        # FacesModel 的 fc_block
        for param in self.model.face_model.fc_block.parameters():
            param.requires_grad = True
        # AudioModel 的 fc
        for param in self.model.audio_model.fc.parameters():
            param.requires_grad = True
        # OpenfaceModel 的 fc
        if self.model.openface_model is not None:
            for param in self.model.openface_model.fc.parameters():
                param.requires_grad = True
    
    def _unfreeze_all(self):
        """解冻所有参数（除了预训练骨干）"""
        print("  解冻所有可训练参数...")
        for param in self.model.parameters():
            param.requires_grad = True
        # 保持 FacesModel 的骨干冻结
        for param in self.model.face_model.backbone.parameters():
            param.requires_grad = False
    
    def _get_trainable_params(self):
        """获取可训练参数"""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        verbose: bool = True
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='训练', disable=not verbose)
        
        for batch in pbar:
            # 数据移到设备
            faces = batch['faces'].to(self.device)  # (B, T, C, H, W)
            audios = batch['audios'].to(self.device)  # (B, 768)
            openfaces = batch['openfaces'].to(self.device) if batch['openfaces'] is not None else None
            labels = batch['labels'].to(self.device)  # (B,)
            
            # 前向传播
            optimizer.zero_grad()
            logits = self.model(faces, audios, openfaces, verbose=False)  # (B, 2)
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item() * faces.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(
        self,
        criterion: nn.Module,
        verbose: bool = True
    ) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc='验证', disable=not verbose)
        
        with torch.no_grad():
            for batch in pbar:
                # 数据移到设备
                faces = batch['faces'].to(self.device)
                audios = batch['audios'].to(self.device)
                openfaces = batch['openfaces'].to(self.device) if batch['openfaces'] is not None else None
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                logits = self.model(faces, audios, openfaces, verbose=False)
                loss = criterion(logits, labels)
                
                # 统计
                total_loss += loss.item() * faces.size(0)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train_stage(
        self,
        stage: int,
        num_epochs: int,
        lr: float,
        weight_decay: float = 1e-4,
        verbose: bool = True
    ):
        """训练一个阶段
        
        Args:
            stage: 阶段编号 (1, 2, 3)
            num_epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            verbose: 是否显示详细信息
        """
        print(f"\n{'='*60}")
        print(f"阶段 {stage} 训练")
        print(f"{'='*60}")
        
        # 设置训练策略
        if stage == 1:
            print("策略: 只训练融合层")
            self._freeze_all_submodels()
            # 融合层始终可训练
        elif stage == 2:
            print("策略: 微调子模型分类头 + 融合层")
            self._unfreeze_classification_heads()
        elif stage == 3:
            print("策略: 端到端微调")
            self._unfreeze_all()
        else:
            raise ValueError(f"不支持的阶段: {stage}")
        
        # 优化器和损失函数
        trainable_params = self._get_trainable_params()
        print(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        
        # 训练循环
        print(f"\n开始训练 {num_epochs} 个 epoch...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # 训练
            train_metrics = self.train_epoch(optimizer, criterion, verbose=verbose)
            print(f"训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            
            # 验证
            val_metrics = self.validate(criterion, verbose=verbose)
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['stage'].append(stage)
            
            # 学习率调度
            scheduler.step(val_metrics['accuracy'])
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch + 1
                self.save_checkpoint(f'best_stage{stage}.pth', stage, epoch + 1)
                print(f"✓ 保存最佳模型 (验证准确率: {self.best_val_acc:.2f}%)")
            
            # 保存最新模型
            self.save_checkpoint(f'latest_stage{stage}.pth', stage, epoch + 1)
        
        print(f"\n阶段 {stage} 训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
    
    def train_all_stages(
        self,
        stage1_epochs: int = 10,
        stage2_epochs: int = 10,
        stage3_epochs: int = 10,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 5e-4,
        stage3_lr: float = 1e-4,
        verbose: bool = True
    ):
        """训练所有阶段
        
        Args:
            stage1_epochs: 阶段1训练轮数
            stage2_epochs: 阶段2训练轮数
            stage3_epochs: 阶段3训练轮数
            stage1_lr: 阶段1学习率
            stage2_lr: 阶段2学习率
            stage3_lr: 阶段3学习率
            verbose: 是否显示详细信息
        """
        start_time = time.time()
        
        # 阶段1: 训练融合层
        self.train_stage(1, stage1_epochs, stage1_lr, verbose=verbose)
        
        # 阶段2: 微调分类头
        self.train_stage(2, stage2_epochs, stage2_lr, verbose=verbose)
        
        # 阶段3: 端到端微调
        self.train_stage(3, stage3_epochs, stage3_lr, verbose=verbose)
        
        # 总结
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"{'='*60}")
        print(f"总训练时间: {elapsed_time / 60:.2f} 分钟")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"模型保存在: {self.save_dir}")
        
        # 保存训练历史
        self.save_history()
    
    def save_checkpoint(self, filename: str, stage: int, epoch: int):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'stage': stage,
            'epoch': epoch,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        print(f"✓ 加载检查点: {filename}")
        print(f"  阶段: {checkpoint['stage']}, Epoch: {checkpoint['epoch']}")
        print(f"  最佳验证准确率: {self.best_val_acc:.2f}%")
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ 训练历史保存到: {history_path}")
