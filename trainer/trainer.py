"""
多模态谎言检测训练器
基于FusionModel的训练策略
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from typing import Dict, Optional
import json
from tqdm import tqdm

from models.fusion import FusionModel


class FusionModelTrainer:
    """融合模型训练器
    
    训练策略：
    1. 冻结预训练backbone（MobileNetV3）
    2. 训练融合层、注意力模块、分类头
    3. 支持多任务学习（各模态+融合）
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
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 损失权重（多任务学习）
        if loss_weights is None:
            loss_weights = {
                'face': 0.2,
                'openface': 0.2,
                'audio': 0.2,
                'fa_au': 0.15,
                'fa_of': 0.15,
                'fused': 1.0  # 融合损失权重最高
            }
        self.loss_weights = loss_weights
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_loss_detail': [],  # 详细损失
            'val_loss_detail': []
        }
        
        # 最佳模型
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def compute_loss(
        self,
        output: Dict,
        labels: torch.Tensor,
        return_details: bool = False
    ) -> torch.Tensor:
        """计算多任务损失
        
        Args:
            output: 模型输出字典
            labels: 真实标签 (B,)
            return_details: 是否返回详细损失
        
        Returns:
            总损失（如果return_details=True，返回(总损失, 详细损失字典)）
        """
        criterion = nn.CrossEntropyLoss()
        
        losses = {}
        total_loss = 0.0
        
        # 各模态损失
        for key in ['face', 'openface', 'audio', 'fa_au', 'fa_of', 'fused']:
            if key in output['logits']:
                logits = output['logits'][key]
                loss = criterion(logits, labels)
                losses[key] = loss.item()
                total_loss += self.loss_weights[key] * loss
        
        if return_details:
            return total_loss, losses
        return total_loss
    
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        verbose: bool = True
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 详细损失统计
        loss_details = {key: 0.0 for key in self.loss_weights.keys()}
        
        pbar = tqdm(self.train_loader, desc='训练', disable=not verbose)
        
        for batch in pbar:
            # 数据移到设备
            faces = batch['faces'].to(self.device)  # (B, T, C, H, W)
            audios = batch['audios'].to(self.device)  # (B, 768)
            openfaces = batch['openfaces'].to(self.device) if batch['openfaces'] is not None else None
            labels = batch['labels'].to(self.device)  # (B,)
            
            # 前向传播
            optimizer.zero_grad()
            output = self.model(faces, openfaces, audios)
            
            # 检查输出是否有NaN/Inf
            has_nan_output = False
            for key, logits in output['logits'].items():
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    has_nan_output = True
                    break
            
            if has_nan_output:
                print(f"\n[WARNING] 前向传播产生NaN/Inf，跳过此批次")
                continue
            
            # 计算损失
            loss, losses = self.compute_loss(output, labels, return_details=True)
            
            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] NaN/Inf loss detected, skipping batch")
                continue
            
            # 反向传播
            loss.backward()
            
            # 检查梯度是否有NaN/Inf
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"\n[WARNING] 梯度包含NaN/Inf，跳过此批次的参数更新")
                optimizer.zero_grad()
                continue
            
            # 梯度裁剪（防止梯度爆炸）
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 如果梯度范数过大，也跳过
            if grad_norm > 10.0:
                print(f"\n[WARNING] 梯度范数过大 ({grad_norm:.2f})，跳过此批次")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            # 统计
            batch_size = faces.size(0)
            total_loss += loss.item() * batch_size
            
            # 使用融合概率计算准确率
            fused_probs = output['probs']['fused']
            _, predicted = torch.max(fused_probs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # 累计详细损失
            for key, val in losses.items():
                loss_details[key] += val * batch_size
            
            # 更新进度条
            if total > 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        # 防止除零
        if total == 0:
            print("\n[ERROR] 所有批次都被跳过（NaN/Inf），无法计算损失")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'loss_detail': {key: float('inf') for key in loss_details.keys()}
            }
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        # 平均详细损失
        for key in loss_details:
            loss_details[key] /= total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'loss_detail': loss_details
        }
    
    def validate(
        self,
        verbose: bool = True
    ) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 详细损失统计
        loss_details = {key: 0.0 for key in self.loss_weights.keys()}
        
        pbar = tqdm(self.val_loader, desc='验证', disable=not verbose)
        
        with torch.no_grad():
            for batch in pbar:
                # 数据移到设备
                faces = batch['faces'].to(self.device)
                audios = batch['audios'].to(self.device)
                openfaces = batch['openfaces'].to(self.device) if batch['openfaces'] is not None else None
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                output = self.model(faces, openfaces, audios)
                
                # 计算损失
                loss, losses = self.compute_loss(output, labels, return_details=True)
                
                # 统计
                batch_size = faces.size(0)
                total_loss += loss.item() * batch_size
                
                # 使用融合概率计算准确率
                fused_probs = output['probs']['fused']
                _, predicted = torch.max(fused_probs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # 累计详细损失
                for key, val in losses.items():
                    loss_details[key] += val * batch_size
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        # 平均详细损失
        for key in loss_details:
            loss_details[key] /= total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'loss_detail': loss_details
        }
    
    def train(
        self,
        num_epochs: int = 30,
        lr: float = 1e-5,  # 降低到非常低的学习率
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
        print(f"开始训练融合模型")
        print(f"{'='*60}")
        
        # 统计可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")
        
        # 优化器
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8,  # 增加数值稳定性
            betas=(0.9, 0.999)  # 默认值
        )
        
        # 学习率调度器 - 使用余弦退火而不是ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=lr * 0.01  # 最小学习率为初始的1%
        )
        
        # 训练循环
        print(f"\n开始训练 {num_epochs} 个 epoch...")
        start_time = time.time()
        
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # 训练
            train_metrics = self.train_epoch(optimizer, verbose=verbose)
            print(f"训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            
            # 打印详细损失
            if verbose:
                print("  详细损失:")
                for key, val in train_metrics['loss_detail'].items():
                    print(f"    {key}: {val:.4f}")
            
            # 验证
            val_metrics = self.validate(verbose=verbose)
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # 打印详细损失
            if verbose:
                print("  详细损失:")
                for key, val in val_metrics['loss_detail'].items():
                    print(f"    {key}: {val:.4f}")
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_loss_detail'].append(train_metrics['loss_detail'])
            self.history['val_loss_detail'].append(val_metrics['loss_detail'])
            
            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch + 1
                self.save_checkpoint('best_model.pth', epoch + 1)
                print(f"[OK] 保存最佳模型 (验证准确率: {self.best_val_acc:.2f}%)")
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 保存最新模型
            self.save_checkpoint('latest_model.pth', epoch + 1)
            
            # 早停
            if no_improve_count >= patience * 2:
                print(f"\n早停: {patience * 2} 个epoch没有改进")
                break
        
        # 总结
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"{'='*60}")
        print(f"总训练时间: {elapsed_time / 60:.2f} 分钟")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"模型保存在: {self.save_dir}")
        
        # 保存训练历史
        self.save_history()
    
    def save_checkpoint(self, filename: str, epoch: int):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'loss_weights': self.loss_weights
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        if 'loss_weights' in checkpoint:
            self.loss_weights = checkpoint['loss_weights']
        print(f"[OK] 加载检查点: {filename}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  最佳验证准确率: {self.best_val_acc:.2f}%")
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"[OK] 训练历史保存到: {history_path}")
