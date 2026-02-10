"""
训练循环模块
负责单个epoch的训练逻辑
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from .loss_computer import LossComputer
from .metrics_evaluator import MetricsEvaluator


class TrainingLoop:
    """训练循环执行器
    
    负责执行单个epoch的训练过程
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_computer: LossComputer,
        device: str = 'cuda'
    ):
        """初始化训练循环
        
        Args:
            model: 训练模型
            loss_computer: 损失计算器
            device: 设备
        """
        self.model = model
        self.loss_computer = loss_computer
        self.device = device
    
    def run_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        verbose: bool = True
    ) -> Dict[str, float]:
        """执行一个训练epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            verbose: 是否显示进度条
        
        Returns:
            训练指标字典
        """
        self.model.train()
        metrics = MetricsEvaluator()
        
        # 初始化详细损失统计
        loss_weights = self.loss_computer.get_weights()
        metrics.loss_details = {key: 0.0 for key in loss_weights.keys()}
        
        pbar = tqdm(train_loader, desc='训练', disable=not verbose)
        
        for batch in pbar:
            # 数据移到设备
            faces = batch['faces'].to(self.device)
            audios = batch['audios'].to(self.device)
            openfaces = batch['openfaces'].to(self.device) if batch['openfaces'] is not None else None
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            output = self.model(faces, openfaces, audios)
            
            # 检查输出是否有NaN/Inf
            if self._has_nan_output(output):
                print(f"\n[WARNING] 前向传播产生NaN/Inf，跳过此批次")
                continue
            
            # 计算损失
            loss, losses = self.loss_computer.compute(output, labels, return_details=True)
            
            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] NaN/Inf loss detected, skipping batch")
                continue
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            if self._has_nan_gradient():
                print(f"\n[WARNING] 梯度包含NaN/Inf，跳过此批次的参数更新")
                optimizer.zero_grad()
                continue
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            if grad_norm > 10.0:
                print(f"\n[WARNING] 梯度范数过大 ({grad_norm:.2f})，跳过此批次")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            # 统计指标
            batch_size = faces.size(0)
            fused_probs = output['probs']['fused']
            _, predicted = torch.max(fused_probs, 1)
            
            metrics.update(
                loss=loss.item(),
                predictions=predicted,
                labels=labels,
                batch_size=batch_size,
                loss_details=losses
            )
            
            # 更新进度条
            if metrics.total > 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{metrics.get_accuracy():.2f}%'
                })
        
        return metrics.compute()
    
    def _has_nan_output(self, output: Dict) -> bool:
        """检查输出是否包含NaN/Inf
        
        Args:
            output: 模型输出字典
        
        Returns:
            是否包含NaN/Inf
        """
        for key, logits in output['logits'].items():
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return True
        return False
    
    def _has_nan_gradient(self) -> bool:
        """检查梯度是否包含NaN/Inf
        
        Returns:
            是否包含NaN/Inf
        """
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False
