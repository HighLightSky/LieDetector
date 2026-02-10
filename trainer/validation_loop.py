"""
验证循环模块
负责模型验证逻辑
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from .loss_computer import LossComputer
from .metrics_evaluator import MetricsEvaluator


class ValidationLoop:
    """验证循环执行器
    
    负责执行模型验证过程
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_computer: LossComputer,
        device: str = 'cuda'
    ):
        """初始化验证循环
        
        Args:
            model: 验证模型
            loss_computer: 损失计算器
            device: 设备
        """
        self.model = model
        self.loss_computer = loss_computer
        self.device = device
    
    def run_validation(
        self,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """执行验证
        
        Args:
            val_loader: 验证数据加载器
            verbose: 是否显示进度条
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        metrics = MetricsEvaluator()
        
        # 初始化详细损失统计
        loss_weights = self.loss_computer.get_weights()
        metrics.loss_details = {key: 0.0 for key in loss_weights.keys()}
        
        pbar = tqdm(val_loader, desc='验证', disable=not verbose)
        
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
                loss, losses = self.loss_computer.compute(output, labels, return_details=True)
                
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
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{metrics.get_accuracy():.2f}%'
                })
        
        return metrics.compute()
