"""
训练循环模块
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from .loss_computer import LossComputer
from .metrics_evaluator import MetricsEvaluator


class TrainingLoop:
    """训练循环执行器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_computer: LossComputer,
        device: str = 'cuda'
    ):
        self.model = model
        self.loss_computer = loss_computer
        self.device = device
    
    def run_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        verbose: bool = True
    ) -> Dict[str, float]:
        """执行一个训练epoch"""
        self.model.train()
        metrics = MetricsEvaluator()
        metrics.loss_details = {key: 0.0 for key in self.loss_computer.get_weights().keys()}
        
        pbar = tqdm(train_loader, desc='训练', disable=not verbose)
        
        for batch in pbar:
            faces = batch['faces'].to(self.device)
            audios = batch['audios'].to(self.device)
            openfaces = batch['openfaces'].to(self.device) if batch['openfaces'] is not None else None
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            output = self.model(faces, openfaces, audios)
            
            loss, losses = self.loss_computer.compute(output, labels, return_details=True)
            
            # 检查损失有效性
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] 跳过批次：损失为 {loss.item()}")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
            
            if metrics.total > 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{metrics.get_accuracy():.2f}%'
                })
        
        return metrics.compute()
