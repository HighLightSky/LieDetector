"""
验证循环模块
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from .loss_computer import LossComputer
from .metrics_evaluator import MetricsEvaluator


class ValidationLoop:
    """验证循环执行器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_computer: LossComputer,
        device: str = 'cuda'
    ):
        self.model = model
        self.loss_computer = loss_computer
        self.device = device
    
    def run_validation(
        self,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """执行验证"""
        self.model.eval()
        metrics = MetricsEvaluator()
        metrics.loss_details = {key: 0.0 for key in self.loss_computer.get_weights().keys()}
        
        pbar = tqdm(val_loader, desc='验证', disable=not verbose)
        
        with torch.no_grad():
            for batch in pbar:
                faces = batch['faces'].to(self.device)
                audios = batch['audios'].to(self.device)
                openfaces = batch['openfaces'].to(self.device) if batch['openfaces'] is not None else None
                labels = batch['labels'].to(self.device)
                
                output = self.model(faces, openfaces, audios)
                loss, losses = self.loss_computer.compute(output, labels, return_details=True)
                
                # 验证时跳过无效损失
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
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
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{metrics.get_accuracy():.2f}%'
                })
        
        return metrics.compute()
