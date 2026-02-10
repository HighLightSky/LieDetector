"""
损失计算模块
负责多任务损失的计算和权重管理
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class LossComputer:
    """多任务损失计算器
    
    负责计算各模态和融合层的加权损失
    """
    
    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        """初始化损失计算器
        
        Args:
            loss_weights: 各任务的损失权重字典
        """
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
        self.criterion = nn.CrossEntropyLoss()
    
    def compute(
        self,
        output: Dict,
        labels: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """计算多任务损失
        
        Args:
            output: 模型输出字典，包含各模态的logits
            labels: 真实标签 (B,)
            return_details: 是否返回详细损失
        
        Returns:
            总损失，如果return_details=True则返回(总损失, 详细损失字典)
        """
        losses = {}
        total_loss = 0.0
        
        # 计算各模态损失
        for key in ['face', 'openface', 'audio', 'fa_au', 'fa_of', 'fused']:
            if key in output['logits']:
                logits = output['logits'][key]
                loss = self.criterion(logits, labels)
                losses[key] = loss.item()
                total_loss += self.loss_weights[key] * loss
        
        if return_details:
            return total_loss, losses
        return total_loss, None
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前损失权重"""
        return self.loss_weights.copy()
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新损失权重
        
        Args:
            new_weights: 新的权重字典
        """
        self.loss_weights.update(new_weights)
