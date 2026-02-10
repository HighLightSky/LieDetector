"""
指标评估模块
负责计算和跟踪训练/验证指标
"""

import torch
from typing import Dict


class MetricsEvaluator:
    """指标评估器
    
    负责计算准确率等评估指标
    """
    
    def __init__(self):
        """初始化指标评估器"""
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.total_loss = 0.0
        self.correct = 0
        self.total = 0
        self.loss_details = {}
    
    def update(
        self,
        loss: float,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        loss_details: Dict[str, float] = None
    ):
        """更新指标
        
        Args:
            loss: 批次损失
            predictions: 预测结果 (B,)
            labels: 真实标签 (B,)
            batch_size: 批次大小
            loss_details: 详细损失字典
        """
        self.total_loss += loss * batch_size
        self.correct += (predictions == labels).sum().item()
        self.total += labels.size(0)
        
        # 累计详细损失
        if loss_details:
            for key, val in loss_details.items():
                if key not in self.loss_details:
                    self.loss_details[key] = 0.0
                self.loss_details[key] += val * batch_size
    
    def compute(self) -> Dict[str, float]:
        """计算最终指标
        
        Returns:
            包含loss、accuracy和loss_detail的字典
        """
        if self.total == 0:
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'loss_detail': {key: float('inf') for key in self.loss_details.keys()}
            }
        
        avg_loss = self.total_loss / self.total
        accuracy = 100.0 * self.correct / self.total
        
        # 平均详细损失
        avg_loss_details = {}
        for key, val in self.loss_details.items():
            avg_loss_details[key] = val / self.total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'loss_detail': avg_loss_details
        }
    
    def get_accuracy(self) -> float:
        """获取当前准确率"""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total
    
    def get_loss(self) -> float:
        """获取当前平均损失"""
        if self.total == 0:
            return float('inf')
        return self.total_loss / self.total
