"""
训练器模块（模块化版本）

主要组件：
- FusionModelTrainer: 主训练器（模块化）
- LieDetectionDataset: 数据集
- LossComputer: 损失计算模块
- MetricsEvaluator: 指标评估模块
- TrainingLoop: 训练循环模块
- ValidationLoop: 验证循环模块
- CheckpointManager: 检查点管理模块
- HistoryTracker: 历史记录模块
"""

from trainer.trainer import FusionModelTrainer
from trainer.dataset import LieDetectionDataset
from trainer.loss_computer import LossComputer
from trainer.metrics_evaluator import MetricsEvaluator
from trainer.training_loop import TrainingLoop
from trainer.validation_loop import ValidationLoop
from trainer.checkpoint_manager import CheckpointManager
from trainer.history_tracker import HistoryTracker

__all__ = [
    'FusionModelTrainer',
    'LieDetectionDataset',
    'LossComputer',
    'MetricsEvaluator',
    'TrainingLoop',
    'ValidationLoop',
    'CheckpointManager',
    'HistoryTracker'
]
