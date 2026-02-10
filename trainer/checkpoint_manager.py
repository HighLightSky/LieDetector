"""
检查点管理模块
负责模型检查点的保存和加载
"""

import torch
from pathlib import Path
from typing import Dict, Any


class CheckpointManager:
    """检查点管理器
    
    负责保存和加载模型检查点
    """
    
    def __init__(self, save_dir: str = 'checkpoints'):
        """初始化检查点管理器
        
        Args:
            save_dir: 检查点保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def save(
        self,
        filename: str,
        model_state: Dict,
        epoch: int,
        history: Dict = None,
        extra_info: Dict = None
    ):
        """保存检查点
        
        Args:
            filename: 文件名
            model_state: 模型状态字典
            epoch: 当前epoch
            history: 训练历史
            extra_info: 额外信息（如损失权重等）
        """
        checkpoint = {
            'model_state_dict': model_state,
            'epoch': epoch,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        
        if history is not None:
            checkpoint['history'] = history
        
        if extra_info is not None:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load(self, filename: str, device: str = 'cuda') -> Dict[str, Any]:
        """加载检查点
        
        Args:
            filename: 文件名
            device: 设备
        
        Returns:
            检查点字典
        """
        checkpoint_path = self.save_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 更新最佳记录
        if 'best_val_acc' in checkpoint:
            self.best_val_acc = checkpoint['best_val_acc']
        if 'best_epoch' in checkpoint:
            self.best_epoch = checkpoint['best_epoch']
        
        return checkpoint
    
    def update_best(self, val_acc: float, epoch: int) -> bool:
        """更新最佳模型记录
        
        Args:
            val_acc: 验证准确率
            epoch: 当前epoch
        
        Returns:
            是否更新了最佳记录
        """
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            return True
        return False
    
    def get_best_info(self) -> Dict[str, Any]:
        """获取最佳模型信息"""
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
    
    def get_save_dir(self) -> Path:
        """获取保存目录"""
        return self.save_dir
