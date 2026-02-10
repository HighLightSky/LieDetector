"""
训练历史记录模块
负责记录和保存训练历史
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class HistoryTracker:
    """训练历史跟踪器
    
    负责记录训练过程中的各项指标
    """
    
    def __init__(self):
        """初始化历史跟踪器"""
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_loss_detail': [],
            'val_loss_detail': []
        }
        self.config = {}  # 存储训练配置
    
    def record_epoch(
        self,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any]
    ):
        """记录一个epoch的指标
        
        Args:
            train_metrics: 训练指标字典
            val_metrics: 验证指标字典
        """
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        
        if 'loss_detail' in train_metrics:
            self.history['train_loss_detail'].append(train_metrics['loss_detail'])
        if 'loss_detail' in val_metrics:
            self.history['val_loss_detail'].append(val_metrics['loss_detail'])
    
    def set_config(self, config: Dict[str, Any]):
        """设置训练配置
        
        Args:
            config: 训练配置字典
        """
        self.config = config.copy()
    
    def get_history(self) -> Dict[str, List]:
        """获取完整历史"""
        return self.history.copy()
    
    def save(self, save_path: Path, use_timestamp: bool = True):
        """保存历史到JSON文件
        
        Args:
            save_path: 保存路径（如果use_timestamp=True，会被修改为带时间戳的路径）
            use_timestamp: 是否使用时间戳文件名
        """
        # 如果使用时间戳，修改保存路径
        if use_timestamp:
            # 创建 history 子目录
            history_dir = save_path.parent / 'history'
            history_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'train_history_{timestamp}.json'
            save_path = history_dir / filename
        
        # 构建完整的保存数据（配置在最前面）
        save_data = {}
        
        # 1. 训练配置（放在最前面）
        if self.config:
            save_data['training_config'] = self.config
        
        # 2. 训练历史数据
        save_data.update(self.history)
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return save_path
    
    def load(self, load_path: Path):
        """从JSON文件加载历史
        
        Args:
            load_path: 加载路径
        """
        if not load_path.exists():
            raise FileNotFoundError(f"历史文件不存在: {load_path}")
        
        with open(load_path, 'r', encoding='utf-8') as f:
            self.history = json.load(f)
    
    def get_best_epoch(self) -> int:
        """获取验证准确率最高的epoch
        
        Returns:
            最佳epoch索引（从1开始）
        """
        if not self.history['val_acc']:
            return 0
        best_idx = max(range(len(self.history['val_acc'])), 
                      key=lambda i: self.history['val_acc'][i])
        return best_idx + 1
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """获取最新的指标"""
        if not self.history['train_loss']:
            return {}
        
        return {
            'train_loss': self.history['train_loss'][-1],
            'train_acc': self.history['train_acc'][-1],
            'val_loss': self.history['val_loss'][-1],
            'val_acc': self.history['val_acc'][-1]
        }
