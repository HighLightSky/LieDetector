"""
实验工具模块
提供实验数据可视化和分析工具
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


class ExperimentVisualizer:
    """实验数据可视化工具"""
    
    def __init__(self, history_path: str = "checkpoints/training_history.json"):
        """
        初始化可视化工具
        
        Args:
            history_path: 训练历史文件路径
        """
        self.history_path = history_path
        self.data = self._load_data()
        self.config = self.data.get('training_config', {})  # 提取训练配置
        
    def _load_data(self) -> Dict:
        """加载训练历史数据"""
        with open(self.history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果数据包含 training_config，说明是新格式
        # 确保向后兼容旧格式
        if 'training_config' in data:
            # 新格式：training_config + 历史数据
            return data
        else:
            # 旧格式：只有历史数据
            return data
    
    def _ensure_output_dir(self, output_dir: str = "experiment_history/svg"):
        """确保输出目录存在"""
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def print_config(self):
        """打印训练配置信息"""
        if not self.config:
            print("没有找到训练配置信息（可能是旧格式的历史文件）")
            return
        
        print("\n" + "="*60)
        print("训练配置信息")
        print("="*60)
        
        for key, value in self.config.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        print("="*60 + "\n")
    
    def _generate_filename(self, prefix: str, output_dir: str) -> str:
        """生成带时间戳的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(output_dir, f"{timestamp}_{prefix}.svg")
    
    def plot_loss_curves(self, 
                        show_train: bool = True, 
                        show_val: bool = True,
                        output_dir: str = "experiment_history/svg",
                        figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        绘制损失曲线
        
        Args:
            show_train: 是否显示训练损失
            show_val: 是否显示验证损失
            output_dir: 输出目录
            figsize: 图像大小
            
        Returns:
            保存的文件路径
        """
        output_dir = self._ensure_output_dir(output_dir)
        
        plt.figure(figsize=figsize)
        epochs = range(1, len(self.data['train_loss']) + 1)
        
        if show_train:
            plt.plot(epochs, self.data['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if show_val:
            plt.plot(epochs, self.data['val_loss'], 'r-', label='Val Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self._generate_filename("loss_curves", output_dir)
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_accuracy_curves(self,
                            show_train: bool = True,
                            show_val: bool = True,
                            output_dir: str = "experiment_history/svg",
                            figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        绘制准确率曲线
        
        Args:
            show_train: 是否显示训练准确率
            show_val: 是否显示验证准确率
            output_dir: 输出目录
            figsize: 图像大小
            
        Returns:
            保存的文件路径
        """
        output_dir = self._ensure_output_dir(output_dir)
        
        plt.figure(figsize=figsize)
        epochs = range(1, len(self.data['train_acc']) + 1)
        
        if show_train:
            plt.plot(epochs, self.data['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        if show_val:
            plt.plot(epochs, self.data['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self._generate_filename("accuracy_curves", output_dir)
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

    def plot_modality_losses(self,
                            modalities: Optional[List[str]] = None,
                            split: str = 'train',
                            output_dir: str = "experiment_history/svg",
                            figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        绘制各模态损失曲线
        
        Args:
            modalities: 要绘制的模态列表，None表示全部
            split: 'train' 或 'val'
            output_dir: 输出目录
            figsize: 图像大小
            
        Returns:
            保存的文件路径
        """
        output_dir = self._ensure_output_dir(output_dir)
        
        loss_key = f'{split}_loss_detail'
        if loss_key not in self.data:
            raise ValueError(f"数据中不存在 {loss_key}")
        
        # 获取所有可用的模态
        available_modalities = list(self.data[loss_key][0].keys())
        if modalities is None:
            modalities = [m for m in available_modalities if m not in ['ent_reg_weight', 'w_mse_weight']]
        
        plt.figure(figsize=figsize)
        epochs = range(1, len(self.data[loss_key]) + 1)
        
        # 为每个模态绘制曲线
        for modality in modalities:
            values = [epoch_data[modality] for epoch_data in self.data[loss_key]]
            plt.plot(epochs, values, label=modality, linewidth=2, marker='o', markersize=3)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{split.capitalize()} Modality Losses', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self._generate_filename(f"{split}_modality_losses", output_dir)
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_combined_overview(self,
                              output_dir: str = "experiment_history/svg",
                              figsize: Tuple[int, int] = (16, 10)) -> str:
        """
        绘制综合概览图（2x2子图）
        
        Args:
            output_dir: 输出目录
            figsize: 图像大小
            
        Returns:
            保存的文件路径
        """
        output_dir = self._ensure_output_dir(output_dir)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        epochs = range(1, len(self.data['train_loss']) + 1)
        
        # 子图1: 损失曲线
        axes[0, 0].plot(epochs, self.data['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.data['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: 准确率曲线
        axes[0, 1].plot(epochs, self.data['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.data['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: 训练模态损失
        # 动态获取可用的模态
        if 'train_loss_detail' in self.data and len(self.data['train_loss_detail']) > 0:
            available_modalities = [k for k in self.data['train_loss_detail'][0].keys() 
                                   if k not in ['ent_reg_weight', 'w_mse_weight', 'entropy_reg', 'weight_mse']]
            for modality in available_modalities:
                try:
                    values = [epoch_data[modality] for epoch_data in self.data['train_loss_detail']]
                    axes[1, 0].plot(epochs, values, label=modality, linewidth=2)
                except KeyError:
                    continue
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Train Modality Losses')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 验证模态损失
        if 'val_loss_detail' in self.data and len(self.data['val_loss_detail']) > 0:
            available_modalities = [k for k in self.data['val_loss_detail'][0].keys() 
                                   if k not in ['ent_reg_weight', 'w_mse_weight', 'entropy_reg', 'weight_mse']]
            for modality in available_modalities:
                try:
                    values = [epoch_data[modality] for epoch_data in self.data['val_loss_detail']]
                    axes[1, 1].plot(epochs, values, label=modality, linewidth=2)
                except KeyError:
                    continue
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Val Modality Losses')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self._generate_filename("combined_overview", output_dir)
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_custom(self,
                   metrics: List[str],
                   labels: Optional[List[str]] = None,
                   title: str = "Custom Metrics",
                   output_dir: str = "experiment_history/svg",
                   figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        自定义绘图
        
        Args:
            metrics: 要绘制的指标列表（如 ['train_loss', 'val_loss']）
            labels: 图例标签，None则使用metrics作为标签
            title: 图表标题
            output_dir: 输出目录
            figsize: 图像大小
            
        Returns:
            保存的文件路径
        """
        output_dir = self._ensure_output_dir(output_dir)
        
        if labels is None:
            labels = metrics
        
        plt.figure(figsize=figsize)
        
        for metric, label in zip(metrics, labels):
            if metric in self.data:
                epochs = range(1, len(self.data[metric]) + 1)
                plt.plot(epochs, self.data[metric], label=label, linewidth=2, marker='o', markersize=3)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self._generate_filename("custom_plot", output_dir)
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
