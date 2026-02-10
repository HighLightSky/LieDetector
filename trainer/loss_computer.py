"""
损失计算模块
负责多任务损失的计算和权重管理

支持三种损失：
1. 任务损失（NLLLoss）
2. 熵正则化（鼓励权重明确性）
3. 权重一致性损失（权重与性能对齐）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class LossComputer:
    """多任务损失计算器
    
    负责计算各模态和融合层的加权损失
    支持熵正则化和权重一致性损失
    """
    
    def __init__(
        self, 
        loss_weights: Optional[Dict[str, float]] = None,
        ent_reg_weight: float = 0.2,
        w_mse_weight: float = 0.05
    ):
        """初始化损失计算器
        
        Args:
            loss_weights: 各任务的损失权重字典
            ent_reg_weight: 熵正则化权重
            w_mse_weight: 权重一致性损失权重
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
        self.ent_reg_weight = ent_reg_weight
        self.w_mse_weight = w_mse_weight
        self.criterion = nn.CrossEntropyLoss()
    
    def compute(
        self,
        output: Dict,
        labels: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """计算多任务损失（包含三种损失）
        
        Args:
            output: 模型输出字典，包含各模态的logits、probs和weights
            labels: 真实标签 (B,)
            return_details: 是否返回详细损失
        
        Returns:
            总损失，如果return_details=True则返回(总损失, 详细损失字典)
        """
        losses = {}
        total_loss = 0.0
        
        # 1. 任务损失（使用NLLLoss）
        fused_probs = output['probs']['fused']
        # 处理NaN/Inf
        fused_probs = torch.nan_to_num(fused_probs, nan=1e-9, posinf=1.0, neginf=1e-9)
        logp = torch.log(fused_probs + 1e-9)
        loss_task = nn.NLLLoss()(logp, labels)
        losses['task'] = loss_task.item()
        total_loss += loss_task
        
        # 2. 熵正则化（如果有融合权重）
        if 'weights' in output:
            W = output['weights']
            # 处理NaN/Inf
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
            ent_reg = (W * torch.log(W + 1e-9)).sum(dim=1).mean()
            losses['entropy_reg'] = ent_reg.item()
            total_loss += ent_reg * self.ent_reg_weight
        
        # 3. 权重一致性损失（权重应与各模态性能对齐）
        if 'weights' in output and 'probs' in output:
            W = output['weights']
            face_probs = output['probs'].get('face')
            of_probs = output['probs'].get('openface')
            audio_probs = output['probs'].get('audio')
            
            if face_probs is not None and of_probs is not None and audio_probs is not None:
                # 处理NaN/Inf
                face_probs = torch.nan_to_num(face_probs, nan=1e-9, posinf=1.0, neginf=1e-9)
                of_probs = torch.nan_to_num(of_probs, nan=1e-9, posinf=1.0, neginf=1e-9)
                audio_probs = torch.nan_to_num(audio_probs, nan=1e-9, posinf=1.0, neginf=1e-9)
                
                labels_col = labels.view(-1, 1)
                p_face = face_probs.gather(1, labels_col).squeeze(1)
                p_of = of_probs.gather(1, labels_col).squeeze(1)
                p_audio = audio_probs.gather(1, labels_col).squeeze(1)
                
                p_stack = torch.stack([p_face, p_of, p_audio], dim=1)
                # 处理NaN/Inf
                p_stack = torch.nan_to_num(p_stack, nan=0.0, posinf=0.0, neginf=0.0)
                p_norm = p_stack / (p_stack.sum(dim=1, keepdim=True) + 1e-9)
                
                loss_w_mse = F.mse_loss(W, p_norm.detach())
                losses['weight_mse'] = loss_w_mse.item()
                total_loss += loss_w_mse * self.w_mse_weight
        
        # 4. 各模态辅助损失（可选）
        for key in ['face', 'openface', 'audio', 'fa_au', 'fa_of']:
            if key in output['logits']:
                logits = output['logits'][key]
                # 处理NaN/Inf
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = self.criterion(logits, labels)
                losses[key] = loss.item()
                total_loss += self.loss_weights.get(key, 0.0) * loss
        
        # 最终处理总损失中的NaN/Inf
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if return_details:
            return total_loss, losses
        return total_loss, None
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前损失权重"""
        weights = self.loss_weights.copy()
        weights['ent_reg_weight'] = self.ent_reg_weight
        weights['w_mse_weight'] = self.w_mse_weight
        return weights
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新损失权重
        
        Args:
            new_weights: 新的权重字典
        """
        if 'ent_reg_weight' in new_weights:
            self.ent_reg_weight = new_weights.pop('ent_reg_weight')
        if 'w_mse_weight' in new_weights:
            self.w_mse_weight = new_weights.pop('w_mse_weight')
        self.loss_weights.update(new_weights)
