"""
损失计算模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class LossComputer:
    """多任务损失计算器"""
    
    def __init__(
        self, 
        loss_weights: Optional[Dict[str, float]] = None,
        ent_reg_weight: float = 0.2,
        w_mse_weight: float = 0.05
    ):
        if loss_weights is None:
            loss_weights = {
                'face': 0.2,
                'openface': 0.2,
                'audio': 0.2,
                'fa_au': 0.15,
                'fa_of': 0.15,
                'fused': 1.0
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
        """计算多任务损失
        
        损失组成：
        1. 融合损失 (fused): 主要分类损失
        2. 辅助损失 (face/openface/audio/fa_au/fa_of): 各模态分类损失
        3. 熵正则化 (entropy_reg): 鼓励权重明确性
        4. 权重一致性 (weight_mse): 权重与性能对齐
        """
        eps = 1e-7
        losses = {}
        total_loss = 0.0
        
        # 1. 融合损失（主要损失）
        fused_logits = output['logits']['fused']
        loss_fused = self.criterion(fused_logits, labels)
        losses['fused'] = loss_fused.item()
        total_loss += loss_fused * self.loss_weights.get('fused', 1.0)
        
        # 2. 辅助损失
        for key in ['face', 'openface', 'audio', 'fa_au', 'fa_of']:
            if key in output['logits']:
                logits = output['logits'][key]
                loss = self.criterion(logits, labels)
                losses[key] = loss.item()
                total_loss += self.loss_weights.get(key, 0.0) * loss
        
        # 3. 熵正则化
        if 'weights' in output:
            W = torch.clamp(output['weights'], min=eps, max=1.0-eps)
            ent_reg = (W * torch.log(W)).sum(dim=1).mean()
            losses['entropy_reg'] = ent_reg.item()
            total_loss += ent_reg * self.ent_reg_weight
        
        # 4. 权重一致性损失
        if 'weights' in output and 'probs' in output:
            W = output['weights']
            face_probs = output['probs'].get('face')
            of_probs = output['probs'].get('openface')
            audio_probs = output['probs'].get('audio')
            
            if face_probs is not None and of_probs is not None and audio_probs is not None:
                face_probs = torch.clamp(face_probs, min=eps, max=1.0-eps)
                of_probs = torch.clamp(of_probs, min=eps, max=1.0-eps)
                audio_probs = torch.clamp(audio_probs, min=eps, max=1.0-eps)
                
                labels_col = labels.view(-1, 1)
                p_face = face_probs.gather(1, labels_col).squeeze(1)
                p_of = of_probs.gather(1, labels_col).squeeze(1)
                p_audio = audio_probs.gather(1, labels_col).squeeze(1)
                
                p_stack = torch.stack([p_face, p_of, p_audio], dim=1)
                p_norm = p_stack / (p_stack.sum(dim=1, keepdim=True) + eps)
                
                loss_w_mse = F.mse_loss(W, p_norm.detach())
                losses['weight_mse'] = loss_w_mse.item()
                total_loss += loss_w_mse * self.w_mse_weight
        
        if return_details:
            return total_loss, losses
        return total_loss, None
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前损失权重"""
        return {
            **self.loss_weights,
            'ent_reg_weight': self.ent_reg_weight,
            'w_mse_weight': self.w_mse_weight
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新损失权重"""
        if 'ent_reg_weight' in new_weights:
            self.ent_reg_weight = new_weights.pop('ent_reg_weight')
        if 'w_mse_weight' in new_weights:
            self.w_mse_weight = new_weights.pop('w_mse_weight')
        self.loss_weights.update(new_weights)
