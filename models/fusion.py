"""
多模态融合模型
基于Transformer和Cross-Attention的高级融合策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.faces import FacesModel
from models.openface import OpenfaceModel
from models.audio import AudioModel
from typing import Optional, List, Tuple, Dict
import numpy as np


class Transpose(nn.Module):
    """转置模块"""
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class W(nn.Module):
    """动态权重学习模块
    
    学习三个模态的重要性权重
    """
    def __init__(self, hidden_dim):
        super(W, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = hidden_dim * 3  # 三个模态拼接后的维度
        
        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )
    
    def forward(self, U_face, U_of, U_audio):
        batch_size = U_face.size(0)
        assert U_face.shape == (batch_size, self.hidden_dim), f"U_face shape mismatch: {U_face.shape}"
        assert U_of.shape == (batch_size, self.hidden_dim), f"U_of shape mismatch: {U_of.shape}"
        assert U_audio.shape == (batch_size, self.hidden_dim), f"U_audio shape mismatch: {U_audio.shape}"
        
        concatenated = torch.cat([U_face, U_of, U_audio], dim=1)  # shape: (B, hidden_dim*3)
        output = self.fc(concatenated)  # shape: (B, 3)
        return output


class CrossAttentionBlock(nn.Module):
    """跨模态注意力模块"""
    def __init__(self, dim_q=256, dim_kv=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_q,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, q, k, v):
        # q: (B, T_q, 256) or (B, 256) if we unsqueeze
        # k,v: (B, T_k, 256)
        if q.dim() == 2:
            q = q.unsqueeze(1)  # (B, 1, 256)
        out, _ = self.attn(q, k, v)
        return out.squeeze(1)  # (B, 256)


class UniEncoder(nn.Module):
    """单模态编码器
    
    使用Transformer对每个模态的时序特征进行编码
    """
    def __init__(self, d_in, d_model=256, nhead=4, num_layers=2, ff_hidden=512, dropout=0.1):
        super().__init__()
        self.project_in = nn.Identity() if d_in == d_model else nn.Linear(d_in, d_model)
        # nn.Identity()（恒等映射）

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        # x: (B, T, d_in)
        x = self.project_in(x)
        x = self.encoder(x)  # (B, T, d_model)
        return x


class FusionModel(nn.Module):
    """多模态融合模型
    
    特点：
    1. 直接使用特征序列高维特征
    2. Transformer时序建模
    3. Cross-Attention跨模态交互
    4. 动态权重学习
    """
    def __init__(
        self,
        device: str = 'cuda',
        face_checkpoint: Optional[str] = None,
        openface_checkpoint: Optional[str] = None,
        audio_checkpoint: Optional[str] = None,
        weights: Optional[List[float]] = None,
        freeze_backbones: bool = True,
        num_classes: int = 2,
        visual_hidden: int = 256,
        fusion_hidden: int = 128,
        dropout: float = 0.3,
        uni_layers: int = 2,
        uni_nhead: int = 4,
        com_heads: int = 4
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.ds = visual_hidden
        self.hidden = visual_hidden
        self.K = num_classes
        
        # --- 子模型 ---
        self.face = FacesModel(device=device)
        self.openface = OpenfaceModel(input_dim=714, num_classes=num_classes)
        self.audio = AudioModel(use_mfcc=False, drop=dropout)
        
        # 兼容性别名
        self.face_model = self.face
        self.openface_model = self.openface
        self.audio_model = self.audio
        
        # --- 加载检查点 ---
        def _try_load(module, ckpt_path):
            if not ckpt_path:
                return
            try:
                sd = torch.load(ckpt_path, map_location='cpu')
                if isinstance(sd, dict) and ('state_dict' in sd or 'model_state_dict' in sd):
                    key = 'state_dict' if 'state_dict' in sd else 'model_state_dict'
                    sd_inner = sd[key]
                    try:
                        module.load_state_dict(sd_inner)
                    except Exception:
                        module.load_state_dict(sd_inner, strict=False)
                else:
                    try:
                        module.load_state_dict(sd)
                    except Exception:
                        module.load_state_dict(sd, strict=False)
                print(f"[FusionModel] loaded checkpoint {ckpt_path} into {module.__class__.__name__}")
            except Exception as e:
                print(f"[FusionModel] warning: failed to load checkpoint {ckpt_path}: {e}")
        
        _try_load(self.face, face_checkpoint)
        _try_load(self.openface, openface_checkpoint)
        _try_load(self.audio, audio_checkpoint)
        
        # 冻结backbone
        if freeze_backbones:
            for name, p in self.face.named_parameters():
                if 'fc_block' not in name:
                    p.requires_grad = False
        
        # 特征维度
        self.face_feat_dim = 1024
        self.openface_feat_dim = 714
        self.audio_feat_dim = 768
        
        # --- E_uni: 单模态Transformer编码器 ---
        self.E_uni_face = nn.Sequential(
            nn.Linear(self.face_feat_dim, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            UniEncoder(d_in=self.ds, d_model=self.ds, nhead=uni_nhead, 
                      num_layers=uni_layers, ff_hidden=self.ds*2, dropout=dropout),
            Transpose(1, 2),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(self.ds, self.ds),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.E_uni_of = nn.Sequential(
            nn.Linear(self.openface_feat_dim, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            UniEncoder(d_in=self.ds, d_model=self.ds, nhead=uni_nhead,
                      num_layers=uni_layers, ff_hidden=self.ds*2, dropout=dropout),
            Transpose(1, 2),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(self.ds, self.ds),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.E_uni_audio = nn.Sequential(
            nn.Linear(self.audio_feat_dim, self.ds*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.ds*2, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # --- E_com: 跨模态注意力 ---
        self.E_com1_fa = nn.Sequential(
            nn.Linear(self.face_feat_dim, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.E_com2_fa = nn.Sequential(
            nn.Linear(self.face_feat_dim, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.E_com_au = nn.Sequential(
            nn.Linear(self.audio_feat_dim, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.E_com_of = nn.Sequential(
            nn.Linear(self.openface_feat_dim, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Transpose(1, 2),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(self.ds, self.ds),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.E_com_fa_au = CrossAttentionBlock(dim_q=self.ds, dim_kv=self.ds, num_heads=com_heads)
        self.E_com_fa_of = CrossAttentionBlock(dim_q=self.ds, dim_kv=self.ds, num_heads=com_heads)
        
        # 动态权重学习
        self.W = W(self.ds)
        
        # 融合分类头
        self.prob_fa_au = nn.Sequential(
            nn.Linear(self.ds*3, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.ds, 2)
        )
        
        self.prob_fa_of = nn.Sequential(
            nn.Linear(self.ds*3, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.ds, 2)
        )
        
        # 时序模块（GRU）
        self.T = nn.GRU(self.ds, self.ds // 2, num_layers=1, 
                       bidirectional=True, batch_first=True)
        
        # 视觉分类头
        self.visual_fc = nn.Sequential(
            nn.Linear(self.ds, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )
        
        # 音频投影
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_feat_dim, visual_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(visual_hidden)
        )
        
        # 融合头（兼容性）
        self.fusion_fc = nn.Sequential(
            nn.Linear(visual_hidden * 3, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )
        
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_parameter('fusion_w', nn.Parameter(torch.log(w)))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, faces, openfaces, audios):
        """前向传播
        
        Args:
            faces: 人脸图像 (B, T, C, H, W)
            openfaces: OpenFace特征 (B, T, 714)
            audios: 音频特征 (B, 768)
        
        Returns:
            dict: 包含logits、probs、weights等信息
        """
        device = next(self.parameters()).device
        
        # --- 人脸预处理 ---
        faces = faces.to(device).float()
        if faces.max() > 2.0:
            faces = faces / 255.0
        
        if faces.ndim == 5:
            if faces.shape[2] == 3:
                B, T, C, H, W = faces.shape
                faces_proc = faces.view(B * T, C, H, W)
            elif faces.shape[-1] == 3:
                B, T, H, W, C = faces.shape
                faces_proc = faces.view(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"Unexpected faces shape: {faces.shape}")
            
            faces_proc = nn.functional.interpolate(
                faces_proc, size=(160, 160), mode='bilinear', align_corners=False
            )  # (B*T, C, 160, 160)
        
        # --- 提取人脸特征（关键：直接访问backbone！）---
        with torch.no_grad():
            face_back_feats = self.face.backbone(faces_proc)  # (B*T, 1024)
            face_frame_logits = self.face.fc_block(face_back_feats)
        
        face_feats = face_back_feats.view(B, T, -1)  # (B, T, 1024)
        face_frame_logits = face_frame_logits.view(B, T, -1)
        logits_face = face_frame_logits.mean(dim=1)
        probs_face = F.softmax(logits_face, dim=1)
        
        # --- OpenFace处理 ---
        of = openfaces.to(device, dtype=torch.float32)
        if of.ndim == 3:
            B_of, T_of, D = of.shape
            if B_of != B:
                raise ValueError(f"Batch size mismatch: faces ({B}) vs openfaces ({B_of})")
            
            # 如果帧数不匹配，插值到相同长度
            if T_of != T:
                # 转换为 (B, D, T_of) 进行插值
                of = of.permute(0, 2, 1)  # (B, 714, T_of)
                of = F.interpolate(of, size=T, mode='linear', align_corners=False)
                of = of.permute(0, 2, 1)  # (B, T, 714)
            
            of_feats = of
            of_proc = of.view(B * T, D)
            of_frame_logits = self.openface(of_proc).view(B, T, -1)
            logits_of = of_frame_logits.mean(dim=1)
            probs_of = F.softmax(logits_of, dim=1)
        
        # --- 音频处理 ---
        logits_au = self.audio(audios)
        probs_au = F.softmax(logits_au, dim=1)
        
        # --- 单模态编码 ---
        U_face = self.E_uni_face(face_feats)    # (B, hidden)
        U_of = self.E_uni_of(of_feats)          # (B, hidden)
        U_audio = self.E_uni_audio(audios)      # (B, hidden)
        
        # --- 跨模态注意力 ---
        X_face1 = self.E_com1_fa(face_feats)  # (B, T, 256)
        X_au1 = self.E_com_au(audios)  # (B, 256)
        X_face2 = self.E_com2_fa(face_feats)  # (B, T, 256)
        X_of1 = self.E_com_of(of_feats)  # (B, 256)
        
        # 音频需要扩展时间维度以匹配人脸
        X_au1 = X_au1.unsqueeze(1)  # (B, 1, 256)
        
        C_fa_au = self.E_com_fa_au(X_au1, X_face1, X_face1)  # (B, 256)
        C_fa_of = self.E_com_fa_of(X_of1, X_face2, X_face2)  # (B, 256)
        
        # --- 动态权重 ---
        W_scores = self.W(U_face, U_of, U_audio)
        W = F.softmax(W_scores, dim=1)
        
        # 加权
        U_face = W[:, 0:1] * U_face
        U_of = W[:, 1:2] * U_of
        U_audio = W[:, 2:3] * U_audio
        
        # --- 融合 ---
        fuse_fa_au = torch.cat([U_face, U_audio, C_fa_au], dim=-1)  # (B, 3*hidden)
        fuse_fa_of = torch.cat([U_face, U_of, C_fa_of], dim=-1)     # (B, 3*hidden)
        
        # 余弦相似度（添加eps防止除零）
        eps = 1e-8
        cos_fa_fa_au = F.cosine_similarity(U_face, C_fa_au, dim=1, eps=eps)
        cos_fa_fa_of = F.cosine_similarity(U_face, C_fa_of, dim=1, eps=eps)
        cos_au_fa_au = F.cosine_similarity(U_audio, C_fa_au, dim=1, eps=eps)
        cos_of_fa_of = F.cosine_similarity(U_of, C_fa_of, dim=1, eps=eps)
        
        # 分类
        logits_fa_au = self.prob_fa_au(fuse_fa_au)
        logits_fa_of = self.prob_fa_of(fuse_fa_of)
        probs_fa_au = F.softmax(logits_fa_au, dim=1)
        probs_fa_of = F.softmax(logits_fa_of, dim=1)
        
        # 最终融合概率和logits
        # 使用加权平均的logits而不是概率
        fused_logits = W[:, 2:3]*logits_fa_au + W[:, 1:2]*logits_fa_of + W[:, 0:1]*logits_face
        fused_probs = F.softmax(fused_logits, dim=1)
        
        ret = {
            'logits': {
                'face': logits_face,
                'openface': logits_of,
                'audio': logits_au,
                'fa_au': logits_fa_au,
                'fa_of': logits_fa_of,
                'fused': fused_logits  # 添加融合logits
            },
            'probs': {
                'face': probs_face,
                'openface': probs_of,
                'audio': probs_au,
                'fa_au': probs_fa_au,
                'fa_of': probs_fa_of,
                'fused': fused_probs
            },
            'weights': W,
            'cos': {
                'fa_fa_au': cos_fa_fa_au,
                'fa_fa_of': cos_fa_fa_of,
                'au_fa_au': cos_au_fa_au,
                'of_fa_of': cos_of_fa_of,
            }
        }
        
        return ret
