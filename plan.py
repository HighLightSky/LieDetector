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
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


#GRU->transformer
class W(nn.Module):
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
            q = q.unsqueeze(1) # (B, 1, 256)
        
        out, _ = self.attn(q, k, v)
        return out.squeeze(1)  # (B, 256)
class UniEncoder(nn.Module):
    """Per-frame encoder implemented as a small transformer encoder stack.
    Accepts batch-first tensors (B, T, D_in) and outputs (B, T, D_out)
    If D_in != D_out, a linear projection is applied.
    """
    def __init__(self, d_in, d_model=256, nhead=4, num_layers=2, ff_hidden=512, dropout=0.1):
        super().__init__()
        self.project_in = nn.Identity() if d_in == d_model else nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=ff_hidden, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        # x: (B, T, d_in)
        x = self.project_in(x)
        x = self.encoder(x)  # (B, T, d_model)
        return x
class FusionModel(nn.Module):
    def __init__(self,
                 device: str = 'cuda',
                 face_checkpoint: Optional[str] = None,
                 openface_checkpoint: Optional[str] = None,
                 audio_checkpoint: Optional[str] = None,
                 weights: Optional[List[float]] = None,
                 freeze_backbones: bool = True,
                 wav2vec_model_or_path: Optional[str] = None,
                 num_classes: int = 2,
                 visual_hidden: int = 256,
                 fusion_hidden: int = 128,
                 dropout: float = 0.3,
                 ent_lambda: float = 0.01,
                 mmd_sigma: float = 1.0,
                 gmm_iters: int = 10,
                 sinkhorn_iters: int = 100,
                 uni_layers: int = 2,
                 uni_nhead: int = 4,
                 com_heads: int = 4):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.wav2vec_model_or_path = wav2vec_model_or_path
        self.ds = visual_hidden
        self.hidden = visual_hidden
        self.K = num_classes
        self.ent_lambda = ent_lambda
        self.mmd_sigma = mmd_sigma
        self.gmm_iters = gmm_iters
        self.sinkhorn_iters = sinkhorn_iters
        self.w_tau = 1.5
        self.w_mix = 0.1

        # --- submodules (assume these classes exist elsewhere) -----------------
        # Keep names for backward compatibility
        self.face = FacesModel(device=device)
        self.openface = OpenfaceModel(input_dim=714, num_classes=num_classes)
        self.audio = AudioModel(use_mfcc=False, drop=dropout)
        self.face_model = self.face
        self.openface_model = self.openface
        self.audio_model = self.audio

        # --- try load checkpoints (robust) ------------------------------------
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
                print(f"[FusionModel_v2] loaded checkpoint {ckpt_path} into {module.__class__.__name__}")
            except Exception as e:
                print(f"[FusionModel_v2] warning: failed to load checkpoint {ckpt_path}: {e}")

        _try_load(self.face, face_checkpoint)
        _try_load(self.openface, openface_checkpoint)
        _try_load(self.audio, audio_checkpoint)

        if freeze_backbones:
            for name, p in self.face.named_parameters():
                if 'fc_block' not in name:
                    p.requires_grad = False

        # feature dims (keep same defaults)
        self.face_feat_dim = 1024
        self.openface_feat_dim = 714
        self.audio_feat_dim = 768

        # --- E_uni: per-modality Transformer stacks ---
        self.E_uni_face = nn.Sequential(
            nn.Linear(self.face_feat_dim, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            UniEncoder(d_in=self.ds, d_model=self.hidden, nhead=uni_nhead, num_layers=uni_layers,
                                     ff_hidden=self.hidden*2, dropout=dropout),
            Transpose(1, 2),
            nn.AdaptiveAvgPool1d(output_size=1), 
            nn.Flatten(),
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Dropout(0.1)
        )
        
        self.E_uni_of = nn.Sequential(
            nn.Linear(self.openface_feat_dim, self.ds),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            UniEncoder(d_in=self.ds, d_model=self.hidden, nhead=uni_nhead, num_layers=uni_layers,
                                   ff_hidden=self.hidden*2, dropout=dropout),
            Transpose(1, 2),
            nn.AdaptiveAvgPool1d(output_size=1), 
            nn.Flatten(),
            nn.Linear(256, 256), 
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

        # --- E_com: shared cross-attention block (applied both directions) ---
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
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Dropout(0.1)
        )

        self.E_com_fa_au = CrossAttentionBlock(dim_q=self.hidden, dim_kv=self.hidden, num_heads=com_heads)
        self.E_com_fa_of = CrossAttentionBlock(dim_q=self.hidden, dim_kv=self.hidden, num_heads=com_heads)


        self.W = W(self.hidden)

        self.prob_fa_au = nn.Sequential(
            nn.Linear(self.hidden*3, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 2)
        )

        self.prob_fa_of = nn.Sequential(
            nn.Linear(self.hidden*3, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 2)
        )

        # --- Temporal module (now the dedicated place for GRU) ---------------
        # GRU takes fused per-frame embeddings and models temporal dependency
        # We'll use a bidirectional GRU and then mean-pool over time
        self.T = nn.GRU(self.hidden, self.hidden // 2, num_layers=1, bidirectional=True, batch_first=True)

        # visual classification head (input dim: hidden*2 after GRU mean pooling)
        self.visual_fc = nn.Sequential(
            nn.Linear(self.hidden , fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

        # audio projection
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_feat_dim, visual_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(visual_hidden)
        )

        # fusion head kept for compatibility
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

        # initialization
        self._init_weights()
        self._lazy_encoder = None

    def _init_weights(self):
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

    def _get_lazy_encoder(self):
        if self._lazy_encoder is None:
            self._lazy_encoder = Wav2Vec2Encoder(device=self.device, model_name_or_path=self.wav2vec_model_or_path or "facebook/wav2vec2-base")
        return self._lazy_encoder

    # --- keep numeric utility funcs from previous model (matrix_sqrt, fit_gmm, etc.) ---
    def matrix_sqrt(self, M):
        d = M.shape[-1]
        I = torch.eye(d, device=M.device)
        M = (M + M.transpose(-1, -2)) * 0.5
        M = M.to(torch.float64)
        M = M + 1e-6 * I
        try:
            vals, vecs = torch.linalg.eigh(M)
        except Exception:
            vals, vecs = torch.linalg.eigh(M.cpu())
            vals = vals.to(M.device)
            vecs = vecs.to(M.device)
        vals = vals.clamp(min=0)
        sqrt_vals = vals.sqrt()
        S = vecs @ torch.diag_embed(sqrt_vals) @ vecs.transpose(-1, -2)
        return S.to(torch.float32)

    def fit_gmm(self, feats, K, max_iter, eps=1e-6):
        feats = feats.to(torch.float32)
        device = feats.device
        N, d = feats.shape
        idx = torch.randperm(N, device=device)[:K]
        mu = feats[idx]
        sigma = torch.eye(d, device=device).unsqueeze(0).repeat(K, 1, 1) * 1.0
        pi = torch.ones(K, device=device) / K

        for it in range(max_iter):
            diff = feats.unsqueeze(1) - mu.unsqueeze(0)  # (N, K, d)
            inv_sigma = torch.linalg.inv(sigma + eps * torch.eye(d, device=device))
            mahal = torch.einsum('nkd,kde,nke->nk', diff, inv_sigma, diff)
            det = torch.det(sigma + eps * torch.eye(d, device=device))
            prob = torch.exp(-0.5 * mahal) / ((2 * math.pi) ** (d / 2) * det.sqrt().unsqueeze(0))
            w = pi.unsqueeze(0) * prob
            w_sum = w.sum(1, keepdim=True)
            w = w / (w_sum + eps)

            pi = w.mean(0)
            mu = torch.einsum('nk,nd->kd', w, feats) / (w.sum(0).unsqueeze(-1) + eps)
            diff = feats.unsqueeze(1) - mu.unsqueeze(0)
            sigma = torch.einsum('nk,nkd,nke->kde', w, diff, diff) / (w.sum(0).unsqueeze(-1).unsqueeze(-1) + eps)

        return pi, mu, sigma, w

    def sinkhorn(self, a, b, C, reg, num_iter):
        device = a.device
        K = torch.exp(-C / reg)
        u = torch.ones_like(a) / a.shape[0]
        v = torch.ones_like(b) / b.shape[0]
        for _ in range(num_iter):
            v = b / (K.t() @ u + 1e-9)
            u = a / (K @ v + 1e-9)
        T = u.unsqueeze(1) * K * v.unsqueeze(0)
        return T

    def compute_stats(self, feats):
        feats = feats.to(torch.float32)
        N, d = feats.shape
        mu = feats.mean(0)
        centered = feats - mu
        cov = (centered.t() @ centered) / (N - 1 + 1e-6)
        std = cov.diag().sqrt() + 1e-6
        skew = ((centered / std.unsqueeze(0)) ** 3).mean(0)
        return mu, cov, skew

    def mmd(self, x, y, sigma):
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        def kernel(a, b):
            return torch.exp(-torch.cdist(a, b) ** 2 / (2 * sigma ** 2))
        kxx = kernel(x, x).mean()
        kyy = kernel(y, y).mean()
        kxy = kernel(x, y).mean()
        return kxx + kyy - 2 * kxy

    def forward(self, faces, openfaces, audios):
        device = next(self.parameters()).device

        # --- faces preprocessing ---
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
        faces_proc = nn.functional.interpolate(faces_proc, size=(160, 160), mode='bilinear', align_corners=False) # (B*T, C, 160, 160)

        # --- face backbone (kept as before) ---
        with torch.no_grad():
            face_back_feats = self.face.backbone(faces_proc)
        face_frame_logits = self.face.fc_block(face_back_feats)
        face_feats = face_back_feats.view(B, T, -1)
        face_frame_logits = face_frame_logits.view(B, T, -1)
        logits_face = face_frame_logits.mean(dim=1)
        probs_face = F.softmax(logits_face, dim=1)

        # --- openface processing ---
        of = openfaces.to(device, dtype=torch.float32)
        if of.ndim == 3:
            B_of, T_of, D = of.shape
            if not (B_of == B and T_of == T):
                raise ValueError(f"Mismatch shapes: faces ({B},{T}) vs openfaces ({B_of},{T_of})")
            of_feats = of
            of_proc = of.view(B * T, D)
            of_frame_logits = self.openface(of_proc).view(B, T, -1)
        logits_of = of_frame_logits.mean(dim=1)
        probs_of = F.softmax(logits_of, dim=1)

        logits_au = self.audio(audios)
        probs_au = F.softmax(logits_au, dim=1)


        U_face = self.E_uni_face(face_feats) # (B, hidden)
        U_of = self.E_uni_of(of_feats)      # (B, hidden)
        U_audio = self.E_uni_audio(audios)  # (B, hidden)


        # --- E_com: shared cross-attention applied both directions ---
        # face attends to of, of attends to face (weights shared via same module)
        X_face1 = self.E_com1_fa(face_feats)
        X_au1 = self.E_com_au(audios)
        X_face2 = self.E_com2_fa(face_feats)
        X_of1 = self.E_com_of(of_feats)
        C_fa_au = self.E_com_fa_au(X_au1, X_face1, X_face1)
        C_fa_of = self.E_com_fa_of(X_of1, X_face2, X_face2)

        W_scores = self.W(U_face, U_of, U_audio)
        W = F.softmax(W_scores, dim=1)
        # W = F.softmax(W_scores / self.w_tau, dim=1)
        # if self.w_mix > 0:
        #     W = (1.0 - self.w_mix) * W + self.w_mix * (1.0 / 3.0)

        U_face = W[:, 0:1]*U_face
        U_of = W[:, 1:2]*U_of
        U_audio = W[:, 2:3]*U_audio

        fuse_fa_au = torch.cat([U_face, U_audio, C_fa_au], dim=-1) # (B, 3*hidden)
        fuse_fa_of = torch.cat([U_face, U_of, C_fa_of], dim=-1) # (B, 3*hidden)

        cos_fa_fa_au = torch.nn.functional.cosine_similarity(U_face, C_fa_au, dim=1)
        cos_fa_fa_of = torch.nn.functional.cosine_similarity(U_face, C_fa_of, dim=1)
        cos_au_fa_au = torch.nn.functional.cosine_similarity(U_audio, C_fa_au, dim=1)
        cos_of_fa_of = torch.nn.functional.cosine_similarity(U_of, C_fa_of, dim=1)


        logits_fa_au = self.prob_fa_au(fuse_fa_au)
        logits_fa_of = self.prob_fa_of(fuse_fa_of)
        probs_fa_au = F.softmax(logits_fa_au, dim=1)
        probs_fa_of = F.softmax(logits_fa_of, dim=1)
        fused_probs = W[:, 2:3]*probs_fa_au + W[:, 1:2]*probs_fa_of + W[:, 0:1]*probs_face
        # fused_probs = torch.cat([1 - fused_logits, fused_logits], dim=1)
        

        ret = {
            'logits': {
                'face': logits_face,
                'openface': logits_of,
                'audio': logits_au,
                'fa_au': logits_fa_au,
                'fa_of': logits_fa_of,
                # 'fused': fused_logits
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
# End of file
