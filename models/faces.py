import torch, torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import glob, numpy as np

class FacesModel(nn.Module):
    def __init__(self, device='cpu', model_name='mobilenetv3_small_100'):
        super().__init__()
        self.device = device

        # 1. 骨干：num_classes=0 提取1024-D特征
        self.backbone = timm.create_model(model_name,
                                          pretrained=True,
                                          num_classes=0)   # (B, 1024)
        self.backbone.to(device)
        self.backbone.eval()          # 推理模式，BN固定

        # 2. 冻结骨干所有参数
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 3. 可训练逐层降维头
        self.fc_block = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 512 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512 // 2),
            nn.Dropout(0.3),
            nn.Linear(512 // 2, 2),
        ).to(device)

        # 4. 原预处理不变
        self.prep = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # ---------- 推理/特征 ----------
    def forward(self, x: torch.Tensor):
        with torch.no_grad(): 
            feat = self.backbone(x)  # (B, 1024)
        
        # 如果是单样本推理，临时切换到 eval 模式
        if x.size(0) == 1 and self.training:
            self.fc_block.eval()
            output = self.fc_block(feat)
            self.fc_block.train()
            return output
        
        return self.fc_block(feat)  # (B, 2)

    # ---------- 目录级平均概率（可选） ----------
    def predict_dir(self, face_dir: str):
        logits_sum = torch.zeros(2, device=self.device)
        n = 0
        for f in glob.glob(face_dir + "/*.jpg"):
            img = Image.open(f).convert('RGB')
            x = self.prep(img).unsqueeze(0).to(self.device)
            logits = self.forward(x).squeeze(0)   # (2,)
            logits_sum += logits
            n += 1
        if n == 0:
            return torch.tensor([0.5, 0.5])
        probs = torch.softmax(logits_sum / n, dim=0).cpu().numpy()
        return probs

    # ---------- 提取1024维特征（与原接口兼容） ----------
    def encode_dir(self, face_dir: str):
        feats = []
        for f in glob.glob(face_dir + "/*.jpg"):
            img = Image.open(f).convert('RGB')
            x = self.prep(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.backbone(x).squeeze(0).cpu().numpy()  # (1024,)
            feats.append(feat)
        if len(feats) == 0:
            return np.zeros(1024, dtype=np.float32)
        return np.stack(feats).mean(axis=0)