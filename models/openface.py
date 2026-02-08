import torch
import torch.nn as nn


class OpenfaceModel(nn.Module):
    """OpenFace 特征分类模型
    
    接收 OpenFace 提取的面部特征（714维），输出分类结果
    """
    def __init__(self, input_dim=714, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: OpenFace 特征张量，shape (B, 714) 或 (B*T, 714)
        
        Returns:
            logits: 分类 logits，shape (B, num_classes) 或 (B*T, num_classes)
        """
        return self.fc(x)
