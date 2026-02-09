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
    
    def forward(self, x, return_features=False):
        """前向传播
        
        Args:
            x: OpenFace 特征张量，shape (B, 714) 或 (B*T, 714)
            return_features: 是否返回中间特征而不是logits
        
        Returns:
            如果 return_features=True: (B, 714) 特征（直接返回输入）
            如果 return_features=False: (B, num_classes) logits
        """
        # 如果只需要特征，直接返回输入（因为输入已经是OpenFace特征）
        if return_features:
            return x
        
        return self.fc(x)
    
    def get_features(self, x):
        """提取714维特征（便捷方法）
        
        Args:
            x: OpenFace 特征张量，shape (B, 714)
        
        Returns:
            特征张量 (B, 714)
        """
        return self.forward(x, return_features=True)
