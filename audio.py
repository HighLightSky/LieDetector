# audio_model.py
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ---------- 1. 独立 Wav2Vec2 编码器 ----------
class Wav2Vec2Encoder:
    def __init__(self, device=None, model_name='facebook/wav2vec2-base'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Wav2Vec2Processor.from_pretrained("E:\solo\pretrained_models\wav2vec2_base")
        self.model = Wav2Vec2Model.from_pretrained("E:\solo\pretrained_models\wav2vec2_base").to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False  # 默认冻结

    @torch.no_grad()
    def __call__(self, wav_path):
        """输入：wav路径；输出：1×768 numpy向量"""
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0).numpy()          # 转单通道
        inputs = self.processor(wav, sampling_rate=16_000, return_tensors='pt', padding=True)
        inputs = inputs.input_values.to(self.device)
        hidden = self.model(inputs).last_hidden_state       # [1, T, 768]
        return hidden.mean(dim=1).squeeze(0).cpu().numpy()  # [768]

# ---------- 2. 可选 MFCC 工具 ----------
def mfcc_vector(wav_path, n_mfcc=40, sr=16_000):
    y, _ = librosa.load(wav_path, sr=sr)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # [40, T]
    return mf.mean(axis=1)                                  # [40]

# ---------- 3.  deception 分类网络 ----------
class AudioModel(nn.Module):    
    def __init__(self,
                 use_mfcc=False,
                 drop=0.2):
        super().__init__()
        self.use_mfcc = use_mfcc
        w2v_dim, mfcc_dim = 768, 40 if use_mfcc else 0
        in_dim = w2v_dim + mfcc_dim

        # self.encoder = Wav2Vec2Encoder()   # 实例化编码器

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),

            nn.Linear(128, 2)
        )

    def forward(self, audio_features):
        """
        audio_features: torch.Tensor (B, 768)
        return: logits [B, 2]
        """

        return self.fc(audio_features)        # [B, 2]

# ---------- 4. 训练/推理示例 ----------
if __name__ == '__main__':
    model = AudioModel(use_mfcc=False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    # 假设你已有 (wav_path, label) 列表
    dummy_data = [('fake1.wav', 0), ('fake2.wav', 1)]  # 请换成真实路径

    for epoch in range(3):        # 仅演示
        for wav, y in dummy_data:
            logits = model(wav)   # [1, 2]
            loss = loss_fn(logits, torch.tensor([y]).long().cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'loss={loss.item():.4f}')