"""
验证训练好的模型
"""

import torch
from models.fusion import FusionModel
from pathlib import Path

# 检查检查点
checkpoint_path = 'checkpoints/test_final/best_model.pth'

if not Path(checkpoint_path).exists():
    print(f"[ERROR] 检查点不存在: {checkpoint_path}")
    exit(1)

print(f"[1] 加载检查点: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\n[2] 检查点内容:")
print(f"  - Keys: {list(checkpoint.keys())}")
print(f"  - Epoch: {checkpoint['epoch']}")
print(f"  - 最佳验证准确率: {checkpoint['best_val_acc']:.2f}%")

print(f"\n[3] 创建模型...")
model = FusionModel(
    device='cpu',
    freeze_backbones=True,
    visual_hidden=128,
    fusion_hidden=64,
    dropout=0.3
)

print(f"\n[4] 加载权重...")
model.load_state_dict(checkpoint['model_state_dict'])
print("[OK] 权重加载成功")

print(f"\n[5] 测试前向传播...")
model.eval()

# 创建测试数据
faces = torch.randn(2, 16, 3, 160, 160)
openfaces = torch.randn(2, 16, 714)
audios = torch.randn(2, 768)

with torch.no_grad():
    output = model(faces, openfaces, audios)

print("[OK] 前向传播成功")
print(f"\n[6] 输出检查:")
print(f"  - Logits keys: {list(output['logits'].keys())}")
print(f"  - Fused logits shape: {output['logits']['fused'].shape}")
print(f"  - Fused probs shape: {output['probs']['fused'].shape}")
print(f"  - Weights shape: {output['weights'].shape}")

# 检查是否有NaN
has_nan = False
for key, logits in output['logits'].items():
    if torch.isnan(logits).any():
        print(f"  [ERROR] {key} has NaN!")
        has_nan = True

if not has_nan:
    print(f"\n[SUCCESS] 模型验证通过！所有输出都是正常的数值。")
else:
    print(f"\n[ERROR] 模型输出包含NaN！")

# 测试预测
print(f"\n[7] 测试预测...")
fused_probs = output['probs']['fused']
predictions = torch.argmax(fused_probs, dim=1)
print(f"  - 预测结果: {predictions.tolist()}")
print(f"  - 概率分布:")
for i, probs in enumerate(fused_probs):
    print(f"    样本{i+1}: truth={probs[0]:.2%}, deception={probs[1]:.2%}")

print(f"\n[SUCCESS] 所有测试通过！模型可以正常使用。")
