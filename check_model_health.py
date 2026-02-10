"""
检查模型健康状态 - 诊断NaN问题
"""

import torch
from pathlib import Path
from models.fusion import FusionModel

def check_model_weights(checkpoint_path):
    """检查模型权重是否包含NaN或Inf"""
    print(f"\n检查检查点: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"[ERROR] 检查点不存在: {checkpoint_path}")
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 检查模型状态
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    has_nan = False
    has_inf = False
    
    print("\n检查权重...")
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            print(f"  [NaN] {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"  [Inf] {name}")
            has_inf = True
    
    if not has_nan and not has_inf:
        print("  ✓ 所有权重正常")
        return True
    else:
        print(f"\n[ERROR] 模型包含异常值:")
        print(f"  - NaN: {has_nan}")
        print(f"  - Inf: {has_inf}")
        return False


def test_forward_pass():
    """测试前向传播是否产生NaN"""
    print("\n测试前向传播...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")
    
    model = FusionModel(
        device=device,
        freeze_backbones=True,
        dropout=0.5
    )
    model.to(device)
    model.eval()
    
    # 创建测试数据（确保在正确设备上）
    batch_size = 2
    faces = torch.randn(batch_size, 3, 3, 224, 224, device=device)
    openfaces = torch.randn(batch_size, 3, 714, device=device)  # 正确的OpenFace维度
    audios = torch.randn(batch_size, 3, 1024, device=device)
    
    with torch.no_grad():
        output = model(faces, openfaces, audios)
    
    # 检查输出
    print("\n检查输出...")
    for key, value in output['logits'].items():
        if torch.isnan(value).any():
            print(f"  [NaN] logits[{key}]")
        elif torch.isinf(value).any():
            print(f"  [Inf] logits[{key}]")
        else:
            print(f"  ✓ logits[{key}]: {value.shape}")
    
    if 'weights' in output:
        weights = output['weights']
        if torch.isnan(weights).any():
            print(f"  [NaN] weights")
        elif torch.isinf(weights).any():
            print(f"  [Inf] weights")
        else:
            print(f"  ✓ weights: {weights.shape}, mean={weights.mean().item():.4f}")
    
    print("\n✓ 前向传播测试完成")


if __name__ == '__main__':
    # 检查最新模型
    latest_path = 'checkpoints/latest_model.pth'
    best_path = 'checkpoints/best_model.pth'
    
    if Path(latest_path).exists():
        check_model_weights(latest_path)
    
    if Path(best_path).exists():
        check_model_weights(best_path)
    
    # 测试新模型
    test_forward_pass()
