"""测试完整的 FusionModel"""
import torch
from plan import FusionModel

def test_fusion_model():
    print("=" * 60)
    print("🔍 测试完整的 FusionModel...")
    print("=" * 60)
    
    try:
        # 1. 初始化模型
        print("\n1️⃣  初始化模型...")
        model = FusionModel(
            device='cpu',
            num_classes=2,
            visual_hidden=256,
            fusion_hidden=128,
            freeze_backbones=True
        )
        model.eval()
        print("   ✅ 模型初始化成功")
        
        # 2. 准备测试数据
        print("\n2️⃣  准备测试数据...")
        batch_size = 2
        num_frames = 16  # 可以是任意帧数
        
        # 面部图像: (B, T, C, H, W)
        faces = torch.randn(batch_size, num_frames, 3, 160, 160)
        print(f"   faces shape: {faces.shape}")
        
        # OpenFace 特征: (B, T, 714)
        openfaces = torch.randn(batch_size, num_frames, 714)
        print(f"   openfaces shape: {openfaces.shape}")
        
        # 音频特征: (B, 768)
        audios = torch.randn(batch_size, 768)
        print(f"   audios shape: {audios.shape}")
        
        # 3. 前向传播
        print("\n3️⃣  执行前向传播...")
        with torch.no_grad():
            output = model(faces, openfaces, audios)
        
        # 4. 检查输出
        print("\n4️⃣  检查输出结果...")
        print(f"   ✅ logits keys: {list(output['logits'].keys())}")
        print(f"   ✅ probs keys: {list(output['probs'].keys())}")
        
        # 5. 显示融合结果
        print("\n5️⃣  融合结果:")
        fused_probs = output['probs']['fused']
        weights = output['weights']
        
        print(f"   融合概率 shape: {fused_probs.shape}")
        print(f"   样本1: 真话={fused_probs[0, 0]:.4f}, 谎言={fused_probs[0, 1]:.4f}")
        print(f"   样本2: 真话={fused_probs[1, 0]:.4f}, 谎言={fused_probs[1, 1]:.4f}")
        
        print(f"\n   模态权重 shape: {weights.shape}")
        print(f"   样本1: face={weights[0, 0]:.4f}, openface={weights[0, 1]:.4f}, audio={weights[0, 2]:.4f}")
        print(f"   样本2: face={weights[1, 0]:.4f}, openface={weights[1, 1]:.4f}, audio={weights[1, 2]:.4f}")
        
        # 6. 显示各模态的预测
        print("\n6️⃣  各模态单独预测:")
        for modality in ['face', 'openface', 'audio']:
            probs = output['probs'][modality]
            print(f"   {modality:10s}: 样本1=[{probs[0, 0]:.4f}, {probs[0, 1]:.4f}], "
                  f"样本2=[{probs[1, 0]:.4f}, {probs[1, 1]:.4f}]")
        
        print("\n" + "=" * 60)
        print("🎉 FusionModel 测试成功！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_fusion_model()
