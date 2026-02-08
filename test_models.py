"""测试预训练模型是否正确加载"""
import torch
from models.faces import FacesModel
from models.audio import Wav2Vec2Encoder

def test_face_model():
    print("=" * 50)
    print("测试 FacesModel (MobileNetV3)...")
    try:
        model = FacesModel(device='cpu')
        model.eval()  # 设置为评估模式
        print("✅ FacesModel 加载成功！")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 160, 160)
        with torch.no_grad():  # 推理时不需要梯度
            output = model(dummy_input)
        print(f"   输出形状: {output.shape} (应该是 [1, 2])")
        print(f"   输出值: {output}")
        return True
    except Exception as e:
        print(f"❌ FacesModel 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_model():
    print("=" * 50)
    print("测试 Wav2Vec2Encoder...")
    try:
        encoder = Wav2Vec2Encoder(device='cpu')
        print("✅ Wav2Vec2Encoder 加载成功！")
        print(f"   模型设备: {encoder.device}")
        return True
    except Exception as e:
        print(f"❌ Wav2Vec2Encoder 加载失败: {e}")
        return False

if __name__ == '__main__':
    print("\n🔍 开始测试预训练模型...\n")
    
    face_ok = test_face_model()
    audio_ok = test_audio_model()
    
    print("\n" + "=" * 50)
    if face_ok and audio_ok:
        print("🎉 所有模型加载成功！")
    else:
        print("⚠️  部分模型加载失败，请检查上面的错误信息")
    print("=" * 50)
