"""
谎言检测推理脚本

使用方法:
1. 默认测试（50个视频，计算准确率）:
   python detect.py

2. 单个视频检测:
   python detect.py --video path/to/video.mp4

3. 批量检测:
   python detect.py --video_dir path/to/videos/

4. 指定检查点和设备:
   python detect.py --checkpoint checkpoints/best_model.pth --device cuda
"""

import argparse
import sys
from pathlib import Path
import random

import torch
import pandas as pd

from detector import LieDetector


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='谎言检测推理')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='模型检查点路径 (默认: checkpoints/best_model.pth)')
    
    # 输入参数
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--video', type=str,
                       help='单个视频文件路径')
    group.add_argument('--video_dir', type=str,
                       help='视频目录路径（批量检测）')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu, 默认: cuda)')
    
    return parser.parse_args()


def load_labels(csv_path='src/dataset/video_labels.csv'):
    """加载视频标签"""
    try:
        df = pd.read_csv(csv_path)
        label_map = {'truth': 0, 'deception': 1}
        labels = {}
        for _, row in df.iterrows():
            video_name = row['video_name']
            labels[video_name] = label_map[row['label']]
        return labels
    except Exception as e:
        print(f"[WARNING] 无法加载标签文件: {e}")
        return None


def calculate_accuracy(results, labels):
    """计算准确率"""
    if labels is None:
        return None
    
    correct = 0
    total = 0
    
    for result in results:
        video_name = Path(result['details']['video_path']).stem
        if video_name in labels:
            true_label = labels[video_name]
            pred_label = result['prediction']
            if pred_label != -1:  # 排除错误样本
                total += 1
                if pred_label == true_label:
                    correct += 1
    
    if total == 0:
        return None
    
    return correct / total


def default_test(detector):
    """默认测试：随机抽取50个视频检测并计算准确率"""
    print("\n" + "="*60)
    print("默认测试模式")
    print("="*60)
    
    # 查找视频
    video_dir = Path('src/videos/cut')
    if not video_dir.exists():
        print(f"[ERROR] 视频目录不存在: {video_dir}")
        print("请先准备数据或使用 --video 或 --video_dir 参数")
        sys.exit(1)
    
    video_files = list(video_dir.glob('*.mp4'))
    if len(video_files) == 0:
        print(f"[ERROR] 在目录中没有找到视频文件")
        sys.exit(1)
    
    # 随机抽取50个视频
    num_samples = min(50, len(video_files))
    sampled_videos = random.sample(video_files, num_samples)
    
    print(f"从 {len(video_files)} 个视频中随机抽取 {num_samples} 个进行测试")
    
    # 加载标签
    labels = load_labels()
    
    # 批量检测
    print("\n开始检测...")
    results = []
    
    for i, video_path in enumerate(sampled_videos, 1):
        print(f"\n[{i}/{num_samples}] {video_path.name}")
        try:
            result = detector.predict(video_path, verbose=False)
            results.append(result)
            print(f"  预测: {result['label_cn']} (置信度: {result['confidence']:.2%})")
        except Exception as e:
            print(f"  检测失败: {e}")
            results.append({
                'prediction': -1,
                'label': 'error',
                'label_cn': '错误',
                'confidence': 0.0,
                'details': {'video_path': str(video_path)}
            })
    
    # 统计结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    truth_count = sum(1 for r in results if r['prediction'] == 0)
    deception_count = sum(1 for r in results if r['prediction'] == 1)
    error_count = sum(1 for r in results if r['prediction'] == -1)
    
    print(f"\n总计: {len(results)} 个视频")
    print(f"  说真话: {truth_count} 个 ({100*truth_count/len(results):.1f}%)")
    print(f"  说谎: {deception_count} 个 ({100*deception_count/len(results):.1f}%)")
    if error_count > 0:
        print(f"  错误: {error_count} 个 ({100*error_count/len(results):.1f}%)")
    
    # 计算准确率
    if labels is not None:
        accuracy = calculate_accuracy(results, labels)
        if accuracy is not None:
            print(f"\n准确率: {accuracy:.2%}")
            
            # 详细统计
            tp = sum(1 for r in results if r['prediction'] == 1 and labels.get(Path(r['details']['video_path']).stem) == 1)
            tn = sum(1 for r in results if r['prediction'] == 0 and labels.get(Path(r['details']['video_path']).stem) == 0)
            fp = sum(1 for r in results if r['prediction'] == 1 and labels.get(Path(r['details']['video_path']).stem) == 0)
            fn = sum(1 for r in results if r['prediction'] == 0 and labels.get(Path(r['details']['video_path']).stem) == 1)
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
                print(f"精确率: {precision:.2%}")
            if tp + fn > 0:
                recall = tp / (tp + fn)
                print(f"召回率: {recall:.2%}")
            if tp + fp > 0 and tp + fn > 0:
                f1 = 2 * precision * recall / (precision + recall)
                print(f"F1分数: {f1:.2%}")
    
    print("="*60)


def main():
    """主函数"""
    args = parse_args()
    
    # 设备检查
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 检查点检查
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] 检查点文件不存在: {checkpoint_path}")
        sys.exit(1)
    
    # 加载检测器
    print("\n" + "="*60)
    print("加载谎言检测器")
    print("="*60)
    print(f"检查点: {checkpoint_path}")
    print(f"设备: {args.device}")
    
    try:
        detector = LieDetector.from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device=args.device,
            freeze_backbones=True
        )
    except Exception as e:
        print(f"[ERROR] 加载检测器失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("检测器加载成功!")
    
    # 单个视频检测
    if args.video:
        video_path = Path(args.video)
        
        if not video_path.exists():
            print(f"\n[ERROR] 视频文件不存在: {video_path}")
            sys.exit(1)
        
        try:
            result = detector.predict(video_path, verbose=True)
            sys.exit(0)
            
        except Exception as e:
            print(f"\n[ERROR] 检测失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 批量检测
    elif args.video_dir:
        video_dir = Path(args.video_dir)
        
        if not video_dir.exists():
            print(f"\n[ERROR] 视频目录不存在: {video_dir}")
            sys.exit(1)
        
        # 查找所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_paths = []
        for ext in video_extensions:
            video_paths.extend(video_dir.glob(f'*{ext}'))
            video_paths.extend(video_dir.glob(f'*{ext.upper()}'))
        
        if len(video_paths) == 0:
            print(f"\n[ERROR] 在目录中没有找到视频文件: {video_dir}")
            sys.exit(1)
        
        print(f"\n找到 {len(video_paths)} 个视频文件")
        
        # 加载标签
        labels = load_labels()
        
        try:
            results = detector.predict_batch(video_paths, verbose=False)
            
            # 统计结果
            print("\n" + "="*60)
            print("批量检测结果汇总")
            print("="*60)
            
            truth_count = sum(1 for r in results if r['prediction'] == 0)
            deception_count = sum(1 for r in results if r['prediction'] == 1)
            error_count = sum(1 for r in results if r['prediction'] == -1)
            
            print(f"\n总计: {len(results)} 个视频")
            print(f"  说真话: {truth_count} 个")
            print(f"  说谎: {deception_count} 个")
            if error_count > 0:
                print(f"  错误: {error_count} 个")
            
            # 计算准确率
            if labels is not None:
                accuracy = calculate_accuracy(results, labels)
                if accuracy is not None:
                    print(f"\n准确率: {accuracy:.2%}")
            
            print("="*60)
            sys.exit(0)
            
        except Exception as e:
            print(f"\n[ERROR] 批量检测失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 默认测试模式
    else:
        try:
            default_test(detector)
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
