"""
预提取特征工具
批量处理视频，提取并保存特征到磁盘
用于加速训练过程
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Optional, List, Dict

from .face_extractor import FaceExtractor
from .openface_extractor import OpenFaceExtractor
from .audio_extractor import AudioExtractor


def precompute_all_features(
    video_list: List[Dict],
    output_dir: str,
    extract_faces: bool = True,
    extract_openface: bool = True,
    extract_audio: bool = True,
    num_frames: int = 16,
    device: str = 'cpu'
) -> int:
    """批量预提取特征（简化版）
    
    Args:
        video_list: 视频列表，每个元素包含 video_path, label, video_id
        output_dir: 输出目录
        extract_faces: 是否提取人脸特征
        extract_openface: 是否提取 OpenFace 特征
        extract_audio: 是否提取音频特征
        num_frames: 提取的帧数
        device: 设备
    
    Returns:
        成功处理的视频数量
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    print(f"总共 {len(video_list)} 个视频")
    
    # 初始化提取器
    extractors = {}
    
    if extract_faces:
        print("初始化 FaceExtractor...")
        extractors['face'] = FaceExtractor(
            method='opencv',
            num_frames=num_frames,
            device=device
        )
    
    if extract_openface:
        try:
            print("初始化 OpenFaceExtractor...")
            extractors['openface'] = OpenFaceExtractor(
                openface_path=None,
                num_frames=num_frames
            )
        except Exception as e:
            print(f"⚠️  OpenFace 初始化失败，跳过: {e}")
            extract_openface = False
    
    if extract_audio:
        print("初始化 AudioExtractor...")
        extractors['audio'] = AudioExtractor(
            method='wav2vec2',
            device=device
        )
    
    print("\n开始提取特征...")
    
    # 统计
    success_count = 0
    fail_count = 0
    
    # 遍历视频
    for item in tqdm(video_list, desc="提取特征"):
        video_path = Path(item['video_path'])
        video_id = item['video_id']
        
        if not video_path.exists():
            print(f"\n⚠️  视频不存在: {video_path}")
            fail_count += 1
            continue
        
        try:
            # 提取面部图像
            if extract_faces:
                try:
                    faces = extractors['face'](video_path, return_tensor=True)
                    torch.save(faces, output_dir / f'{video_id}_faces.pt')
                except Exception as e:
                    print(f"\n⚠️  人脸提取失败 {video_id}: {e}")
            
            # 提取 OpenFace 特征
            if extract_openface:
                try:
                    openfaces = extractors['openface'](video_path, return_tensor=True)
                    torch.save(openfaces, output_dir / f'{video_id}_openface.pt')
                except Exception as e:
                    print(f"\n⚠️  OpenFace 提取失败 {video_id}: {e}")
            
            # 提取音频特征
            if extract_audio:
                try:
                    audios = extractors['audio'](video_path, from_video=True, return_tensor=True)
                    torch.save(audios, output_dir / f'{video_id}_audio.pt')
                except Exception as e:
                    print(f"\n⚠️  音频提取失败 {video_id}: {e}")
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ 提取失败 {video_id}: {e}")
            fail_count += 1
            continue
    
    # 打印统计
    print("\n" + "=" * 50)
    print("特征提取完成!")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📊 总计: {len(video_list)}")
    print("=" * 50)
    
    return success_count


def precompute_features(
    annotation_file: str,
    video_dir: str,
    output_dir: str,
    face_method: str = 'opencv',
    openface_path: Optional[str] = None,
    audio_method: str = 'wav2vec2',
    num_frames: int = 16,
    device: str = 'cuda',
    skip_existing: bool = True
):
    """批量预提取特征
    
    Args:
        annotation_file: 标注文件路径
        video_dir: 视频目录
        output_dir: 输出目录
        face_method: 人脸检测方法
        openface_path: OpenFace 可执行文件路径
        audio_method: 音频特征提取方法
        num_frames: 提取的帧数
        device: 设备
        skip_existing: 是否跳过已存在的特征
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取标注
    df = pd.read_csv(annotation_file)
    print(f"总共 {len(df)} 个视频")
    
    # 初始化提取器
    print("初始化提取器...")
    face_extractor = FaceExtractor(
        method=face_method,
        num_frames=num_frames,
        device=device
    )
    
    openface_extractor = OpenFaceExtractor(
        openface_path=openface_path,
        num_frames=num_frames
    )
    
    audio_extractor = AudioExtractor(
        method=audio_method,
        device=device
    )
    
    print("开始提取特征...")
    
    # 统计
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    # 遍历视频
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
        video_id = row['video_id']
        
        # 创建输出目录
        feature_dir = output_dir / video_id
        feature_dir.mkdir(exist_ok=True)
        
        # 检查是否已存在
        if skip_existing:
            faces_path = feature_dir / 'faces.pt'
            openfaces_path = feature_dir / 'openfaces.pt'
            audios_path = feature_dir / 'audios.pt'
            
            if faces_path.exists() and openfaces_path.exists() and audios_path.exists():
                skip_count += 1
                continue
        
        # 查找视频文件
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            candidate = Path(video_dir) / f"{video_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        if video_path is None:
            print(f"\n警告: 未找到视频 {video_id}")
            fail_count += 1
            continue
        
        try:
            # 提取面部图像
            faces = face_extractor(video_path)
            torch.save(faces, feature_dir / 'faces.pt')
            
            # 提取 OpenFace 特征
            openfaces = openface_extractor(video_path)
            torch.save(openfaces, feature_dir / 'openfaces.pt')
            
            # 提取音频特征
            audios = audio_extractor(video_path, from_video=True)
            torch.save(audios, feature_dir / 'audios.pt')
            
            success_count += 1
            
        except Exception as e:
            print(f"\n错误: 提取失败 {video_id}: {e}")
            fail_count += 1
            continue
    
    # 打印统计
    print("\n" + "=" * 50)
    print("特征提取完成!")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(df)}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='预提取多模态特征')
    
    parser.add_argument(
        '--annotation_file',
        type=str,
        required=True,
        help='标注文件路径 (CSV)'
    )
    
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='视频目录'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )
    
    parser.add_argument(
        '--face_method',
        type=str,
        default='opencv',
        choices=['opencv', 'dlib', 'mtcnn'],
        help='人脸检测方法'
    )
    
    parser.add_argument(
        '--openface_path',
        type=str,
        default=None,
        help='OpenFace 可执行文件路径'
    )
    
    parser.add_argument(
        '--audio_method',
        type=str,
        default='wav2vec2',
        choices=['wav2vec2', 'mfcc'],
        help='音频特征提取方法'
    )
    
    parser.add_argument(
        '--num_frames',
        type=int,
        default=16,
        help='提取的帧数'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='设备'
    )
    
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='跳过已存在的特征'
    )
    
    args = parser.parse_args()
    
    precompute_features(
        annotation_file=args.annotation_file,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        face_method=args.face_method,
        openface_path=args.openface_path,
        audio_method=args.audio_method,
        num_frames=args.num_frames,
        device=args.device,
        skip_existing=args.skip_existing
    )


if __name__ == '__main__':
    main()
