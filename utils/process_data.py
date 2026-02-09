"""
数据处理脚本 - 批量提取视频特征
从 src/videos/cut/ 批量处理视频，提取人脸、音频、OpenFace 特征
"""

import torch
from pathlib import Path
import time
import signal
from contextlib import contextmanager
from dataloader import FaceExtractor, AudioExtractor, OpenFaceExtractor


class TimeoutException(Exception):
    """超时异常"""
    pass


@contextmanager
def timeout(seconds):
    """超时上下文管理器
    
    Args:
        seconds: 超时秒数
    """
    def timeout_handler(signum, frame):
        raise TimeoutException(f"处理超时 ({seconds}秒)")
    
    # Windows 不支持 signal.SIGALRM，需要使用其他方法
    import platform
    if platform.system() == 'Windows':
        # Windows 使用线程计时器
        import threading
        timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutException(f"处理超时 ({seconds}秒)")))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    else:
        # Unix/Linux 使用 signal
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def process_videos_in_batches(video_dir: Path, output_dir: Path, batch_size: int = 50):
    """批量处理视频特征提取
    
    Args:
        video_dir: 视频目录 (src/videos/cut/)
        output_dir: 输出目录 (src/features/)
        batch_size: 每批处理的视频数量
    """
    
    # 获取所有视频文件
    video_files = sorted(video_dir.glob('*.mp4'))
    total_videos = len(video_files)
    
    if total_videos == 0:
        print("未找到视频文件")
        return
    
    print(f"找到 {total_videos} 个视频文件")
    print(f"输出目录: {output_dir}")
    print(f"批次大小: {batch_size}\n")
    
    # 初始化提取器
    print("初始化提取器...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    face_extractor = FaceExtractor(
        method='opencv',
        target_size=(160, 160),
        num_frames=16,
        device='cpu'
    )
    
    audio_extractor = AudioExtractor(
        method='wav2vec2',
        device=device
    )
    
    try:
        openface_extractor = OpenFaceExtractor(
            openface_path=None,
            num_frames=16,
            feature_dim=714
        )
        use_openface = True
        print("✓ 人脸提取器 | ✓ 音频提取器 | ✓ OpenFace 提取器\n")
    except Exception:
        openface_extractor = None
        use_openface = False
        print("✓ 人脸提取器 | ✓ 音频提取器 | ✗ OpenFace 不可用\n")
    
    # 统计信息
    stats = {
        'processed': 0,
        'skipped': 0,
        'faces_ok': 0,
        'audios_ok': 0,
        'openfaces_ok': 0,
        'failed': 0
    }
    
    start_time = time.time()
    
    # 分批处理
    num_batches = (total_videos + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_videos)
        batch_videos = video_files[batch_start:batch_end]
        
        print(f"=== 批次 {batch_idx + 1}/{num_batches} ({batch_start + 1}-{batch_end}/{total_videos}) ===\n")
        
        for i, video_path in enumerate(batch_videos, 1):
            video_name = video_path.stem
            global_idx = batch_start + i
            
            # 检查是否已存在
            face_path = output_dir / f'{video_name}_faces.pt'
            audio_path = output_dir / f'{video_name}_audios.pt'
            openface_path = output_dir / f'{video_name}_openfaces.pt'
            
            all_exist = face_path.exists() and audio_path.exists()
            if use_openface:
                all_exist = all_exist and openface_path.exists()
            
            if all_exist:
                stats['skipped'] += 1
                print(f"[{global_idx}/{total_videos}] {video_name} - 已跳过")
                continue
            
            # 处理视频（带超时控制）
            video_start = time.time()
            print(f"[{global_idx}/{total_videos}] {video_name}", end=' ', flush=True)
            
            # 标记是否有任何特征提取成功
            any_success = False
            video_error = False
            timeout_error = False
            
            try:
                # 提取人脸特征（带超时）
                if not face_path.exists():
                    try:
                        # 检查是否已经超时
                        elapsed = time.time() - video_start
                        if elapsed > 40:
                            timeout_error = True
                            raise TimeoutException("总处理时间超过40秒")
                        
                        faces = face_extractor.extract_from_video(video_path, return_tensor=True)
                        torch.save(faces, face_path)
                        stats['faces_ok'] += 1
                        any_success = True
                        print("F", end='', flush=True)
                    except TimeoutException as e:
                        print(f"\n  ⏱ 人脸提取超时")
                        timeout_error = True
                    except Exception as e:
                        error_msg = str(e)
                        if "无法打开视频" in error_msg or "视频读取失败" in error_msg or "未检测到任何人脸" in error_msg:
                            print(f"\n  ✗ 人脸提取失败: {error_msg[:80]}")
                            video_error = True
                        else:
                            print(f"\n  ✗ 人脸提取异常: {error_msg[:80]}")
                else:
                    stats['faces_ok'] += 1
                    any_success = True
                    print("f", end='', flush=True)
                
                # 如果视频无效或超时，跳过后续处理
                if video_error or timeout_error:
                    stats['failed'] += 1
                    video_time = time.time() - video_start
                    if timeout_error:
                        print(f" ⏱ 超时跳过 ({video_time:.1f}s)")
                    else:
                        print(f" ✗ 视频无效，已跳过 ({video_time:.1f}s)")
                    continue
                
                # 提取音频特征（带超时检查）
                if not audio_path.exists():
                    try:
                        # 检查是否已经超时
                        elapsed = time.time() - video_start
                        if elapsed > 40:
                            timeout_error = True
                            raise TimeoutException("总处理时间超过40秒")
                        
                        audios = audio_extractor.extract_from_video(
                            video_path, 
                            return_tensor=True, 
                            keep_audio=False
                        )
                        torch.save(audios, audio_path)
                        stats['audios_ok'] += 1
                        any_success = True
                        print("A", end='', flush=True)
                    except TimeoutException as e:
                        print(f"\n  ⏱ 音频提取超时")
                        timeout_error = True
                    except Exception as e:
                        error_msg = str(e)
                        print(f"\n  ✗ 音频提取失败: {error_msg[:80]}")
                else:
                    stats['audios_ok'] += 1
                    any_success = True
                    print("a", end='', flush=True)
                
                # 如果超时，跳过 OpenFace
                if timeout_error:
                    stats['failed'] += 1
                    video_time = time.time() - video_start
                    print(f" ⏱ 超时跳过 ({video_time:.1f}s)")
                    continue
                
                # 提取 OpenFace 特征（带超时检查）
                if use_openface and not openface_path.exists():
                    try:
                        # 检查是否已经超时
                        elapsed = time.time() - video_start
                        if elapsed > 40:
                            timeout_error = True
                            raise TimeoutException("总处理时间超过40秒")
                        
                        openfaces = openface_extractor.extract_from_video(
                            video_path,
                            return_tensor=True,
                            save_csv=False,
                            verbose=False
                        )
                        torch.save(openfaces, openface_path)
                        stats['openfaces_ok'] += 1
                        any_success = True
                        print("O", end='', flush=True)
                    except TimeoutException as e:
                        print(f"\n  ⏱ OpenFace 提取超时")
                        timeout_error = True
                    except Exception as e:
                        error_msg = str(e)
                        print(f"\n  ✗ OpenFace 提取失败: {error_msg[:80]}")
                elif use_openface:
                    stats['openfaces_ok'] += 1
                    any_success = True
                    print("o", end='', flush=True)
                
                video_time = time.time() - video_start
                
                if timeout_error:
                    stats['failed'] += 1
                    print(f" ⏱ 超时跳过 ({video_time:.1f}s)")
                elif any_success:
                    stats['processed'] += 1
                    print(f" ✓ ({video_time:.1f}s)")
                else:
                    stats['failed'] += 1
                    print(f" ✗ ({video_time:.1f}s)")
                    
            except KeyboardInterrupt:
                print("\n\n用户中断处理")
                raise
            except Exception as e:
                # 捕获所有未预期的异常，防止程序卡住
                video_time = time.time() - video_start
                stats['failed'] += 1
                print(f"\n  ✗ 处理异常: {str(e)[:80]} ({video_time:.1f}s)")
                print("  继续处理下一个视频...")
        
        # 批次统计
        elapsed = time.time() - start_time
        avg_time = elapsed / (stats['processed'] + stats['failed']) if (stats['processed'] + stats['failed']) > 0 else 0
        remaining_videos = total_videos - batch_end
        remaining_time = remaining_videos * avg_time / 60 if avg_time > 0 else 0
        
        print(f"\n已处理: {stats['processed']} | 已跳过: {stats['skipped']} | 失败: {stats['failed']}")
        print(f"平均: {avg_time:.2f}s/视频 | 预计剩余: {remaining_time:.1f}分钟\n")
    
    # 最终统计
    print("=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"总视频数: {total_videos}")
    print(f"已处理: {stats['processed']}")
    print(f"已跳过: {stats['skipped']}")
    print(f"失败: {stats['failed']}")
    print(f"\n人脸特征: {stats['faces_ok']}")
    print(f"音频特征: {stats['audios_ok']}")
    if use_openface:
        print(f"OpenFace 特征: {stats['openfaces_ok']}")
    print(f"\n输出目录: {output_dir.absolute()}")


def preprocess_features():
    """主函数，提取所有cut视频的三大特征，并保存"""
    video_dir = Path('src/videos/cut')
    output_dir = Path('src/features')
    
    if not video_dir.exists():
        print(f"视频目录不存在: {video_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n批量特征提取\n")
    process_videos_in_batches(video_dir, output_dir, batch_size=50)


if __name__ == '__main__':
    preprocess_features()

