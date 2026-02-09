"""
YouTube 视频下载和切片工具
从 DOLOS 数据集下载视频并根据时间戳进行切片
"""

import os
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import time


class VideoDownloader:
    """YouTube 视频下载和切片器"""
    
    def __init__(
        self,
        csv_path: str = 'src/DOLOS/dolos_timestamps.csv',
        full_video_dir: str = 'src/videos/full',
        cut_video_dir: str = 'src/videos/cut'
    ):
        self.csv_path = csv_path
        self.full_video_dir = Path(full_video_dir)
        self.cut_video_dir = Path(cut_video_dir)
        
        # 创建目录
        self.full_video_dir.mkdir(parents=True, exist_ok=True)
        self.cut_video_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储视频信息
        self.video_clips: Dict[str, List[Dict]] = defaultdict(list)
    
    def parse_time(self, time_str: str) -> float:
        """将时间字符串转换为秒数
        
        Args:
            time_str: 时间字符串，格式如 "2:06", "1:23:45", "6" (秒)
        
        Returns:
            秒数
        """
        time_str = time_str.strip()
        parts = time_str.split(':')
        
        if len(parts) == 1:  # SS (只有秒)
            return float(parts[0])
        elif len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            raise ValueError(f"无效的时间格式: {time_str}")
    
    def load_csv(self):
        """加载 CSV 文件并解析视频信息"""
        print(f"正在加载 CSV 文件: {self.csv_path}")
        
        # 尝试不同的编码
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(self.csv_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    
                    # 读取第一行检查列名
                    first_row = next(reader, None)
                    if first_row is None:
                        continue
                    
                    # 清理列名（去除空格和 BOM）
                    fieldnames = [name.strip() for name in reader.fieldnames]
                    
                    # 检查必需的列
                    required_cols = ['YT_Video_ID', 'file_name', 'start_time', 'end_time', 'label']
                    if not all(col in fieldnames for col in required_cols):
                        print(f"  尝试编码 {encoding} - 列名不匹配")
                        print(f"  找到的列: {fieldnames}")
                        continue
                    
                    # 处理第一行
                    video_id = first_row['YT_Video_ID'].strip()
                    self.video_clips[video_id].append({
                        'file_name': first_row['file_name'].strip(),
                        'start_time': self.parse_time(first_row['start_time']),
                        'end_time': self.parse_time(first_row['end_time']),
                        'label': first_row['label'].strip()
                    })
                    
                    # 处理剩余行
                    for row in reader:
                        video_id = row['YT_Video_ID'].strip()
                        self.video_clips[video_id].append({
                            'file_name': row['file_name'].strip(),
                            'start_time': self.parse_time(row['start_time']),
                            'end_time': self.parse_time(row['end_time']),
                            'label': row['label'].strip()
                        })
                    
                    print(f"✓ 使用编码: {encoding}")
                    break
                    
            except (UnicodeDecodeError, KeyError) as e:
                print(f"  尝试编码 {encoding} - 失败: {e}")
                continue
        else:
            raise RuntimeError("无法读取 CSV 文件，尝试了所有编码")
        
        print(f"共找到 {len(self.video_clips)} 个不同的视频")
        total_clips = sum(len(clips) for clips in self.video_clips.values())
        print(f"共需要切片 {total_clips} 个视频片段")
    
    def download_video(self, video_id: str) -> Path:
        """下载单个 YouTube 视频
        
        Args:
            video_id: YouTube 视频 ID
        
        Returns:
            下载的视频文件路径
        """
        output_path = self.full_video_dir / f"{video_id}.mp4"
        
        # 如果已经下载，跳过
        if output_path.exists():
            print(f"  视频已存在，跳过下载: {video_id}")
            return output_path
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"  正在下载: {url}")
        print(f"  " + "-" * 58)
        
        try:
            # 使用 yt-dlp 下载视频
            # 使用 android 客户端绕过 YouTube 限制
            # 不使用 capture_output，让输出直接显示到终端
            process = subprocess.Popen(
                [
                    'yt-dlp',
                    '--extractor-args', 'youtube:player_client=android',
                    '--no-check-certificates',
                    '-f', 'best[ext=mp4]/best',  # 简化格式选择
                    '--merge-output-format', 'mp4',
                    '--progress',  # 显示进度
                    '--newline',  # 每个进度更新换行
                    '-o', str(output_path),
                    url
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时输出
            for line in process.stdout:
                # 添加缩进使输出更整齐
                print(f"  {line.rstrip()}")
            
            # 等待进程结束
            return_code = process.wait(timeout=600)
            
            if return_code == 0:
                print(f"  " + "-" * 58)
                print(f"  ✓ 下载成功: {video_id}")
                return output_path
            else:
                print(f"  " + "-" * 58)
                print(f"  ✗ 下载失败: {video_id} (退出码: {return_code})")
                return None
        
        except FileNotFoundError:
            raise RuntimeError(
                "未找到 yt-dlp。请安装: pip install yt-dlp"
            )
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"  " + "-" * 58)
            print(f"  ✗ 下载超时: {video_id}")
            return None
        except Exception as e:
            print(f"  " + "-" * 58)
            print(f"  ✗ 下载出错: {video_id}")
            print(f"    错误: {e}")
            return None
    
    def cut_video(
        self,
        input_path: Path,
        output_name: str,
        start_time: float,
        end_time: float
    ) -> bool:
        """切片视频
        
        Args:
            input_path: 输入视频路径
            output_name: 输出文件名
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
        
        Returns:
            是否成功
        """
        output_path = self.cut_video_dir / f"{output_name}.mp4"
        
        # 如果已经切片，跳过
        if output_path.exists():
            print(f"    ✓ 片段已存在: {output_name}")
            return True
        
        duration = end_time - start_time
        
        # 格式化时间显示
        start_str = f"{int(start_time//60)}:{int(start_time%60):02d}"
        end_str = f"{int(end_time//60)}:{int(end_time%60):02d}"
        
        print(f"    切片中: {start_str} -> {end_str} ({duration:.1f}秒)...", end=' ')
        
        try:
            # 使用 ffmpeg 切片
            result = subprocess.run(
                [
                    'ffmpeg',
                    '-i', str(input_path),
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',  # 直接复制，不重新编码（更快）
                    '-y',  # 覆盖输出文件
                    '-loglevel', 'error',  # 只显示错误
                    str(output_path)
                ],
                check=True,
                capture_output=True,
                timeout=60
            )
            print("✓")
            return True
        
        except FileNotFoundError:
            print("✗")
            raise RuntimeError(
                "未找到 ffmpeg。请安装: https://ffmpeg.org/download.html"
            )
        except subprocess.CalledProcessError as e:
            print("✗")
            print(f"      错误: {e.stderr.decode()}")
            return False
        except subprocess.TimeoutExpired:
            print("✗ (超时)")
            return False
    
    def download_all_videos(self):
        """下载所有完整视频"""
        print("\n" + "="*60)
        print("步骤 1: 下载完整视频")
        print("="*60)
        
        success_count = 0
        fail_count = 0
        
        for i, video_id in enumerate(self.video_clips.keys(), 1):
            print(f"\n[{i}/{len(self.video_clips)}] 处理视频: {video_id}")
            result = self.download_video(video_id)
            
            if result:
                success_count += 1
            else:
                fail_count += 1
            
            # 避免请求过快
            time.sleep(1)
        
        print(f"\n下载完成: 成功 {success_count}, 失败 {fail_count}")
    
    def cut_all_videos(self):
        """切片所有视频"""
        print("\n" + "="*60)
        print("步骤 2: 切片视频")
        print("="*60)
        
        success_count = 0
        fail_count = 0
        total_clips = sum(len(clips) for clips in self.video_clips.values())
        
        current = 0
        for video_id, clips in self.video_clips.items():
            full_video_path = self.full_video_dir / f"{video_id}.mp4"
            
            if not full_video_path.exists():
                print(f"\n跳过 {video_id} (完整视频不存在)")
                fail_count += len(clips)
                continue
            
            print(f"\n处理视频: {video_id} ({len(clips)} 个片段)")
            
            for clip in clips:
                current += 1
                print(f"  [{current}/{total_clips}] {clip['file_name']}")
                
                result = self.cut_video(
                    full_video_path,
                    clip['file_name'],
                    clip['start_time'],
                    clip['end_time']
                )
                
                if result:
                    success_count += 1
                else:
                    fail_count += 1
        
        print(f"\n切片完成: 成功 {success_count}, 失败 {fail_count}")
    
    def generate_dataset_csv(self):
        """生成数据集 CSV 文件
        
        遍历所有切片视频，根据文件名中的关键词分类
        - 包含 'lie' 或 'deception': deception (说谎)
        - 包含 'truth' 或 'true': truth (说真话)
        """
        print("\n" + "="*60)
        print("步骤 3: 生成数据集 CSV")
        print("="*60)
        
        # 创建输出目录
        dataset_dir = Path('src/dataset')
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有视频文件
        video_files = list(self.cut_video_dir.glob('*.mp4'))
        
        if not video_files:
            print("⚠️  未找到任何视频文件")
            return
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        # 分类统计
        deception_count = 0
        truth_count = 0
        unknown_count = 0
        
        # 准备 CSV 数据
        csv_data = []
        
        for video_file in sorted(video_files):
            filename = video_file.stem  # 不含后缀的文件名
            filename_lower = filename.lower()
            
            # 根据文件名判断类型
            # 优先检查 'lie' 和 'deception'（说谎）
            if 'lie' in filename_lower or 'deception' in filename_lower:
                label = 'deception'
                deception_count += 1
            # 然后检查 'truth' 和 'true'（说真话）
            elif 'truth' in filename_lower or 'true' in filename_lower:
                label = 'truth'
                truth_count += 1
            else:
                label = 'unknown'
                unknown_count += 1
                print(f"  ⚠️  无法识别类型: {filename}")
            
            csv_data.append({
                'video_name': filename,
                'label': label
            })
        
        # 保存 CSV 文件
        csv_path = dataset_dir / 'video_labels.csv'
        
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['video_name', 'label'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\n✓ CSV 文件已生成: {csv_path}")
        print(f"\n分类统计:")
        print(f"  - 说谎 (deception): {deception_count} 个")
        print(f"  - 说真话 (truth): {truth_count} 个")
        if unknown_count > 0:
            print(f"  - 未知类型: {unknown_count} 个")
        print(f"  - 总计: {len(csv_data)} 个")
        
        return csv_path
    
    def generate_report(self):
        """生成下载和切片报告"""
        print("\n" + "="*60)
        print("步骤 4: 生成报告")
        print("="*60)
        
        report_path = Path('dataloader/video_download_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# DOLOS 视频下载和切片报告\n\n")
            
            # 统计信息
            f.write("## 统计信息\n\n")
            f.write(f"- **CSV 文件**: `{self.csv_path}`\n")
            f.write(f"- **完整视频目录**: `{self.full_video_dir}`\n")
            f.write(f"- **切片视频目录**: `{self.cut_video_dir}`\n")
            f.write(f"- **视频总数**: {len(self.video_clips)}\n")
            
            total_clips = sum(len(clips) for clips in self.video_clips.values())
            f.write(f"- **切片总数**: {total_clips}\n\n")
            
            # 完整视频列表
            f.write("## 完整视频列表\n\n")
            f.write("| 序号 | 视频 ID | 切片数量 | 状态 |\n")
            f.write("|------|---------|----------|------|\n")
            
            for i, (video_id, clips) in enumerate(self.video_clips.items(), 1):
                full_video_path = self.full_video_dir / f"{video_id}.mp4"
                status = "✓ 已下载" if full_video_path.exists() else "✗ 未下载"
                f.write(f"| {i} | {video_id} | {len(clips)} | {status} |\n")
            
            # 切片详情
            f.write("\n## 切片详情\n\n")
            
            for video_id, clips in sorted(self.video_clips.items()):
                f.write(f"### {video_id}\n\n")
                f.write("| 文件名 | 开始时间 | 结束时间 | 标签 | 状态 |\n")
                f.write("|--------|----------|----------|------|------|\n")
                
                for clip in clips:
                    clip_path = self.cut_video_dir / f"{clip['file_name']}.mp4"
                    status = "✓" if clip_path.exists() else "✗"
                    
                    start_str = f"{int(clip['start_time']//60)}:{int(clip['start_time']%60):02d}"
                    end_str = f"{int(clip['end_time']//60)}:{int(clip['end_time']%60):02d}"
                    
                    f.write(
                        f"| {clip['file_name']} | {start_str} | {end_str} | "
                        f"{clip['label']} | {status} |\n"
                    )
                
                f.write("\n")
            
            # 使用说明
            f.write("## 使用说明\n\n")
            f.write("### 依赖安装\n\n")
            f.write("```bash\n")
            f.write("# 安装 yt-dlp (YouTube 下载工具)\n")
            f.write("pip install yt-dlp\n\n")
            f.write("# 安装 ffmpeg (视频处理工具)\n")
            f.write("# Windows: 从 https://ffmpeg.org/download.html 下载\n")
            f.write("# Linux: sudo apt install ffmpeg\n")
            f.write("# macOS: brew install ffmpeg\n")
            f.write("```\n\n")
            
            f.write("### 运行脚本\n\n")
            f.write("```python\n")
            f.write("from dataloader.video_downloader import VideoDownloader\n\n")
            f.write("downloader = VideoDownloader()\n")
            f.write("downloader.run()\n")
            f.write("```\n\n")
            
            f.write("### 目录结构\n\n")
            f.write("```\n")
            f.write("src/\n")
            f.write("├── videos/\n")
            f.write("│   ├── full/          # 完整视频\n")
            f.write("│   │   ├── UCPIBQDI0QM.mp4\n")
            f.write("│   │   ├── hhSNKJ_lcNU.mp4\n")
            f.write("│   │   └── ...\n")
            f.write("│   └── cut/           # 切片视频\n")
            f.write("│       ├── AN_WILTY_EP15_truth1.mp4\n")
            f.write("│       ├── AN_WILTY_EP15_truth2.mp4\n")
            f.write("│       └── ...\n")
            f.write("└── DOLOS/\n")
            f.write("    └── dolos_timestamps.csv\n")
            f.write("```\n")
        
        print(f"✓ 报告已生成: {report_path}")
    
    def run(self):
        """运行完整流程"""
        print("="*60)
        print("DOLOS 视频下载和切片工具")
        print("="*60)
        
        # 加载 CSV
        self.load_csv()
        
        # 下载完整视频
        self.download_all_videos()
        
        # 切片视频
        self.cut_all_videos()
        
        # 生成数据集 CSV
        self.generate_dataset_csv()
        
        # 生成报告
        self.generate_report()
        
        print("\n" + "="*60)
        print("全部完成！")
        print("="*60)


# 示例用法
if __name__ == '__main__':
    downloader = VideoDownloader()
    downloader.run()
