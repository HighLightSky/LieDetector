"""
运行视频下载和切片脚本
"""

from dataloader.video_downloader import VideoDownloader


def download_videos():
    # 创建下载器实例
    downloader = VideoDownloader(
        csv_path='src/DOLOS/dolos_timestamps.csv',
        full_video_dir='src/videos/full',
        cut_video_dir='src/videos/cut'
    )
    
    # 运行完整流程
    downloader.run()
