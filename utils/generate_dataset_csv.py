"""
生成数据集标签 CSV 文件
遍历所有视频切片，根据文件名自动分类为 deception 或 truth
"""

from dataloader.video_downloader import VideoDownloader


def generate_dataset():
    """生成数据集标签 CSV 文件"""
    downloader = VideoDownloader()
    downloader.generate_dataset_csv()


if __name__ == '__main__':
    generate_dataset()
