"""
运行入口
"""

from utils import download_videos, generate_dataset_csv, process_data

if __name__ == '__main__':
    # 下载和切片YouTube原始视频
    # download_videos.download_videos()

    # 根据视频切片情况更新切片统计数据集，并给视频打上格式化标签
    # generate_dataset_csv.generate_dataset()

    # 提取所有cut视频的三大特征，并保存
    process_data.preprocess_features()