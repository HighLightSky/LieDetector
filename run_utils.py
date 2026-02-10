"""
运行数据预处理工具
"""

from utils import download_videos, generate_dataset_csv, process_data, experiment_tools

if __name__ == '__main__':
    # 下载和切片YouTube原始视频
    # download_videos.download_videos()

    # 根据视频切片情况更新切片统计数据集，并给视频打上格式化标签
    # generate_dataset_csv.generate_dataset()

    # 提取所有cut视频的三大特征，并保存
    # process_data.preprocess_features()

    # 根据当前训练日志绘制训练曲线
    visualizer = experiment_tools.ExperimentVisualizer(r"checkpoints\history\train_history_20260210_183013.json")
    visualizer.plot_combined_overview()