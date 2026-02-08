"""
面部图像提取器
从视频中检测、裁剪并提取面部图像序列
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Union
import warnings


class FaceExtractor:
    """从视频中提取面部图像序列
    
    支持多种人脸检测方法：
    - opencv: OpenCV Haar Cascade (快速，但精度较低)
    - dlib: dlib 人脸检测器 (中等速度和精度)
    - mtcnn: MTCNN (慢，但精度高)
    
    Args:
        method: 人脸检测方法 ('opencv', 'dlib', 'mtcnn')
        target_size: 输出图像大小 (height, width)
        num_frames: 要提取的帧数，None 表示提取所有帧
        device: 设备 ('cpu' 或 'cuda')
    """
    
    def __init__(
        self,
        method: str = 'opencv',
        target_size: Tuple[int, int] = (160, 160),
        num_frames: Optional[int] = 16,
        device: str = 'cpu'
    ):
        self.method = method
        self.target_size = target_size
        self.num_frames = num_frames
        self.device = device
        
        # 初始化人脸检测器
        self._init_detector()
    
    def _init_detector(self):
        """初始化人脸检测器"""
        if self.method == 'opencv':
            # OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise RuntimeError("无法加载 OpenCV 人脸检测器")
        
        elif self.method == 'dlib':
            try:
                import dlib
                self.detector = dlib.get_frontal_face_detector()
            except ImportError:
                raise ImportError("请安装 dlib: pip install dlib")
        
        elif self.method == 'mtcnn':
            try:
                from facenet_pytorch import MTCNN
                self.detector = MTCNN(
                    keep_all=False,
                    device=self.device,
                    post_process=False
                )
            except ImportError:
                raise ImportError("请安装 facenet-pytorch: pip install facenet-pytorch")
        
        else:
            raise ValueError(f"不支持的检测方法: {self.method}")
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """检测单帧中的人脸
        
        Args:
            frame: BGR 格式的图像 (H, W, 3)
        
        Returns:
            人脸边界框 (x, y, w, h) 或 None
        """
        if self.method == 'opencv':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            if len(faces) > 0:
                # 返回最大的人脸
                areas = [w * h for (x, y, w, h) in faces]
                max_idx = np.argmax(areas)
                return tuple(faces[max_idx])
            return None
        
        elif self.method == 'dlib':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)
            if len(faces) > 0:
                # 返回第一个人脸
                face = faces[0]
                x, y = face.left(), face.top()
                w, h = face.width(), face.height()
                return (x, y, w, h)
            return None
        
        elif self.method == 'mtcnn':
            # MTCNN 需要 RGB 格式
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.detector.detect(rgb)
            if boxes is not None and len(boxes) > 0:
                # 返回第一个人脸
                x1, y1, x2, y2 = boxes[0].astype(int)
                return (x1, y1, x2 - x1, y2 - y1)
            return None
        
        return None
    
    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.2
    ) -> np.ndarray:
        """裁剪人脸区域
        
        Args:
            frame: 原始图像
            bbox: 人脸边界框 (x, y, w, h)
            margin: 边距比例，扩大裁剪区域
        
        Returns:
            裁剪后的人脸图像
        """
        x, y, w, h = bbox
        
        # 添加边距
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(frame.shape[1], x + w + margin_w)
        y2 = min(frame.shape[0], y + h + margin_h)
        
        # 裁剪
        face = frame[y1:y2, x1:x2]
        
        # 调整大小
        face_resized = cv2.resize(face, self.target_size)
        
        return face_resized
    
    def extract_from_video(
        self,
        video_path: Union[str, Path],
        return_tensor: bool = True,
        normalize: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """从视频中提取面部图像序列
        
        Args:
            video_path: 视频文件路径
            return_tensor: 是否返回 PyTorch tensor
            normalize: 是否归一化到 [0, 1]
        
        Returns:
            面部图像序列
            - 如果 return_tensor=True: torch.Tensor (T, 3, H, W)
            - 如果 return_tensor=False: np.ndarray (T, H, W, 3)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 确定要提取的帧索引
        if self.num_frames is None:
            frame_indices = list(range(total_frames))
        else:
            # 均匀采样
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        # 提取人脸
        face_crops = []
        last_valid_face = None  # 用于填充检测失败的帧
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                warnings.warn(f"无法读取帧 {idx}")
                if last_valid_face is not None:
                    face_crops.append(last_valid_face.copy())
                continue
            
            # 检测人脸
            bbox = self.detect_face(frame)
            
            if bbox is not None:
                # 裁剪人脸
                face = self.crop_face(frame, bbox)
                face_crops.append(face)
                last_valid_face = face
            else:
                # 检测失败，使用上一个有效人脸
                if last_valid_face is not None:
                    face_crops.append(last_valid_face.copy())
                else:
                    warnings.warn(f"帧 {idx} 未检测到人脸，且无历史人脸可用")
        
        cap.release()
        
        if len(face_crops) == 0:
            raise RuntimeError(f"视频中未检测到任何人脸: {video_path}")
        
        # 转换为数组
        faces = np.stack(face_crops)  # (T, H, W, 3)
        
        if return_tensor:
            # 转换为 tensor: (T, H, W, 3) -> (T, 3, H, W)
            faces = torch.from_numpy(faces).float()
            faces = faces.permute(0, 3, 1, 2)
            
            if normalize:
                faces = faces / 255.0
        else:
            if normalize:
                faces = faces.astype(np.float32) / 255.0
        
        return faces
    
    def extract_from_images(
        self,
        image_paths: List[Union[str, Path]],
        return_tensor: bool = True,
        normalize: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """从图像列表中提取面部
        
        Args:
            image_paths: 图像文件路径列表
            return_tensor: 是否返回 PyTorch tensor
            normalize: 是否归一化到 [0, 1]
        
        Returns:
            面部图像序列
        """
        face_crops = []
        
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                warnings.warn(f"图像文件不存在: {img_path}")
                continue
            
            # 读取图像
            frame = cv2.imread(str(img_path))
            if frame is None:
                warnings.warn(f"无法读取图像: {img_path}")
                continue
            
            # 检测人脸
            bbox = self.detect_face(frame)
            
            if bbox is not None:
                face = self.crop_face(frame, bbox)
                face_crops.append(face)
            else:
                warnings.warn(f"图像中未检测到人脸: {img_path}")
        
        if len(face_crops) == 0:
            raise RuntimeError("所有图像中均未检测到人脸")
        
        # 转换为数组
        faces = np.stack(face_crops)
        
        if return_tensor:
            faces = torch.from_numpy(faces).float()
            faces = faces.permute(0, 3, 1, 2)
            if normalize:
                faces = faces / 255.0
        else:
            if normalize:
                faces = faces.astype(np.float32) / 255.0
        
        return faces
    
    def __call__(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """便捷调用接口
        
        Args:
            source: 视频文件路径或图像路径列表
            **kwargs: 传递给 extract_from_video 或 extract_from_images
        
        Returns:
            面部图像序列
        """
        if isinstance(source, (list, tuple)):
            return self.extract_from_images(source, **kwargs)
        else:
            return self.extract_from_video(source, **kwargs)


# 示例用法
if __name__ == '__main__':
    # 创建提取器
    extractor = FaceExtractor(
        method='opencv',
        target_size=(160, 160),
        num_frames=16
    )
    
    # 从视频提取
    try:
        faces = extractor('path/to/video.mp4')
        print(f"提取的人脸序列形状: {faces.shape}")  # (16, 3, 160, 160)
    except Exception as e:
        print(f"提取失败: {e}")
