"""简单图像数据增强工具（教学示例）。

仅依赖 numpy 与 Pillow，覆盖水平翻转、随机旋转、随机亮度三种常见增强。
所有增强方法接受 PIL.Image.Image，返回 PIL.Image.Image，可在训练流水线中链式调用。
"""


import numpy as np
from PIL import Image


class ImageAugmenter:
    """对单张图像做常见数据增强。"""

    def __init__(self, seed=None):
        # 用独立随机数发生器，便于通过 seed 复现结果
        self._rng = np.random.default_rng(seed)

    def horizontal_flip(self, image: Image.Image) -> Image.Image:
        """水平翻转（左右镜像）。"""
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def random_rotation(self, image: Image.Image, max_angle: float = 30) -> Image.Image:
        """随机旋转，角度在 [-max_angle, max_angle] 度内均匀采样。"""
        angle = self._rng.uniform(-max_angle, max_angle)
        # expand=False 保持原始尺寸，超出的区域会被裁掉/补黑
        return image.rotate(angle)

    def random_brightness(
        self, image: Image.Image, min_factor: float = 0.5, max_factor: float = 1.5
    ) -> Image.Image:
        """随机调整亮度：factor<1 变暗，>1 变亮，结果裁剪回 [0,255]。"""
        factor = self._rng.uniform(min_factor, max_factor)
        arr = np.asarray(image, dtype=np.float32) * factor
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def to_array(image: Image.Image) -> np.ndarray:
        """转成 numpy 数组，形状为 (H, W, C) 或 (H, W)。"""
        return np.asarray(image)
