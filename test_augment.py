"""ImageAugmenter 的 pytest 单元测试。

运行方式（在 py312 环境下）：
    python -m pytest test_augment.py -v
"""

import numpy as np
from PIL import Image

from augment import ImageAugmenter


def _make_image(size=(32, 32), channels=3) -> Image.Image:
    """生成一个随机的测试图像。"""
    return Image.fromarray(
        np.random.randint(0, 256, (*size, channels), dtype=np.uint8)
    )


def test_horizontal_flip_preserves_shape():
    """水平翻转应保证输出尺寸与输入一致。"""
    img = _make_image()
    out = ImageAugmenter().horizontal_flip(img)
    assert out.size == img.size


def test_random_brightness_stays_in_range():
    """随机亮度调整后，像素值仍应落在合法范围 [0, 255]。"""
    img = _make_image()
    out = ImageAugmenter(seed=42).random_brightness(img)
    arr = np.asarray(out)
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_seed_makes_augmentation_deterministic():
    """给定相同 seed，两次随机旋转结果应完全一致（可复现性）。"""
    img = _make_image()
    a1 = ImageAugmenter(seed=0).random_rotation(img)
    a2 = ImageAugmenter(seed=0).random_rotation(img)
    assert np.array_equal(np.asarray(a1), np.asarray(a2))
