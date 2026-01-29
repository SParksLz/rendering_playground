import numpy as np
import warp as wp

class Canvas:
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
    ):
        self.width = width
        self.height = height

    @property
    def aspect_ratio(self) -> float:
        """宽高比 width/height，应与 CameraConfig 的幕布宽高比一致，否则画面会拉伸。"""
        return self.width / self.height

    # (0, 0) is left bottom corner
    # (width: u, height: v) is right top corner
    def get_uvs(self) -> wp.array(dtype=wp.vec2):
        # 创建像素坐标网格
        # x从0到width-1, y从0到height-1
        x = np.arange(self.width, dtype=np.float32)
        y = np.arange(self.height, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        
        # 计算uv坐标
        # 使用 (width-1) 和 (height-1) 作为分母，确保边界像素精确映射到边界值
        # 这样右下角像素 (width-1, 0) 会映射到 (1.0, 0.0)
        # 左上角像素 (0, height-1) 会映射到 (0.0, 1.0)

        width = self.width - 1
        height = self.height - 1

        u = X / width
        v = (height - Y) / height

        # 展平并组合成 (width*height, 2) 的数组
        uvs = np.stack([u.flatten(), v.flatten()], axis=1)
        
        return uvs