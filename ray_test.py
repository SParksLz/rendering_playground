import numpy as np
import warp as wp


class Canvas:
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
    ):
        self.width = width
        self.height = height


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


@wp.struct
class Ray:
    a: wp.vec3
    b: wp.vec3

# class Camera:

# p(t) = A + t*B.
# t*t*dot(B,B) + 2*t*dot(A-C,A-C) + dot(C,C) - R*R = 0
@wp.func
def hit_sphere(
    ray: Ray, 
    center: wp.vec3,
    radius: float,
) -> float:
    oc = ray.a - center
    a = wp.dot(ray.b, ray.b)
    b = 2.0 * wp.dot(oc, ray.b)
    c = wp.dot(oc, oc) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return -1.0
    else:
        return (-b - wp.sqrt(discriminant)) / (2.0 * a)

@wp.kernel
def render(
    uvs: wp.array(dtype=wp.vec2),
    output: wp.array(dtype=wp.vec3),
):

    ray = Ray()
    ray.a = wp.vec3(0.0, 0.0, 0.0)
    ray.b = wp.vec3(0.0, 0.0, 1.0)
    tid = wp.tid()
    uv = uvs[tid]
    
    # 简单的 ray tracing 示例：根据 uv 坐标生成颜色
    # 这里可以替换为实际的 ray tracing 逻辑
    result = hit_sphere(ray, wp.vec3(0.0, 0.0, -3.0), 0.2)
    if result > 0.0:
        color = wp.vec3(1.0, 0.0, 0.0)
    else:
        color = wp.vec3(0.0, 0.0, 0.0)

    output[tid] = color

    # color = wp.vec3(uv[0], uv[1], 0.0)
    output[tid] = color


def save_to_image(
    data: wp.array(dtype=wp.vec3),
    save_path: str,
):
    output_np = data.numpy()
    
    # 重塑为 (height, width, 3) 格式
    # 注意：需要翻转垂直方向，因为图像坐标从顶部开始，而我们的 uv 坐标从底部开始
    image = output_np.reshape(canvas.height, canvas.width, 3)
    # image = np.flipud(image)  # 翻转垂直方向
    
    # 将浮点数 [0, 1] 转换为 uint8 [0, 255]
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)

    # 保存为 PNG 图片
    try:
        from PIL import Image
        img = Image.fromarray(image, 'RGB')
        img.save(save_path)
        print(f"渲染结果已保存到 {save_path} ({canvas.width}x{canvas.height})")
    except ImportError:
        # 如果没有 PIL，使用 matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.imsave(save_path, image)
            print(f"渲染结果已保存到 {save_path} ({canvas.width}x{canvas.height})")
        except ImportError:
            print("需要安装 PIL 或 matplotlib 来保存图片")
            print(f"图像数组形状: {image.shape}, 数据类型: {image.dtype}")
    



if __name__ == "__main__":
    canvas = Canvas()
    uvs = canvas.get_uvs()
    
    # 将 uvs 转换为 warp array
    uvs_wp = wp.array(uvs, dtype=wp.vec2, device=wp.get_preferred_device())

    
    # 创建输出数组
    output = wp.zeros(canvas.width * canvas.height, dtype=wp.vec3, device=uvs_wp.device)
    
    # 运行 ray tracing kernel
    wp.launch(
        kernel = render, 
        dim = canvas.width * canvas.height, 
        inputs = [uvs_wp, output],
    )

    save_to_image(output, "ray_trace_result.png")
    
    
