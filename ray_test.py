import numpy as np
import warp as wp
import enum
from source.rtscene import RtScene
from source.canvas import Canvas


@wp.struct
class CameraConfig:
    """相机/幕布配置：感光元件位置 + 幕布在 3D 空间中的矩形（左下角 + 水平/垂直边向量）"""
    origin: wp.vec3           # 感光元件位置（射线起点）
    lower_left_corner: wp.vec3 # 幕布左下角
    horizontal: wp.vec3       # 幕布水平方向向量（宽）
    vertical: wp.vec3         # 幕布垂直方向向量（高）


@wp.func
def get_ray_direction(cam: CameraConfig, uv: wp.vec2) -> wp.vec3:
    """根据 uv [0,1]^2 计算射线方向：幕布上对应点的位置向量（因 origin 为原点）"""
    return cam.lower_left_corner + uv.x * cam.horizontal + uv.y * cam.vertical


@wp.struct
class Ray:
    origin: wp.vec3
    direction: wp.vec3


@wp.func
def point_at_parameter(ray: Ray, t: float) -> wp.vec3:
    return ray.origin + t * ray.direction


# @wp.struct
# class Sphere:
    # position: wp.array(dtype=wp.vec3)
    # radius:wp.array(dtype=float)
# class Camera:

# p(t) = A + t*B.
# t*t*dot(B,B) + 2*t*dot(A-C,A-C) + dot(C,C) - R*R = 0
@wp.func
def hit_sphere(
    ray: Ray, #ray
    center: wp.vec3, #position of sphere
    radius: float, #radius of sphere
) -> float:
    oc = ray.origin - center
    a = wp.dot(ray.direction, ray.direction)
    b = 2.0 * wp.dot(oc, ray.direction)
    c = wp.dot(oc, oc) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return -1.0
    else:
        return (-b - wp.sqrt(discriminant)) / (2.0 * a)
    # return discriminant > 0
@wp.func
def normal(
    ray: Ray,
    p: wp.vec3, 
    r: float,
):
    result = hit_sphere(ray, p, r)
    if result > 0.0:
        n = wp.normalize(point_at_parameter(ray, result) - wp.vec3(0.0, 0.0, -1.0))
        return 0.5 * n + wp.vec3(0.5, 0.5, 0.5)
    unit_direction = wp.normalize(ray.direction)
    t = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - t) * wp.vec3(1.0, 1.0, 1.0) + t * wp.vec3(0.5, 0.7, 1.0)

@wp.kernel
def sphere_render(
    uvs: wp.array(dtype=wp.vec2),
    output: wp.array(dtype=wp.vec3),
    camera: wp.array(dtype=CameraConfig),
    p: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=float),
):
    tid = wp.tid()
    uv = uvs[tid]
    cam = camera[0]

    # sphere_count = p.shape[0]

    ray = Ray()
    ray.origin = cam.origin
    ray.direction = get_ray_direction(cam, uv)

    color = normal(ray, p[0], r[0])

    #TODO : hit sphere and get color

    # breakpoint()
    
    # 简单的 ray tracing 示例：根据 uv 坐标生成颜色
    # 这里可以替换为实际的 ray tracing 逻辑
    # result = hit_sphere(ray, p[0], r[0])
    # # print(result)
    # if result:
    #     color = wp.vec3(1.0, 0.0, 0.0)
    # else:
    #     color = wp.vec3(0.0, 0.0, 0.0)

    # output[tid] = color

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
    



def make_camera_config(
    origin=(0.0, 0.0, 0.0),
    viewport_height=2.0,
    aspect_ratio: float = 2.0,
    focal_length=1.0,
) -> "CameraConfig":
    """构造默认相机配置：幕布在 z=-focal_length，中心对准 origin。
    宽高比 aspect_ratio = viewport_width / viewport_height，应与 Canvas 的 width/height 一致，否则画面会拉伸。"""
    viewport_width = aspect_ratio * viewport_height
    llc = (
        -viewport_width / 2.0,
        -viewport_height / 2.0,
        -focal_length,
    )
    horz = (viewport_width, 0.0, 0.0)
    vert = (0.0, viewport_height, 0.0)
    cam = CameraConfig()
    cam.origin = wp.vec3(*origin)
    cam.lower_left_corner = wp.vec3(*llc)
    cam.horizontal = wp.vec3(*horz)
    cam.vertical = wp.vec3(*vert)
    return cam


if __name__ == "__main__":
    canvas = Canvas()
    device = wp.get_preferred_device()
    RtScene.set_device(device)
    RtScene.add_sphere(position=(0.0, 0.0, -1.0), radius=0.5)
    RtScene.add_sphere(position=(0.0, 0.5, -2.0), radius=0.5)
    RtScene.add_sphere(position=(0.0, -100.5, -1), radius=100.0)

    #sphere data
    p, r = RtScene.test_get_sphere_warp_array()


    # 相机宽高比必须与 canvas 一致，否则画面会拉伸/压扁
    aspect_ratio = canvas.aspect_ratio
    uvs = canvas.get_uvs()

    # 将 uvs 转换为 warp array
    uvs_wp = wp.array(uvs, dtype=wp.vec2, device=device)



    # 相机配置：幕布宽高比与 canvas 一致
    camera = make_camera_config(
        origin=wp.vec3(0.0, 0.0, 0.0),
        viewport_height=2.0,
        aspect_ratio=aspect_ratio,
        focal_length=1.0,
    )
    camera_arr = wp.array([camera], dtype=CameraConfig, device=device)

    # 创建输出数组
    output = wp.zeros(canvas.width * canvas.height, dtype=wp.vec3, device=device)

    # 运行 ray tracing kernel
    wp.launch(
        kernel=sphere_render,
        dim=canvas.width * canvas.height,
        inputs=[
            uvs_wp, output, camera_arr,
            p, r,
        ],
    )

    save_to_image(output, "ray_trace_result.png")
    
    
