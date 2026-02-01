import numpy as np
import warp as wp
import enum
from source.rtscene import RtScene
from source.canvas import Canvas

from pathlib import Path


@wp.struct
class HitRecord:
    t: float = float(0.0)
    p: wp.vec3 = wp.vec3(0.0, 0.0, 0.0)
    normal: wp.vec3 = wp.vec3(0.0, 0.0, 0.0)

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
    t_min: float,
    t_max: float,
    hit_record_array: wp.array(dtype=HitRecord),
    tid: int,
) -> bool:
    oc = ray.origin - center
    a = wp.dot(ray.direction, ray.direction)
    b = wp.dot(oc, ray.direction)
    c = wp.dot(oc, oc) - radius * radius
    discriminant = b * b - a * c

    if discriminant > 0.0:
        temp = (-b - wp.sqrt(discriminant)) / a
        if temp < t_max and temp > t_min:
            hit_record_array[tid].t = temp
            hit_record_array[tid].p = point_at_parameter(ray, hit_record_array[tid].t)
            hit_record_array[tid].normal = (hit_record_array[tid].p - center) / radius
            return True
        temp = (-b + wp.sqrt(discriminant)) / a
        if temp < t_max and temp > t_min:
            hit_record_array[tid].t = temp
            hit_record_array[tid].p = point_at_parameter(ray, hit_record_array[tid].t)
            hit_record_array[tid].normal = (hit_record_array[tid].p - center) / radius
            return True
    return False

# @wp.func
# def hit_multi_sphere(
#     ray: Ray,
#     p: wp.array(dtype=wp.vec3),
#     r: wp.array(dtype=float),
#     t_min: float,
#     t_max: float,
#     # hit_record: HitRecord,
# ) -> tuple[bool, HitRecord]:
#     hit_any = bool(False)
#     closet_so_far = t_max

#     hit_record = HitRecord()

#     for i in range(p.shape[0]):
#         temp = HitRecord()
#         if hit_sphere(ray, p[i], r[i], t_min, closet_so_far, temp):
#             hit_any = True
#             closet_so_far = temp.t
#             hit_record.t = temp.t
#             hit_record.p = temp.p
#             hit_record.normal = temp.normal
#     return hit_any, hit_record



@wp.func
def normal(
    ray: Ray,
    p: wp.array(dtype=wp.vec3), 
    r: wp.array(dtype=float),
    t_min: float,
    t_max: float,
    hit_record_array: wp.array(dtype=HitRecord),
    tid: int,
):


    # result, hit_record= hit_multi_sphere(ray, p, r, t_min, t_max)
    hit_any = bool(False)
    closet_so_far = t_max

    for i in range(p.shape[0]):
        # temp = HitRecord()
        if hit_sphere(ray, p[i], r[i], t_min, closet_so_far, hit_record_array, tid):
            hit_any = True
            closet_so_far = hit_record_array[tid].t
            # hit_record[0] = temp

    if hit_any:
        # n = wp.normalize(point_at_parameter(ray, hit_record.t) - wp.vec3(0.0, 0.0, -1.0))
        return 0.5 * wp.vec3(
            hit_record_array[tid].normal.x + 1.0,
            hit_record_array[tid].normal.y + 1.0,
            hit_record_array[tid].normal.z + 1.0,
        )
    unit_direction = wp.normalize(ray.direction)
    t = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - t) * wp.vec3(1.0, 1.0, 1.0) + t * wp.vec3(0.5, 0.7, 1.0)

@wp.kernel
def sphere_render(
    canvas_width: wp.int32,
    canvas_height: wp.int32,
    uvs: wp.array(dtype=wp.vec2),
    output: wp.array(dtype=wp.vec3),
    camera: wp.array(dtype=CameraConfig),
    p: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=float),
    hit_record_array: wp.array(dtype=HitRecord),
):
    tid = wp.tid()

    uv = uvs[tid]

    color = wp.vec3(0.0, 0.0, 0.0)

    # average multiple rays to get a more accurate color
    for i in range(100):
        seed_x = wp.uint32(tid * 7919 + i * 2)
        seed_y = wp.uint32(tid * 7919 + i * 2 + 1)

        cam = camera[0]
        # uv.x += wp.randf(seed) / float(canvas_width)
        # uv.y += wp.randf(seed) / float(canvas_height)

        temp = wp.vec2(
            uv.x + (wp.randf(seed_x) / float(canvas_width)),
            uv.y + (wp.randf(seed_y) / float(canvas_height)),
        )



        ray = Ray()
        ray.origin = cam.origin
        ray.direction = get_ray_direction(
            cam, temp)


        color += normal(ray, p, r, 0.0, wp.inf, hit_record_array, tid)

    color /= 100.0



    # if result:
    #     color = 0.5 * wp.normalize(hit_record.normal) + wp.vec3(0.5, 0.5, 0.5)

    #     # print(color)
    # else:
    #     color = wp.vec3(0.0, 0.0, 0.0)
    # if hit_record.t > 0.0:
    #     color = hit_record.normal
    # else:


    # color = normal(ray, p[2], r[2])
    # color = normal(ray, p[1], r[1])
    # color = normal(ray, p[0], r[0])

    #TODO : hit multiple spheres and get color
    #... and more complex logic

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
    temp_path = Path("./temp")
    temp_path.mkdir(exist_ok=True)
    
    canvas = Canvas(
        width=1920,
        height=1080,
    )
    device = wp.get_preferred_device()
    RtScene.set_device(device)
    RtScene.add_sphere(position=(0.0, 0.0, -1.0), radius=0.5)
    RtScene.add_sphere(position=(-0.5, -0.15, -0.55), radius=0.15)
    RtScene.add_sphere(position=(2.0, 0.0, -2.0), radius=0.5)
    RtScene.add_sphere(position=(0.0, -100.5, -1.0), radius=100.0)

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

    hit_record = wp.array(dtype=HitRecord, shape=(canvas.width * canvas.height, ), device=device)

    # 运行 ray tracing kernel
    for i in range(5):
        with wp.ScopedTimer("rendering", active=True):
            wp.launch(
                kernel=sphere_render,
                dim=canvas.width * canvas.height,
                inputs=[
                    canvas.width, canvas.height,
                    uvs_wp, output, camera_arr,
                    p, r,
                    hit_record,
                ],
            )



    save_to_image(output, str(temp_path / "ray_trace_result.png"))
    
    
