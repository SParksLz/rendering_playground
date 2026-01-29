from .rtobject import RtObject, RtSphere, HitableType
import warp as wp

class RtScene:
    _object_map: dict[HitableType, list[RtObject]] = {ht: [] for ht in HitableType}
    _device: wp.Device = None

    @classmethod
    def _add_object(cls, object: RtObject, type: HitableType):
        RtScene._object_map[type].append(object)
    
    @classmethod
    def set_device(cls, device: wp.Device):
        RtScene._device = device

    @classmethod
    def add_sphere(cls, position: tuple[float, float, float], radius: float):
        RtScene._add_object(RtSphere(position=position, radius=radius), HitableType.SPHERE)

    @classmethod
    def get_object_by_type(cls, type: HitableType) -> list[RtObject]:
        return RtScene._object_map[type]

    @classmethod
    def get_all_objects(cls) -> list[RtObject]:
        return [obj for objects in RtScene._object_map.values() for obj in objects]

    @classmethod
    def clear(cls):
        for lst in RtScene._object_map.values():
            lst.clear()

    @classmethod
    def test_get_sphere_warp_array(cls):
        spheres = RtScene.get_object_by_type(HitableType.SPHERE)
        p = wp.array([sphere.get_position() for sphere in spheres], dtype=wp.vec3, device=RtScene._device)
        r = wp.array([sphere.get_radius() for sphere in spheres], dtype=float, device=RtScene._device)
        return p, r