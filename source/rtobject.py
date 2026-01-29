import enum

class HitableType(enum.IntEnum):
    SPHERE = 0

class RtObject:
    def __init__(self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.position = position
    
    def get_position(self) -> tuple[float, float, float]:
        return self.position

class RtSphere(RtObject):
    def __init__(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 0.5,
    ):
        self.radius = radius
        super().__init__(position)

    def get_radius(self) -> float:
        return self.radius