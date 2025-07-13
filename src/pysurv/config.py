from pysurv.models import validate_angle_unit

__all__ = ["config"]


class Config:
    def __init__(self):
        self._angle_unit = "grad"

    @property
    def angle_unit(self):
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, new_angle_unit):
        new_angle_unit = validate_angle_unit(new_angle_unit)
        self._angle_unit = new_angle_unit

    def __str__(self):
        str = ""
        for key, value in self.__dict__.items():
            str += f"{key}: {value}\n"[1:]
        return str


config = Config()
