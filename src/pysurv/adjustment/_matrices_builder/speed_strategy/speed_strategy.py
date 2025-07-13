import numpy as np

from ..matrices_builder import MatricesBuilder
from .speed_xyw_builder import SpeedXYWBuilder
from .speed_sw_builder import SpeedSWBuilder


class SpeedStrategy(MatricesBuilder):
    def __init__(self, parent):
        super().__init__(parent)

    def build_xyw(self, calculate_weights):
        speed_xyw_builder = SpeedXYWBuilder(parent=self)
        return speed_xyw_builder.build_xyw(calculate_weights)
        
    
    def build_sw(self):
        speed_sw_builder = SpeedSWBuilder(parent=self)
        return speed_sw_builder.build_sw()
        