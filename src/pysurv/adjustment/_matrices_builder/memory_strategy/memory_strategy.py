import numpy as np
import pandas as pd

from ..matrices_builder import MatricesBuilder
from .memory_xyw_builder import MemoryXYWBuilder
from .memory_sw_builder import MemorySWBuilder


class MemoryStrategy(MatricesBuilder):
    def __init__(self, parent):
        super().__init__(parent)

    def build_xyw(self, calculate_weights):
        memory_xyw_builder = MemoryXYWBuilder(parent=self)
        return memory_xyw_builder.build_xyw(calculate_weights)
        
    def build_sw(self):
        memory_sw_builder = MemorySWBuilder(parent=self)
        return memory_sw_builder.build_sw()
