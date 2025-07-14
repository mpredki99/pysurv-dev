from ..xyw_sw_build_strategy import XYWsWBuildStrategy
from .speed_sw_builder import SpeedSWBuilder
from .speed_xyw_builder import SpeedXYWBuilder


class SpeedStrategy(XYWsWBuildStrategy):
    def __init__(self, dataset, matrix_x_indexer, default_sigmas):
        super().__init__()
        self._xyw_builder = SpeedXYWBuilder(dataset, matrix_x_indexer, default_sigmas)
        self._sw_builder = SpeedSWBuilder(dataset, matrix_x_indexer, default_sigmas)

    @property
    def xyw_builder(self):
        return self._xyw_builder

    @property
    def sw_builder(self):
        return self._sw_builder
