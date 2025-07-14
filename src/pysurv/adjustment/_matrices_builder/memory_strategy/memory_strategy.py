from ..xyw_sw_build_strategy import XYWsWBuildStrategy
from .memory_sw_builder import MemorySWBuilder
from .memory_xyw_builder import MemoryXYWBuilder


class MemoryStrategy(XYWsWBuildStrategy):
    def __init__(self, dataset, matrix_x_indexer, default_sigmas):
        super().__init__()
        self._xyw_builder = MemoryXYWBuilder(dataset, matrix_x_indexer, default_sigmas)
        self._sw_builder = MemorySWBuilder(dataset, matrix_x_indexer, default_sigmas)

    @property
    def xyw_builder(self):
        return self._xyw_builder

    @property
    def sw_builder(self):
        return self._sw_builder
