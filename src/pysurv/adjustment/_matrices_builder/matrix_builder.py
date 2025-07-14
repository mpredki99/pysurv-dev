from abc import ABC, abstractmethod


class MatrixBuilder(ABC):
    def __init__(self, dataset, matrix_x_indexer):
        self._dataset = dataset
        self._matrix_x_indexer = matrix_x_indexer

    @abstractmethod
    def build(self):
        pass
