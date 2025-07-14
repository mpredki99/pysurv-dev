from abc import ABC, abstractmethod


class XYWsWBuildStrategy(ABC):
    @property
    @abstractmethod
    def xyw_builder(self):
        pass

    @property
    @abstractmethod
    def sw_builder(self):
        pass
