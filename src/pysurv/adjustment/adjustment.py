from .lsq_matrices import LSQMatrices


class Adjustment:
    def __init__(
        self, dataset, method="weighted", default_sigmas=None, comutations_priority=None
    ):
        self._dataset = dataset
        self._lsq_matrices = LSQMatrices(
            self._dataset,
            method=method,
            default_sigmas=default_sigmas,
            comutations_priority=comutations_priority,
        )

    @property
    def method(self):
        return self._lsq_matrices._method

    @property
    def comutations_priority(self):
        from ._xyw_matrices_builder.xyw_build_strategy_factory import strategies

        for strategy_name, strategy in strategies.items():
            if isinstance(self._lsq_matrices._xyw_build_strategy, strategy):
                return strategy_name
