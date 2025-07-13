from .lsq_matrices import LSQMatrices


class Adjustment:
    def __init__(
        self,
        dataset,
        method="weighted",
        free_adjustment=None,
        default_sigmas=None,
        comutations_priority=None,
    ):
        self._dataset = dataset
        self._lsq_matrices = LSQMatrices(
            self._dataset,
            method=method,
            free_adjustment=free_adjustment,
            default_sigmas=default_sigmas,
            comutations_priority=comutations_priority,
        )

    @property
    def method(self):
        return self._lsq_matrices._method

    @property
    def comutations_priority(self):
        from ._matrices_builder.matrices_builder_factory import strategies

        for strategy_name, strategy in strategies.items():
            if isinstance(self._lsq_matrices._matrices_build_strategy, strategy):
                return strategy_name
