from .lsq_matrices import LSQMatrices


class Adjustment:
    def __init__(
        self,
        dataset,
        method="weighted",
        free_adjustment=None,
        default_sigmas_index=None,
        computations_priority=None,
    ):
        self._dataset = dataset
        self._method = method
        self._free_adjustment = free_adjustment
        self._computations_priority = computations_priority

        self._lsq_matrices = LSQMatrices(
            self._dataset,
            calculate_weights=self._method != "ordinary",
            default_sigmas_index=default_sigmas_index,
            computations_priority=computations_priority,
        )

    @property
    def method(self):
        return self._method

    @property
    def free_adjustment(self):
        return self._free_adjustment

    @property
    def computations_priority(self):
        self._computations_priority
