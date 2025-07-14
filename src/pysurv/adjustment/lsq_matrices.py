from ._matrices_builder.inner_constraints_builder.matrix_r_builder import MatrixRBuilder
from ._matrices_builder.inner_constraints_builder.matrix_sx_builder import (
    MatrixSXBuilder,
)
from ._matrices_builder.matrix_x_indexer import MatrixXIndexer
from ._matrices_builder.xyw_sw_build_strategy_factory import get_strategy


class LSQMatrices:

    def __init__(
        self,
        dataset,
        calculate_weights=True,
        default_sigmas_index=None,
        computations_priority=None,
    ):
        self._dataset = dataset
        self._calculate_weights = calculate_weights

        self._matrix_x_indexer = MatrixXIndexer(self._dataset)
        self._X = None
        self._Y = None
        self._W = None
        self._sW = None

        self._inner_constraints = None
        self._R = None
        self._sX = None

        self._matrices_xyw_sw_build_strategy = get_strategy(
            self._dataset,
            self._matrix_x_indexer,
            default_sigmas_index,
            name=computations_priority,
        )

        if "hz" in self.dataset.measurements.angular_measurements_columns:
            self._update_stations_orientation()

    @property
    def dataset(self):
        return self._dataset

    @property
    def calculate_weights(self):
        return self._calculate_weights

    @property
    def matrix_x_indexer(self):
        return self._matrix_x_indexer

    @property
    def matrix_X(self):
        if self._X is None:
            self._build_xyw_matrices()
        return self._X

    @property
    def matrix_Y(self):
        if self._Y is None:
            self._build_xyw_matrices()
        return self._Y

    @property
    def matrix_W(self):
        if self._W is None and self._calculate_weights:
            self._build_xyw_matrices()
        return self._W

    @property
    def matrix_sW(self):
        if self._sW is None:
            self._sW = self._matrices_xyw_sw_build_strategy.sw_builder.build(
                self.matrix_X.shape[1]
            )
        return self._sW

    @property
    def inner_constraints(self):
        if self._inner_constraints is None:
            self._apply_inner_constraints()
        return self._inner_constraints

    @property
    def matrix_R(self):
        if self._R is None:
            self._apply_inner_constraints()
        return self._R

    @property
    def matrix_sX(self):
        if self._sX is None:
            matrix_sx_builder = MatrixSXBuilder(
                self.dataset, self.matrix_x_indexer, self.matrix_X.shape[1]
            )
            self._sX = matrix_sx_builder.build()
        return self._sX

    def _build_xyw_matrices(self):
        self._X, self._Y, self._W = (
            self._matrices_xyw_sw_build_strategy.xyw_builder.build(
                calculate_weights=self._calculate_weights
            )
        )

    def _apply_inner_constraints(self):
        inner_constraints_builder = MatrixRBuilder(
            self.dataset, self.matrix_x_indexer, self.matrix_X.shape[1]
        )
        self._R, self._inner_constraints = inner_constraints_builder.build()

    def _update_stations_orientation(self):
        hz = self._dataset.measurements.hz.dropna()
        stn_pk = hz.reset_index()["stn_pk"]
        first_hz_occurence = stn_pk.drop_duplicates().index

        self._dataset.stations.append_oreintation_constant(
            hz.iloc[first_hz_occurence], self.dataset.controls
        )

    def update_xy_matrices(self):
        if "hz" in self.dataset.measurements.angular_measurements_columns:
            self._update_stations_orientation()
        self._X, self._Y, _ = self._matrices_xyw_sw_build_strategy.xyw_builder.build(
            calculate_weights=False
        )

    def update_w_matrix(self):
        pass

    def update_sw_matrix(self):
        pass
