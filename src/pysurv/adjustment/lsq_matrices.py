import numpy as np
import pandas as pd

from ._xyw_matrices_builder.xyw_build_strategy_factory import get_strategy
from .sigma_config import sigma_config


class LSQMatrices:

    def __init__(
        self, dataset, method="weighted", default_sigmas=None, comutations_priority=None
    ):
        self._dataset = dataset
        self._X = None
        self._Y = None
        self._W = None

        default_sigmas = default_sigmas or sigma_config.default_index
        self._default_sigmas = sigma_config[default_sigmas]

        self._method = self._validate_method(method)

        self._xyw_build_strategy = get_strategy(
            parent=self,
            name=comutations_priority,
        )

        self._cordinates_index_in_x_matrix = None
        self._orientations_index_in_x_matrix = None

    def _validate_method(self, method):
        return method

    @property
    def dataset(self):
        return self._dataset

    @property
    def matrix_X(self):
        return self._X

    @property
    def matrix_Y(self):
        return self._Y

    @property
    def matrix_W(self):
        return self._W

    @property
    def default_sigmas(self):
        return self._default_sigmas

    @property
    def method(self):
        return self._method

    @property
    def cordinates_index_in_x_matrix(self):
        return self._cordinates_index_in_x_matrix

    @property
    def orientations_index_in_x_matrix(self):
        return self._orientations_index_in_x_matrix

    def update_stations_orientation(self):
        if "hz" in self.dataset.measurements.angular_measurements_columns:
            hz = self.dataset.measurements.hz.dropna()
            stn_pk = hz.reset_index()["stn_pk"]
            first_hz_occurence = stn_pk.drop_duplicates().index

            self.dataset.stations.append_oreintation_constant(
                hz.iloc[first_hz_occurence], self.dataset.controls
            )

    def coordinates_index(self):
        coordinates = self._dataset.controls.coordinates

        notna_coords = pd.notna(coordinates.values)
        ctrl_idx = coordinates.values.flatten()
        ctrl_idx = np.nan_to_num(ctrl_idx, nan=-1, neginf=-1, posinf=-1)
        ctrl_idx[notna_coords.flatten()] = np.arange(notna_coords.sum().sum())
        ctrl_idx = ctrl_idx.reshape(coordinates.shape).astype(int)

        self._cordinates_index_in_x_matrix = pd.DataFrame(
            ctrl_idx, index=coordinates.index, columns=coordinates.columns
        )

    def orientation_index(self):
        if not "orientation" in self._dataset.stations.columns:
            return
        orientations = self._dataset.stations.orientation

        notna_orientation = pd.notna(orientations)
        orientations_idx = orientations.fillna(-1)
        start_idx = self._cordinates_index_in_x_matrix.max().max() + 1
        end_idx = start_idx + notna_orientation.sum().sum()
        orientations_idx[notna_orientation] = np.arange(start_idx, end_idx)

        self._orientations_index_in_x_matrix = orientations_idx.rename(
            "orientation_idx"
        ).astype(int)

    def build_matrices(self):
        self.update_stations_orientation()

        self.coordinates_index()
        self.orientation_index()

        self._xyw_build_strategy.calculate_weights = self._method != "ordinary"
        self._X, self._Y, self._W = self._xyw_build_strategy.build()
        print(self._X, self._Y, self._W)

    def update_xy_matrices(self):
        self.update_stations_orientation()
        self._xyw_build_strategy.calculate_weights = False
        self._X, self._Y = self._xyw_build_strategy.build()
