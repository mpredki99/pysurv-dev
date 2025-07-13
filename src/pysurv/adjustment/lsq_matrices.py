import numpy as np
import pandas as pd

from pysurv.models.models import validate_method
from ._matrices_builder.inner_constraints_builder import InnerConstraintsBuilder
from ._matrices_builder.matrices_builder_factory import get_strategy
from .robust import *
from .sigma_config import sigma_config


class LSQMatrices:

    def __init__(
        self,
        dataset,
        method="weighted",
        free_adjustment=None,
        default_sigmas=None,
        comutations_priority=None,
    ):
        self._dataset = dataset
        self._X = None
        self._Y = None
        self._W = None

        self._inner_constraints = None
        self._R = None
        
        self._sW = None
        self._sX = None

        default_sigmas = default_sigmas or sigma_config.default_index
        self._default_sigmas = sigma_config[default_sigmas]

        self._method = validate_method(method)

        self._matrices_build_strategy = get_strategy(
            parent=self,
            name=comutations_priority,
        )

        self._cordinates_index_in_x_matrix = None
        self._orientations_index_in_x_matrix = None

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

    def build_xyw_matrices(self):
        self.update_stations_orientation()

        self.coordinates_index()
        self.orientation_index()

        calculate_w = self._method != "ordinary"
        self._X, self._Y, self._W = self._matrices_build_strategy.build_xyw(calculate_weights=calculate_w)
        print(self._X, self._Y, self._W)
        
        
    def update_xy_matrices(self):
        self.update_stations_orientation()

        self._X, self._Y, _ = self._matrices_build_strategy.build_xyw(calculate_weights=False)
        
    def update_w_matrix(self):
        pass

    def apply_inner_constraints(self):
        inner_constraints_builder = InnerConstraintsBuilder(parent=self)
        self._R, self._inner_constraints = inner_constraints_builder.build_r_matrix()
        print(self._R, self._inner_constraints)
    
    def build_sw_matrix(self):
        self._sW = self._matrices_build_strategy.build_sw()
        print(self._sW)
        
    def build_sx_matrix(self):
        self._sX = np.zeros(self._X.shape[1])
        coord_idx = self._cordinates_index_in_x_matrix.values.flatten() 
        coord_idx = coord_idx[coord_idx != -1]
        self._sX[coord_idx] = 1
        print(self._sX)
        

