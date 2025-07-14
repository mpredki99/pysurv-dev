import numpy as np
import pandas as pd

from ..constants import INVALID_INDEX


class MatrixXIndexer:
    def __init__(self, dataset):

        self._controls = dataset.controls
        self._stations = dataset.stations

        self._coordinates_indices = None
        self._orientations_indices = None

    @property
    def coordinates_indices(self):
        if self._coordinates_indices is None:
            self._map_coordinates_index()
        return self._coordinates_indices

    @property
    def orientations_indices(self):
        if self._orientations_indices is None:
            if self._coordinates_indices is None:
                self._map_coordinates_index()
            self._map_orientations_index()
        return self._orientations_indices

    def _map_coordinates_index(self) -> None:
        coord_idx = self._controls.index
        coord_columns = self._controls.coordinates_columns

        coord_values = self._controls.coordinates.values
        mask = np.isfinite(coord_values)

        index_map = np.full(coord_values.shape, INVALID_INDEX, dtype=int)
        index_map[mask] = np.arange(np.count_nonzero(mask))

        self._coordinates_indices = pd.DataFrame(
            index_map, index=coord_idx, columns=coord_columns
        )

    def _map_orientations_index(self):
        orientations = self._stations.orientation

        mask = np.isfinite(orientations.values)
        start_idx = self._coordinates_indices.max().max() + 1
        end_idx = start_idx + np.count_nonzero(mask)

        index_values = np.full(mask.shape, INVALID_INDEX, dtype=int)
        index_values[mask] = np.arange(start_idx, end_idx)

        self._orientations_indices = pd.Series(
            index_values, index=orientations.index, name="orientation_idx"
        )
