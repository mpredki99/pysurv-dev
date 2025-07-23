# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import warnings
from typing import Any, Generator, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import DimensionError

from pysurv.validators._models import ControlPointModel
from pysurv.warnings._warnings import GeometryAssigningWarning, InvalidGeometryWarning


class Controls(gpd.GeoDataFrame):
    """
    Controls dataset class for storing and managing control points including geometry type.

    Internaly sets index from control point 'id'. It uses virtual geometry attribute what
    can have impact on performance. If you want to use concrete geometry attribute use
    to_geodataframe method.

    Inherits from
    -------------
    geopandas.GeoDataFrame
    """

    _metadata = ["_geometry_column_name", "_geometry_crs"]

    def __init__(
        self,
        data: Iterable | dict | pd.DataFrame,
        swap_xy: bool = False,
        crs: Any = None,
        geometry_name: str = "geometry",
        **kwargs,
    ) -> None:

        _first_init = kwargs.pop("_first_init", True)
        super().__init__(data, **kwargs)

        self._geometry_column_name = self._validate_name(geometry_name)
        self._geometry_crs = self._get_crs_from_user(crs=crs, epsg=None)

        if _first_init:
            self.set_index(["id"], inplace=True)

        if swap_xy:
            self.swap_xy(inplace=True)

    def _validate_name(self, name):
        name = name.strip()
        if not name.isidentifier():
            raise ValueError("Name is not valid identifier.")
        return name

    @classmethod
    def _geodataframe_constructor_with_fallback(cls, *args, **kwargs):
        # if issubclass(cls, Controls):
        return Controls(*args, **kwargs, _first_init=False)
        # return super()._geodataframe_constructor_with_fallback(cls, *args, **kwargs)

    def __contains__(self, name: Any) -> bool:
        """Add check if a column or the virtual geometry exists in the Controls dataset."""
        if name == self.active_geometry_name:
            return True
        return super().__contains__(name)

    def __getattr__(self, name: str):
        if name == self.active_geometry_name and name != "geometry":
            return self.geometry
        return super().__getattr__(name)

    def __getitem__(self, name: str | list) -> pd.Series | pd.DataFrame:
        """Include virtual geometry for slicing the dataset."""
        if isinstance(name, str) and name == self.active_geometry_name:
            return self.geometry

        if isinstance(name, (list, pd.Index)) and self.active_geometry_name in name:
            name = list(name)
            name.remove(self.active_geometry_name)
            subset = super().__getitem__(name)
            subset = subset.join(self.geometry)

        subset = super().__getitem__(name)

        if len(subset.shape) == 1:
            return subset

        return self._constructor(
            subset, crs=self.crs, geometry_name=self.active_geometry_name
        )

    def _get_crs_from_user(self, crs: Any = None, epsg: int | None = None):
        """Return a CRS object from user input."""
        from pyproj import CRS

        if crs is not None:
            return CRS.from_user_input(crs)
        elif epsg is not None:
            return CRS.from_epsg(epsg)

    def iterfeatures(
        self,
        na: str | None = "null",
        include_coordinates_columns: bool = False,
        show_bbox: bool = False,
        drop_id: bool = False,
    ) -> Generator[dict[str, str], Any, None]:
        """Yield features from the Controls dataset."""
        if include_coordinates_columns:
            frame = self
        else:
            frame = self.drop(columns=self.coordinate_columns)

        gdf = gpd.GeoDataFrame(frame, geometry=self.geometry, crs=self.crs)

        features_generator = gdf.iterfeatures(
            na=na, show_bbox=show_bbox, drop_id=drop_id
        )
        return features_generator

    @property
    def crs(self):
        return self._geometry_crs

    def set_crs(
        self,
        crs: Any = None,
        epsg: int | None = None,
        inplace: bool = False,
        allow_override: bool = False,
    ):
        """Set CRS of the virtual geometry."""
        crs = self._get_crs_from_user(crs=crs, epsg=epsg)

        if not allow_override and self.crs is not None and self.crs != crs:
            raise ValueError("Passed CRS not valid or geometry already has CRS.")

        frame = self if inplace else self.copy()
        frame._geometry_crs = crs

        assert frame._geometry_crs == frame.geometry.crs

        if not inplace:
            return frame

    def to_crs(self, crs: Any = None, epsg: int | None = None, inplace: bool = False):
        """Trasform current geometry CRS into new one."""
        new_geom = self.geometry.to_crs(crs=crs, epsg=epsg)
        new_coords = new_geom.get_coordinates(include_z=True)

        if not all(new_geom.is_valid):
            warnings.warn(
                "Some of the features have invalid geometry. Please check the results.",
                InvalidGeometryWarning,
            )

        frame = self if inplace else self.copy()
        frame.set_crs(new_geom.crs, inplace=True, allow_override=True)
        frame[self.coordinate_columns] = new_coords

        if not inplace:
            return frame

    def rename_geometry(self, col: str, inplace: bool = False):
        """Change name of the virtual geometry attribute."""
        frame = self if inplace else self.copy()
        name = self._validate_name(col)

        if frame._geometry_column_name != "geometry":
            frame.__delattr__(frame._geometry_column_name)

        frame._geometry_column_name = name
        frame.__setattr__(name, frame.geometry)

        if not inplace:
            return frame

    @property
    def geometry(self) -> gpd.GeoSeries:
        """Return virtual geometry based on coordinates columns."""
        coords = self.coordinates.dropna(how="all")

        if "x" not in coords.columns:
            coords["x"] = np.inf
        if "y" not in coords.columns:
            coords["y"] = np.inf

        return gpd.GeoSeries(
            gpd.points_from_xy(**coords),
            index=coords.index,
            crs=self.crs,
            name=self.active_geometry_name,
        )

    @geometry.setter
    def geometry(self, value: Any) -> None:
        """Setter for virtual geometry. Warns user that virtual geometry can not be modyfied directly."""
        warnings.warn(
            "Controls GeoDataFrame uses virtual geometry. Use coordinates columns instead.",
            GeometryAssigningWarning,
        )

    @property
    def x(self) -> pd.Series:
        """Return the 'x' coordinate column."""
        if "x" not in self.coordinate_columns:
            raise DimensionError("No 'x' geometry column.")
        return self["x"]

    @property
    def y(self) -> pd.Series:
        """Return the 'y' coordinate column."""
        if "y" not in self.coordinate_columns:
            raise DimensionError("No 'y' geometry column.")
        return self["y"]

    @property
    def z(self) -> pd.Series:
        """Return the 'z' coordinate column."""
        if "z" not in self.coordinate_columns:
            raise DimensionError("No 'z' geometry column.")
        return self["z"]

    @property
    def coordinate_columns(self) -> pd.Index:
        """Return columns corresponding to coordinate values."""
        return self.columns[
            self.columns.isin(ControlPointModel.COLUMN_LABELS["coordinates"])
        ]

    @property
    def sigma_columns(self) -> pd.Index:
        """Return columns corresponding to coordinate standard deviations."""
        return self.columns[self.columns.isin(ControlPointModel.COLUMN_LABELS["sigma"])]

    @property
    def coordinates(self) -> pd.DataFrame:
        """Return data subset of the coordinate columns."""
        return self[self.coordinate_columns]

    @property
    def coordinate_sigmas(self) -> pd.DataFrame:
        """Return data subset of coordinate columns standard deviations."""
        return self[self.sigma_columns]

    def swap_xy(self, inplace: bool = False) -> None:
        """Swap the 'x' and 'y' coordinate columns and their sigmas in the dataset."""
        if {"x", "y"}.issubset(self.coordinate_columns):
            frame = self if inplace else self.copy()
            frame.rename(
                columns={"x": "y", "y": "x", "sx": "sy", "sy": "sx"}, inplace=True
            )
        if not inplace:
            return frame

    def copy(self, deep=True):
        """Return a copy of the Controls object."""
        crs = self._geometry_crs
        geometry_name = self._geometry_column_name
        frame = super().copy(deep=deep)
        return self._constructor(
            frame, crs=crs, geometry_name=geometry_name, swap_xy=False
        )

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame representation of the controls dataset."""
        gdf = gpd.GeoDataFrame(self, geometry=self.geometry, crs=self._geometry_crs)
        return gdf.rename_geometry(self.active_geometry_name)
