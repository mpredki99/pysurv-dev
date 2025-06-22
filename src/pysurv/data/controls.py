import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import DimensionError

from pysurv.exceptions import GeometryCreationError
from pysurv.models import ControlPointModel


class Controls(gpd.GeoDataFrame):
    def __init__(self, data=None, *args, swap_xy=False, crs=None, **kwargs):
        if {"x", "y", "z"}.issubset(data.columns):
            self._init_with_3D_geometry(data=data, *args, crs=None, **kwargs)
        elif {"x", "y"}.issubset(data.columns):
            self._init_with_2D_geometry(data=data, *args, crs=None, **kwargs)
        elif {"z"}.issubset(data.columns):
            self._init_with_1D_geometry(data=data, *args, crs=None, **kwargs)
        else:
            self._init_with_no_geometry(data=data, *args, crs=None, **kwargs)

        self._validate_geometry_creation()

        if {"id"}.issubset(self.columns):
            self.set_index(["id"], inplace=True)

        if swap_xy:
            self.swap_xy()

    def _init_with_3D_geometry(self, data=None, *args, crs=None, **kwargs):
        super().__init__(
            data,
            *args,
            geometry=gpd.points_from_xy(
                data.pop("x"), data.pop("y"), data.pop("z"), crs=crs
            ),
            **kwargs,
        )

    def _init_with_2D_geometry(self, data=None, *args, crs=None, **kwargs):
        super().__init__(
            data,
            *args,
            geometry=gpd.points_from_xy(data.pop("x"), data.pop("y"), crs=crs),
            **kwargs,
        )

    def _init_with_1D_geometry(self, data=None, *args, crs=None, **kwargs):
        super().__init__(
            data,
            *args,
            geometry=gpd.points_from_xy(np.inf, np.inf, data.pop("z"), crs=crs),
            **kwargs,
        )

    def _init_with_no_geometry(self, data=None, *args, crs=None, **kwargs):
        super().__init__(data, *args, crs=crs, **kwargs)

    def _validate_geometry_creation(self):
        if any(col in self.columns for col in ["x", "y", "z"]):
            cols = [col for col in ["x", "y", "z"] if col in self.columns]
            raise GeometryCreationError(
                f"Geometry was not created from column with coordinates: {cols}"
            )

    @property
    def x(self):
        x_geom = self.geometry.x
        if np.all(x_geom == np.inf) or np.all(x_geom.isna()):
            raise DimensionError("No 'x' geometry column.")
        return x_geom.where(x_geom != np.inf)

    @property
    def y(self):
        y_geom = self.geometry.y
        if np.all(y_geom == np.inf) or np.all(y_geom.isna()):
            raise DimensionError("No 'y' geometry column.")
        return y_geom.where(y_geom != np.inf)

    @property
    def z(self):
        z_geom = self.geometry.z
        if np.all(z_geom == np.inf) or np.all(z_geom.isna()):
            raise DimensionError("No 'z' geometry column.")
        return z_geom.where(z_geom != np.inf)

    @property
    def geometry_columns(self, include_z=True):
        return self.get_control_coordinates(include_z=include_z).columns

    @property
    def sigma_columns(self):
        return self.columns[self.columns.isin(ControlPointModel.COLUMN_LABELS["sigma"])]

    def swap_xy(self):
        swapped_xy = False
        if {"x", "y", "z"}.issubset(self.geometry_columns):
            self._swap_xy_3D_point()
            swapped_xy = True
        elif {"x", "y"}.issubset(self.geometry_columns):
            self._swap_xy_2D_point()
            swapped_xy = True

        if swapped_xy:
            self._swap_xy_sigma()

    def _swap_xy_3D_point(self):
        self.geometry = gpd.points_from_xy(self.y, self.x, self.z)

    def _swap_xy_2D_point(self):
        self.geometry = gpd.points_from_xy(self.y, self.x)

    def _swap_xy_sigma(self):
        self.rename(columns={"sx": "sy", "sy": "sx"}, inplace=True)

    def get_control_coordinates(self, include_z=True):
        coordinates = self.geometry.get_coordinates(include_z=include_z)
        coordinates = coordinates.where(coordinates != np.inf)

        return coordinates.dropna(axis=1, how="all")

    def display(self, include_z=True):
        table = self.get_control_coordinates(include_z=include_z).join(
            self.drop(columns=["geometry"])
        )
        if not include_z:
            table.drop(
                columns=[
                    "sz",
                ],
                errors="ignore",
                inplace=True,
            )
        return table

    def copy(self, *args, **kwargs):
        gdf = super().copy(*args, **kwargs)
        return Controls(gdf, swap_xy=False)
