import warnings

import geopandas as gpd
import numpy as np
from shapely.errors import DimensionError

from pysurv.models import ControlPointModel


class Controls(gpd.GeoDataFrame):
    def __init__(
        self,
        data=None,
        *args,
        swap_xy=False,
        crs=None,
        geometry_name="geometry",
        **kwargs,
    ):
        super().__init__(data, *args, **kwargs)

        self._geometry_column_name = geometry_name
        self._geometry_crs = self._get_crs_from_user(crs=crs, epsg=None)

        if {"id"}.issubset(self.columns):
            self.set_index(["id"], inplace=True)

        if swap_xy:
            self.swap_xy()

    def __contains__(self, key):
        if key == self._geometry_column_name:
            return True
        return super().__contains__(key)

    def __getattr__(self, name):
        if name == self._geometry_column_name:
            return self.geometry
        else:
            super().__getattr__(name)

    def __getitem__(self, name):
        if isinstance(name, str) and name == self._geometry_column_name:
            return self.geometry

        if isinstance(name, list) and self._geometry_column_name in name:
            name.remove(self._geometry_column_name)
            geometry_name = self._geometry_column_name
            crs = self._geometry_crs
            frame = super().__getitem__(name)
            frame_with_geom = frame.join(self.geometry)
            return Controls(frame_with_geom, crs=crs, geometry_name=geometry_name)

        return super().__getitem__(name)

    def _get_crs_from_user(self, crs=None, epsg=None):
        from pyproj import CRS

        if crs is not None:
            return CRS.from_user_input(crs)
        elif epsg is not None:
            return CRS.from_epsg(epsg)

    def iterfeatures(
        self,
        na="null",
        include_coordinates_columns=False,
        show_bbox=False,
        drop_id=False,
    ):
        data = (
            self
            if include_coordinates_columns
            else self.drop(columns=self.coordinates_columns)
        )
        geometry = self.geometry
        crs = self._geometry_crs

        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
        gdf.rename_geometry(self._geometry_column_name, inplace=True)

        features_generator = gdf.iterfeatures(
            na=na, show_bbox=show_bbox, drop_id=drop_id
        )
        return features_generator

    def set_crs(self, crs=None, epsg=None, inplace=False, allow_override=False):
        crs = self._get_crs_from_user(crs=crs, epsg=epsg)

        if (
            not allow_override
            and self._geometry_crs is not None
            and self._geometry_crs == crs
        ):
            raise ValueError(
                "Passed crs not valid or already exists with the same value."
            )

        gdf = self if inplace else self.copy()
        gdf._geometry_crs = crs

        assert gdf._geometry_crs == gdf.geometry.crs

        if not inplace:
            return gdf

    def to_crs(self, crs=None, epsg=None, inplace=False):
        new_geom = self.geometry.to_crs(crs=crs, epsg=epsg)
        new_coords = new_geom.get_coordinates(include_z=True)

        if not all(new_geom.is_valid):
            warnings.warn(
                "Some of the features have invalid geometry. Please check the results."
            )

        gdf = self if inplace else self.copy()
        gdf.set_crs(new_geom.crs, inplace=True)
        gdf[self.coordinates_columns] = new_coords[self.coordinates_columns]

        if not inplace:
            return gdf

    def rename_geometry(self, col, inplace=False):

        gdf = self if inplace else self.copy()

        name = col.strip()

        if not name.isidentifier():
            raise ValueError("Name is not valid identifier.")

        if gdf._geometry_column_name != "geometry":
            gdf.__delattr__(gdf._geometry_column_name)

        gdf._geometry_column_name = name
        gdf.__setattr__(name, gdf.geometry)

        if not inplace:
            return gdf

    @property
    def geometry(self):

        geom = self[self.coordinates_columns].dropna(how="all")

        if "x" not in geom.columns:
            geom["x"] = np.inf
        if "y" not in geom.columns:
            geom["y"] = np.inf

        return gpd.GeoSeries(
            gpd.points_from_xy(**geom),
            index=geom.index,
            crs=self._geometry_crs,
            name=self._geometry_column_name,
        )

    @geometry.setter
    def geometry(self, value):
        warnings.warn(
            "Controls GeoDataFrame uses virtual geometry. Use coordinate columns instead."
        )

    @property
    def x(self):
        if "x" not in self.coordinates_columns:
            raise DimensionError("No 'x' geometry column.")
        return self["x"]

    @property
    def y(self):
        if "y" not in self.coordinates_columns:
            raise DimensionError("No 'y' geometry column.")
        return self["y"]

    @property
    def z(self):
        if "z" not in self.coordinates_columns:
            raise DimensionError("No 'z' geometry column.")
        return self["z"]

    @property
    def coordinates(self):
        return self[self.coordinates_columns]

    @property
    def coordinates_columns(self):
        return self.columns[
            self.columns.isin(ControlPointModel.COLUMN_LABELS["coordinates"])
        ]

    @property
    def sigma_columns(self):
        return self.columns[self.columns.isin(ControlPointModel.COLUMN_LABELS["sigma"])]

    def swap_xy(self):
        if {"x", "y"}.issubset(self.coordinates_columns):
            self.rename(
                columns={"x": "y", "y": "x", "sx": "sy", "sy": "sx"}, inplace=True
            )

    def copy(self, *args, **kwargs):
        crs = self._geometry_crs
        gdf = super().copy(*args, **kwargs)
        return Controls(gdf, crs=crs, swap_xy=False)
