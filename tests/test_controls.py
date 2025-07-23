# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.errors import DimensionError

from pysurv.data import Controls
from pysurv.warnings import GeometryAssigningWarning, InvalidGeometryWarning


def test_set_index_init(valid_control_data: pd.DataFrame) -> None:
    """Test that index is set to 'id' on initialization."""
    controls = Controls(valid_control_data)
    assert controls.index.name == "id"


def test_swap_xy_init(control_data_without_sy: pd.DataFrame) -> None:
    """Test swap_xy argument on initialization."""
    controls = Controls(control_data_without_sy, swap_xy=True)

    assert "sx" not in controls.columns
    for point in controls.index:
        assert controls.at[point, "x"] == 200.00
        assert controls.at[point, "y"] == 100.00
        assert controls.at[point, "z"] == 300.00
        assert controls.at[point, "sy"] == 0.01
        assert controls.at[point, "sz"] == 0.02


def test_swap_xy_on_copy(control_data_without_sy: pd.DataFrame) -> None:
    """Test swap_xy on copy method returns a copy with swapped columns."""
    controls = Controls(control_data_without_sy)
    swapped = controls.swap_xy()

    assert "sx" not in swapped.columns
    for point in controls.index:
        assert swapped.at[point, "x"] == 200.00
        assert swapped.at[point, "y"] == 100.00
        assert swapped.at[point, "z"] == 300.00
        assert swapped.at[point, "sy"] == 0.01
        assert swapped.at[point, "sz"] == 0.02


def test_swap_xy_without_y(control_data_without_y: pd.DataFrame) -> None:
    """Test swap_xy when 'y' column is missing."""
    controls = Controls(control_data_without_y)
    controls.swap_xy(inplace=True)

    assert "sx" in controls.columns
    assert "y" not in controls.columns
    for point in controls.index:
        assert controls.at[point, "x"] == 100.00
        assert controls.at[point, "z"] == 300.00
        assert controls.at[point, "sx"] == 0.01
        assert controls.at[point, "sz"] == 0.02


def test_geometry_columns_name_init(valid_control_data: pd.DataFrame) -> None:
    """Test custom geometry column name on initialization."""
    controls = Controls(valid_control_data, geometry_name="custom_name")
    assert controls.active_geometry_name == "custom_name"


def test_contains(valid_control_data: pd.DataFrame) -> None:
    """Test __contains__ for geometry and ordinary column."""
    controls = Controls(valid_control_data)

    assert "geometry" in controls
    assert "x" in controls


def test_getattr(valid_control_data: pd.DataFrame) -> None:
    """Test __getattr__ for geometry name and ordinary column."""
    controls = Controls(valid_control_data)

    assert isinstance(controls.active_geometry_name, str)
    assert isinstance(controls.x, pd.Series)


def test_get_geom_attr_default_name(valid_control_data: pd.DataFrame) -> None:
    """Test geometry attribute with default name."""
    controls = Controls(valid_control_data)
    assert isinstance(controls.geometry, gpd.GeoSeries)


def test_get_geom_attr_custom_name(valid_control_data: pd.DataFrame) -> None:
    """Test geometry attribute with custom name."""
    controls = Controls(valid_control_data, geometry_name="custom_name")
    assert isinstance(controls.custom_name, gpd.GeoSeries)
    assert isinstance(controls.geometry, gpd.GeoSeries)


def test_getitem_geometry_column_name_default(valid_control_data: pd.DataFrame) -> None:
    """Test __getitem__ for default geometry column name."""
    controls = Controls(valid_control_data)
    assert isinstance(controls["geometry"], gpd.GeoSeries)


def test_getitem_geometry_column_name_custom(valid_control_data: pd.DataFrame) -> None:
    """Test __getitem__ for custom geometry column name."""
    controls = Controls(valid_control_data, geometry_name="custom_name")
    assert isinstance(controls["custom_name"], gpd.GeoSeries)


def test_getitem_geometry_column_in_list(valid_control_data: pd.DataFrame) -> None:
    """Test __getitem__ with geometry column in a list."""
    controls = Controls(valid_control_data)
    assert isinstance(controls[["geometry", "sx", "sy", "sz"]], Controls)


def test_getitem_geometry_column_in_index(valid_control_data: pd.DataFrame) -> None:
    """Test __getitem__ with geometry column in an Index."""
    controls = Controls(valid_control_data)
    assert isinstance(controls[pd.Index(["geometry", "sx", "sy", "sz"])], Controls)


def test_getitem_multiple_columns(valid_control_data: pd.DataFrame) -> None:
    """Test __getitem__ with multiple columns."""
    controls = Controls(valid_control_data)
    assert isinstance(controls[["x", "y", "z"]], Controls)


def test_getitem_single_column_frame(valid_control_data: pd.DataFrame) -> None:
    """Test __getitem__ with a single column as a frame."""
    controls = Controls(valid_control_data)
    assert isinstance(controls[["x"]], Controls)


def test_getitem_single_column_series(valid_control_data: pd.DataFrame) -> None:
    """Test __getitem__ with a single column as a series."""
    controls = Controls(valid_control_data)
    assert isinstance(controls["x"], pd.Series)


def test_iterfeatures_with_coordinates(valid_control_data: pd.DataFrame) -> None:
    """Test iterfeatures with coordinates columns included."""
    controls = Controls(valid_control_data)

    for feature in controls.iterfeatures(include_coordinates_columns=True):
        assert "id" in feature
        assert all(key in feature["properties"] for key in ["x", "y", "z"])
        assert all(key in feature["properties"] for key in ["sx", "sy", "sz"])
        assert "geometry" in feature


def test_iterfeatures_without_coordinates(valid_control_data: pd.DataFrame) -> None:
    """Test iterfeatures without coordinates columns."""
    controls = Controls(valid_control_data)

    for feature in controls.iterfeatures(include_coordinates_columns=False):
        assert not all(key in feature["properties"] for key in ["x", "y", "z"])
        assert all(key in feature["properties"] for key in ["sx", "sy", "sz"])
        assert "geometry" in feature
        assert "id" in feature


def test_iterfeatures_without_id(valid_control_data: pd.DataFrame) -> None:
    """Test iterfeatures with drop_id=True."""
    controls = Controls(valid_control_data)

    for feature in controls.iterfeatures(
        include_coordinates_columns=False, drop_id=True
    ):
        assert not all(key in feature["properties"] for key in ["x", "y", "z"])
        assert all(key in feature["properties"] for key in ["sx", "sy", "sz"])
        assert "geometry" in feature
        assert "id" not in feature


def test_set_crs_on_init(valid_control_data: pd.DataFrame) -> None:
    """Test setting CRS on initialization."""
    controls = Controls(valid_control_data, crs="EPSG: 2180")
    assert controls.crs == "EPSG: 2180"


def test_set_crs_by_crs_inplace(valid_control_data: pd.DataFrame) -> None:
    """Test set_crs by CRS string inplace."""
    controls = Controls(valid_control_data)
    controls.set_crs(crs="EPSG: 2180", inplace=True)
    assert controls.crs == "EPSG: 2180"


def test_set_crs_by_epsg_inplace(valid_control_data: pd.DataFrame) -> None:
    """Test set_crs by EPSG code inplace."""
    controls = Controls(valid_control_data)
    controls.set_crs(epsg=2180, inplace=True)
    assert controls.crs == "EPSG: 2180"


def test_set_crs_by_crs_on_copy(valid_control_data: pd.DataFrame) -> None:
    """Test set_crs by CRS string returns a copy."""
    controls = Controls(valid_control_data)
    controls_copy = controls.set_crs(crs="EPSG: 2180")

    assert controls.crs is None
    assert controls_copy.crs == "EPSG: 2180"
    assert controls is not controls_copy


def test_set_crs_by_epsg_on_copy(valid_control_data: pd.DataFrame) -> None:
    """Test set_crs by EPSG code returns a copy."""
    controls = Controls(valid_control_data)
    controls_copy = controls.set_crs(epsg=2180)

    assert controls.crs is None
    assert controls_copy.crs == "EPSG: 2180"
    assert controls is not controls_copy


def test_set_crs_override_error(valid_control_data: pd.DataFrame) -> None:
    """Test set_crs override error."""
    controls = Controls(valid_control_data, crs="EPSG:2180")
    with pytest.raises(ValueError):
        controls.set_crs(epsg=2178, inplace=True)


def test_set_crs_override(valid_control_data: pd.DataFrame) -> None:
    """Test set_crs with allow_override=True."""
    controls = Controls(valid_control_data, crs="EPSG:2180")
    coords_2180 = controls.coordinates

    controls.set_crs(epsg=2178, inplace=True, allow_override=True)
    coords_2178 = controls.coordinates

    assert controls.crs == "EPSG: 2178"
    assert all(coords_2180 == coords_2178)


def test_to_crs_inplace(valid_control_data: pd.DataFrame) -> None:
    """Test to_crs inplace transformation."""
    controls = Controls(valid_control_data, crs="EPSG:2180")
    coords_2180 = controls.coordinates

    with pytest.warns(InvalidGeometryWarning):
        controls.to_crs(epsg=2178, inplace=True)
    coords_2178 = controls.coordinates

    assert controls.crs == "EPSG: 2178"
    assert all(coords_2180 != coords_2178)

    controls.to_crs(epsg=2180, inplace=True)
    coords_2180_after = controls.coordinates

    assert controls.crs == "EPSG: 2180"
    assert all(coords_2180 == coords_2180_after)


def test_to_crs_on_copy(valid_control_data: pd.DataFrame) -> None:
    """Test to_crs returns a copy with new CRS."""
    controls = Controls(valid_control_data, crs="EPSG:2180")
    coords_2180 = controls.coordinates

    with pytest.warns(InvalidGeometryWarning):
        controls_2178 = controls.to_crs(epsg=2178)
    coords_2178 = controls_2178.coordinates

    assert controls_2178.crs == "EPSG: 2178"
    assert all(coords_2180 != coords_2178)
    assert controls is not controls_2178

    controls_2180_after = controls_2178.to_crs(epsg=2180)
    coords_2180_after = controls_2180_after.coordinates

    assert controls_2180_after.crs == "EPSG: 2180"
    assert all(coords_2180 == coords_2180_after)
    assert controls_2178 is not coords_2180_after


def test_rename_geometry_inplace(valid_control_data: pd.DataFrame) -> None:
    """Test rename_geometry inplace."""
    controls = Controls(valid_control_data)
    controls.rename_geometry("new_geometry_name", inplace=True)

    assert controls.active_geometry_name == "new_geometry_name"


def test_rename_geometry_on_copy(valid_control_data: pd.DataFrame) -> None:
    """Test rename_geometry returns a copy."""
    controls = Controls(valid_control_data)
    controls_new_name = controls.rename_geometry("new_geometry_name")

    assert controls.active_geometry_name == "geometry"
    assert controls_new_name.active_geometry_name == "new_geometry_name"


def test_rename_geometry_invalid(valid_control_data: pd.DataFrame) -> None:
    """Test rename_geometry with invalid name raises ValueError."""
    controls = Controls(valid_control_data)
    with pytest.raises(ValueError):
        controls.rename_geometry("2D_points")


def test_rename_geometry_twice(valid_control_data: pd.DataFrame) -> None:
    """Test renaming geometry column twice deletes previous name."""
    controls = Controls(valid_control_data)
    controls.rename_geometry("new_geometry_name", inplace=True)
    controls.rename_geometry("another_geometry_name", inplace=True)

    assert controls.active_geometry_name == "another_geometry_name"
    assert not controls.active_geometry_name == "new_geometry_name"


def test_geometry_property(valid_control_data: pd.DataFrame) -> None:
    """Test geometry property with custom name and CRS."""
    controls = Controls(
        valid_control_data, geometry_name="custom_name", crs="EPSG: 2178"
    )
    geometry = controls.geometry

    assert isinstance(geometry, gpd.GeoSeries)
    assert geometry.crs == "EPSG: 2178"
    assert geometry.name == "custom_name"


def test_1D_geometry_property(control_data_1D: pd.DataFrame) -> None:
    """Test geometry property for 1D data."""
    controls = Controls(control_data_1D, geometry_name="data_1D", crs="EPSG: 2180")
    geometry = controls.geometry

    assert isinstance(geometry, gpd.GeoSeries)
    assert geometry.crs == "EPSG: 2180"
    assert geometry.name == "data_1D"
    assert not np.isfinite(geometry.x).all()
    assert not np.isfinite(geometry.y).all()


def test_geometry_setter(valid_control_data: pd.DataFrame) -> None:
    """Test geometry setter emits warning."""
    controls = Controls(valid_control_data)

    with pytest.warns(GeometryAssigningWarning):
        controls.geometry = "New_values"


def test_x_property(valid_control_data: pd.DataFrame) -> None:
    """Test x property returns a non-empty Series."""
    controls = Controls(valid_control_data)
    assert isinstance(controls.x, pd.Series)
    assert not controls.x.empty


def test_x_property_error(control_data_1D: pd.DataFrame) -> None:
    """Test x property raises error for 1D data."""
    controls = Controls(control_data_1D)
    with pytest.raises(DimensionError):
        controls.x


def test_y_property(valid_control_data: pd.DataFrame) -> None:
    """Test y property returns a non-empty Series."""
    controls = Controls(valid_control_data)
    assert isinstance(controls.y, pd.Series)
    assert not controls.y.empty


def test_y_property_error(control_data_1D: pd.DataFrame) -> None:
    """Test y property raises error for 1D data."""
    controls = Controls(control_data_1D)
    with pytest.raises(DimensionError):
        controls.y


def test_z_property(valid_control_data: pd.DataFrame) -> None:
    """Test z property returns a non-empty Series."""
    controls = Controls(valid_control_data)
    assert isinstance(controls.z, pd.Series)
    assert not controls.z.empty


def test_z_property_error(control_data_2D: pd.DataFrame) -> None:
    """Test z property raises error for 2D data."""
    controls = Controls(control_data_2D)
    with pytest.raises(DimensionError):
        controls.z
        print(controls.z)


def test_coordinate_columns(valid_control_data: pd.DataFrame) -> None:
    """Test coordinate_columns property."""
    controls = Controls(valid_control_data)
    coordinate_columns = {"x", "y", "z"}

    assert not controls.coordinate_columns.has_duplicates
    assert set(controls.coordinate_columns) == coordinate_columns


def test_sigma_columns(valid_control_data: pd.DataFrame) -> None:
    """Test sigma_columns property."""
    controls = Controls(valid_control_data)
    sigma_columns = {"sx", "sy", "sz"}

    assert not controls.sigma_columns.has_duplicates
    assert set(controls.sigma_columns) == sigma_columns


def test_coordinates_property(valid_control_data: pd.DataFrame) -> None:
    """Test coordinates property returns Controls with correct columns."""
    controls = Controls(valid_control_data)

    assert set(controls.coordinates.columns) == set(controls.coordinate_columns)
    assert isinstance(controls.coordinates, Controls)
    assert not controls.coordinates.empty


def test_coordinate_sigmas_property(valid_control_data: pd.DataFrame) -> None:
    """Test coordinate_sigmas property returns Controls with correct columns."""
    controls = Controls(valid_control_data)

    assert set(controls.coordinate_sigmas.columns) == set(controls.sigma_columns)
    assert isinstance(controls.coordinate_sigmas, Controls)
    assert not controls.coordinate_sigmas.empty


def test_copy(valid_control_data: pd.DataFrame) -> None:
    """Test copy method returns a new Controls object with same properties."""
    controls = Controls(valid_control_data, geometry_name="data_1D", crs="EPSG: 2180")
    controls_copy = controls.copy()

    assert controls is not controls_copy
    assert controls.crs == controls_copy.crs
    assert controls.active_geometry_name == controls_copy.active_geometry_name


def test_to_geodataframe(valid_control_data: pd.DataFrame) -> None:
    """Test to_geodataframe returns a GeoDataFrame with correct properties."""
    controls = Controls(valid_control_data, geometry_name="data_1D", crs="EPSG: 2180")
    gdf = controls.to_geodataframe()

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == controls.crs
    assert gdf.active_geometry_name == controls.active_geometry_name


def test_geopandas_buffer(valid_control_data: pd.DataFrame) -> None:
    """Test buffer method returns valid polygons."""
    controls = Controls(valid_control_data, geometry_name="data_1D", crs="EPSG: 2180")
    buffer = controls.buffer(10)

    assert all(buffer.geom_type == "Polygon")
    assert not buffer.empty
    assert all(buffer.is_valid)


def test_geopandas_cx(valid_control_data: pd.DataFrame) -> None:
    """Test cx indexer returns correct row."""
    controls = Controls(valid_control_data, geometry_name="data_1D", crs="EPSG: 2180")
    assert all(controls.cx[1500:2500, 1500:2500] == controls.loc["C2", :])
