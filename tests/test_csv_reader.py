import numpy as np
import pytest

from pysurv.exceptions._exceptions import (
    EmptyDatasetError,
    InvalidDataError,
    MissingMandatoryColumnsError,
)
from pysurv.reader.csv_reader import CSVReader


class TestCSVReader:

    def test_mandatory_init_arguments(self):
        with pytest.raises(TypeError):
            CSVReader()

    def test_invalid_measurement_path(self, valid_controls_file):
        with pytest.raises(FileNotFoundError) as e:
            CSVReader("invalid/path/measurements.csv", valid_controls_file)
        assert "Measurements file not found:" in str(e.value)

    def test_invalid_controls_path(self, valid_measurements_file):
        with pytest.raises(FileNotFoundError) as e:
            reader = CSVReader(valid_measurements_file, "invalid/path/controls.csv")
        assert "Controls file not found:" in str(e.value)

    def test_raise_validation_mode(self, valid_measurements_file, valid_controls_file):
        reader = CSVReader(valid_measurements_file, valid_controls_file)
        assert reader._validation_mode == "raise"

    def test_skip_validation_mode(self, valid_measurements_file, valid_controls_file):
        reader = CSVReader(
            valid_measurements_file, valid_controls_file, validation_mode="skip"
        )
        assert reader._validation_mode == "skip"

    def test_none_validation_mode(self, valid_measurements_file, valid_controls_file):
        reader = CSVReader(
            valid_measurements_file, valid_controls_file, validation_mode=None
        )
        assert reader._validation_mode is None

    def test_invalid_validation_mode(
        self, valid_measurements_file, valid_controls_file
    ):
        with pytest.raises(ValueError):
            CSVReader(
                valid_measurements_file,
                valid_controls_file,
                validation_mode="invalid_mode",
            )

    def test_measurements_columns_name_standarization(
        self, measurements_file_columns_to_rename, valid_controls_file
    ):
        reader = CSVReader(measurements_file_columns_to_rename, valid_controls_file)
        reader.read_measurements()

        assert "stn_pk" in reader.stations.columns
        assert "STN_PK" not in reader.stations.columns
        assert "stn_id" in reader.stations.columns
        assert "STN_ID" not in reader.stations.columns
        assert "stn_h" in reader.stations.columns
        assert "STN_H" not in reader.stations.columns
        assert "stn_sh" in reader.stations.columns
        assert "STN_SH" not in reader.stations.columns

        assert "stn_pk" in reader.measurements.columns
        assert "STN_PK" not in reader.measurements.columns
        assert "trg_id" in reader.measurements.columns
        assert "TRG_ID" not in reader.measurements.columns
        assert "trg_h" in reader.measurements.columns
        assert "TRG_H" not in reader.measurements.columns
        assert "trg_sh" in reader.measurements.columns
        assert "TRG_SH" not in reader.measurements.columns
        assert "sd" in reader.measurements.columns
        assert "SD" not in reader.measurements.columns
        assert "ssd" in reader.measurements.columns
        assert "SSD" not in reader.measurements.columns
        assert "hd" in reader.measurements.columns
        assert "HD" not in reader.measurements.columns
        assert "shd" in reader.measurements.columns
        assert "SHD" not in reader.measurements.columns
        assert "vd" in reader.measurements.columns
        assert "VD" not in reader.measurements.columns
        assert "svd" in reader.measurements.columns
        assert "SVD" not in reader.measurements.columns
        assert "dx" in reader.measurements.columns
        assert "DX" not in reader.measurements.columns
        assert "sdx" in reader.measurements.columns
        assert "SDX" not in reader.measurements.columns
        assert "dy" in reader.measurements.columns
        assert "DY" not in reader.measurements.columns
        assert "sdy" in reader.measurements.columns
        assert "SDY" not in reader.measurements.columns
        assert "dz" in reader.measurements.columns
        assert "DZ" not in reader.measurements.columns
        assert "sdz" in reader.measurements.columns
        assert "SDZ" not in reader.measurements.columns
        assert "a" in reader.measurements.columns
        assert "A" not in reader.measurements.columns
        assert "sa" in reader.measurements.columns
        assert "SA" not in reader.measurements.columns
        assert "hz" in reader.measurements.columns
        assert "HZ" not in reader.measurements.columns
        assert "shz" in reader.measurements.columns
        assert "SHZ" not in reader.measurements.columns
        assert "vz" in reader.measurements.columns
        assert "VZ" not in reader.measurements.columns
        assert "svz" in reader.measurements.columns
        assert "SVZ" not in reader.measurements.columns
        assert "vh" in reader.measurements.columns
        assert "VH" not in reader.measurements.columns
        assert "svh" in reader.measurements.columns
        assert "SVH" not in reader.measurements.columns

    def test_controls_columns_name_standarization_e_n_el(
        self, valid_measurements_file, controls_file_column_to_rename_e_n_el
    ):
        reader = CSVReader(
            valid_measurements_file, controls_file_column_to_rename_e_n_el
        )
        reader.read_controls()

        assert "id" in reader.controls.columns
        assert "NR" not in reader.controls.columns
        assert "x" in reader.controls.columns
        assert "E" not in reader.controls.columns
        assert "y" in reader.controls.columns
        assert "N" not in reader.controls.columns
        assert "z" in reader.controls.columns
        assert "EL" not in reader.controls.columns
        assert "sx" in reader.controls.columns
        assert "SE" not in reader.controls.columns
        assert "sy" in reader.controls.columns
        assert "SN" not in reader.controls.columns
        assert "sz" in reader.controls.columns
        assert "SEL" not in reader.controls.columns

    def test_controls_columns_name_standarization_easting_northing_height(
        self,
        valid_measurements_file,
        controls_file_column_to_rename_easting_northing_height,
    ):
        reader = CSVReader(
            valid_measurements_file,
            controls_file_column_to_rename_easting_northing_height,
        )
        reader.read_controls()

        assert "id" in reader.controls.columns
        assert "NR" not in reader.controls.columns
        assert "x" in reader.controls.columns
        assert "EASTING" not in reader.controls.columns
        assert "y" in reader.controls.columns
        assert "NORTHING" not in reader.controls.columns
        assert "z" in reader.controls.columns
        assert "HEIGHT" not in reader.controls.columns
        assert "sx" in reader.controls.columns
        assert "SE" not in reader.controls.columns
        assert "sy" in reader.controls.columns
        assert "SN" not in reader.controls.columns
        assert "sz" in reader.controls.columns
        assert "SEL" not in reader.controls.columns

    def test_measurements_file_filtering(
        self, measurements_file_to_filter, valid_controls_file
    ):
        reader = CSVReader(measurements_file_to_filter, valid_controls_file)
        reader.read_measurements()

        assert "UNNECESSARY COLUMN" not in reader.measurements.columns
        assert "EXTRA COLUMN" not in reader.measurements.columns

    def test_controls_file_filtering(
        self, valid_measurements_file, controls_file_to_filter
    ):
        reader = CSVReader(valid_measurements_file, controls_file_to_filter)
        reader.read_controls()

        assert "UNNECESSARY COLUMN" not in reader.controls.columns
        assert "EXTRA COLUMN" not in reader.controls.columns

    def test_measurements_missing_mandatory_columns(
        self, measurements_file_missing_mandatory_columns, valid_controls_file
    ):
        reader = CSVReader(
            measurements_file_missing_mandatory_columns, valid_controls_file
        )
        with pytest.raises(MissingMandatoryColumnsError):
            reader.read_measurements()

    def test_controls_missing_mandatory_columns(
        self, valid_measurements_file, controls_file_missing_mandatory_columns
    ):
        reader = CSVReader(
            valid_measurements_file, controls_file_missing_mandatory_columns
        )
        with pytest.raises(MissingMandatoryColumnsError):
            reader.read_controls()

    def test_measurements_data_validation_none(
        self, invalid_measurements_file, valid_controls_file
    ):
        reader = CSVReader(
            invalid_measurements_file, valid_controls_file, validation_mode=None
        )
        with pytest.raises(ValueError):
            reader.read_measurements()

        trg_h = reader.measurements.columns.get_loc("trg_h")
        trg_sh = reader.measurements.columns.get_loc("trg_sh")
        sd = reader.measurements.columns.get_loc("sd")
        ssd = reader.measurements.columns.get_loc("ssd")
        hd = reader.measurements.columns.get_loc("hd")
        shd = reader.measurements.columns.get_loc("shd")
        vd = reader.measurements.columns.get_loc("vd")
        svd = reader.measurements.columns.get_loc("svd")
        dx = reader.measurements.columns.get_loc("dx")
        sdx = reader.measurements.columns.get_loc("sdx")
        dy = reader.measurements.columns.get_loc("dy")
        sdy = reader.measurements.columns.get_loc("sdy")
        dz = reader.measurements.columns.get_loc("dz")
        sdz = reader.measurements.columns.get_loc("sdz")
        a = reader.measurements.columns.get_loc("a")
        sa = reader.measurements.columns.get_loc("sa")
        hz = reader.measurements.columns.get_loc("hz")
        shz = reader.measurements.columns.get_loc("shz")
        vz = reader.measurements.columns.get_loc("vz")
        svz = reader.measurements.columns.get_loc("svz")
        vh = reader.measurements.columns.get_loc("vh")
        svh = reader.measurements.columns.get_loc("svh")

        assert not reader.measurements.empty

        assert reader.measurements.iloc[1, trg_h] == "Invalid type"
        assert reader.measurements.iloc[2, trg_sh] == "Invalid type"
        assert reader.measurements.iloc[4, trg_sh] == "-0.01"
        assert reader.measurements.iloc[2, sd] == "Invalid type"
        assert reader.measurements.iloc[3, sd] == "-100.0"
        assert reader.measurements.iloc[1, ssd] == "-0.01"
        assert reader.measurements.iloc[4, ssd] == "Invalid type"
        assert reader.measurements.iloc[1, hd] == "-100.0"
        assert reader.measurements.iloc[2, hd] == "Invalid type"
        assert reader.measurements.iloc[2, shd] == "Invalid type"
        assert reader.measurements.iloc[3, shd] == "-0.01"
        assert reader.measurements.iloc[3, vd] == "Invalid type"
        assert reader.measurements.iloc[1, svd] == "Invalid type"
        assert reader.measurements.iloc[2, svd] == "-0.01"
        assert reader.measurements.iloc[3, dx] == "Invalid type"
        assert reader.measurements.iloc[0, sdx] == "Invalid type"
        assert reader.measurements.iloc[1, sdx] == "-0.008"
        assert reader.measurements.iloc[1, dy] == "Invalid type"
        assert reader.measurements.iloc[0, sdy] == "-0.01"
        assert reader.measurements.iloc[2, sdy] == "Invalid type"
        assert reader.measurements.iloc[3, dz] == "Invalid type"
        assert reader.measurements.iloc[1, sdz] == "-0.001"
        assert reader.measurements.iloc[2, sdz] == "Invalid type"
        assert reader.measurements.iloc[1, a] == "Invalid type"
        assert reader.measurements.iloc[2, sa] == "-0.01"
        assert reader.measurements.iloc[3, sa] == "Invalid type"
        assert reader.measurements.iloc[0, hz] == "Invalid type"
        assert reader.measurements.iloc[2, shz] == "-0.002"
        assert reader.measurements.iloc[3, shz] == "Invalid type"
        assert reader.measurements.iloc[1, vz] == "Invalid type"
        assert reader.measurements.iloc[0, svz] == "-0.01"
        assert reader.measurements.iloc[3, svz] == "Invalid type"
        assert reader.measurements.iloc[4, vh] == "Invalid type"
        assert reader.measurements.iloc[1, svh] == "Invalid type"
        assert reader.measurements.iloc[2, svh] == "-0.1"

    def test_measurements_data_validation_skip(
        self, invalid_measurements_file, valid_controls_file
    ):
        reader = CSVReader(
            invalid_measurements_file, valid_controls_file, validation_mode="skip"
        )
        with pytest.warns():
            reader.read_measurements()

        trg_h = reader.measurements.columns.get_loc("trg_h")
        trg_sh = reader.measurements.columns.get_loc("trg_sh")
        sd = reader.measurements.columns.get_loc("sd")
        ssd = reader.measurements.columns.get_loc("ssd")
        hd = reader.measurements.columns.get_loc("hd")
        shd = reader.measurements.columns.get_loc("shd")
        vd = reader.measurements.columns.get_loc("vd")
        svd = reader.measurements.columns.get_loc("svd")
        dx = reader.measurements.columns.get_loc("dx")
        sdx = reader.measurements.columns.get_loc("sdx")
        dy = reader.measurements.columns.get_loc("dy")
        sdy = reader.measurements.columns.get_loc("sdy")
        dz = reader.measurements.columns.get_loc("dz")
        sdz = reader.measurements.columns.get_loc("sdz")
        a = reader.measurements.columns.get_loc("a")
        sa = reader.measurements.columns.get_loc("sa")
        hz = reader.measurements.columns.get_loc("hz")
        shz = reader.measurements.columns.get_loc("shz")
        vz = reader.measurements.columns.get_loc("vz")
        svz = reader.measurements.columns.get_loc("svz")
        vh = reader.measurements.columns.get_loc("vh")
        svh = reader.measurements.columns.get_loc("svh")

        assert not reader.measurements.empty

        assert np.isnan(reader.measurements.iloc[1, trg_h])
        assert np.isnan(reader.measurements.iloc[2, trg_sh])
        assert np.isnan(reader.measurements.iloc[4, trg_sh])
        assert np.isnan(reader.measurements.iloc[2, sd])
        assert np.isnan(reader.measurements.iloc[3, sd])
        assert np.isnan(reader.measurements.iloc[1, ssd])
        assert np.isnan(reader.measurements.iloc[4, ssd])
        assert np.isnan(reader.measurements.iloc[1, hd])
        assert np.isnan(reader.measurements.iloc[2, hd])
        assert np.isnan(reader.measurements.iloc[2, shd])
        assert np.isnan(reader.measurements.iloc[3, shd])
        assert np.isnan(reader.measurements.iloc[3, vd])
        assert np.isnan(reader.measurements.iloc[1, svd])
        assert np.isnan(reader.measurements.iloc[2, svd])
        assert np.isnan(reader.measurements.iloc[3, dx])
        assert np.isnan(reader.measurements.iloc[0, sdx])
        assert np.isnan(reader.measurements.iloc[1, sdx])
        assert np.isnan(reader.measurements.iloc[1, dy])
        assert np.isnan(reader.measurements.iloc[0, sdy])
        assert np.isnan(reader.measurements.iloc[2, sdy])
        assert np.isnan(reader.measurements.iloc[3, dz])
        assert np.isnan(reader.measurements.iloc[1, sdz])
        assert np.isnan(reader.measurements.iloc[2, sdz])
        assert np.isnan(reader.measurements.iloc[1, a])
        assert np.isnan(reader.measurements.iloc[2, sa])
        assert np.isnan(reader.measurements.iloc[3, sa])
        assert np.isnan(reader.measurements.iloc[0, hz])
        assert np.isnan(reader.measurements.iloc[2, shz])
        assert np.isnan(reader.measurements.iloc[3, shz])
        assert np.isnan(reader.measurements.iloc[1, vz])
        assert np.isnan(reader.measurements.iloc[0, svz])
        assert np.isnan(reader.measurements.iloc[3, svz])
        assert np.isnan(reader.measurements.iloc[4, vh])
        assert np.isnan(reader.measurements.iloc[1, svh])
        assert np.isnan(reader.measurements.iloc[2, svh])

    def test_measurements_data_validation_raise(
        self, invalid_measurements_file, valid_controls_file
    ):
        reader = CSVReader(invalid_measurements_file, valid_controls_file)
        with pytest.raises(InvalidDataError):
            reader.read_measurements()

    def test_controls_data_validation_none(
        self, valid_measurements_file, invalid_controls_file
    ):
        reader = CSVReader(
            valid_measurements_file, invalid_controls_file, validation_mode=None
        )
        with pytest.raises(ValueError):
            reader.read_controls()

        x = reader.controls.columns.get_loc("x")
        y = reader.controls.columns.get_loc("y")
        z = reader.controls.columns.get_loc("z")
        sx = reader.controls.columns.get_loc("sx")
        sy = reader.controls.columns.get_loc("sy")
        sz = reader.controls.columns.get_loc("sz")

        assert not reader.controls.empty

        assert reader.controls.iloc[0, x] == "Invalid type"
        assert reader.controls.iloc[1, y] == "Invalid type"
        assert reader.controls.iloc[2, z] == "Invalid type"
        assert reader.controls.iloc[0, sx] == "Invalid type"
        assert reader.controls.iloc[2, sx] == "-0.01"
        assert reader.controls.iloc[2, sy] == "Invalid type"
        assert reader.controls.iloc[3, sy] == "-0.015"
        assert reader.controls.iloc[0, sz] == "Invalid type"
        assert reader.controls.iloc[2, sz] == "-0.01"

    def test_controls_data_validation_skip(
        self, valid_measurements_file, invalid_controls_file
    ):
        reader = CSVReader(
            valid_measurements_file, invalid_controls_file, validation_mode="skip"
        )
        with pytest.warns():
            reader.read_controls()

        x = reader.controls.columns.get_loc("x")
        y = reader.controls.columns.get_loc("y")
        z = reader.controls.columns.get_loc("z")
        sx = reader.controls.columns.get_loc("sx")
        sy = reader.controls.columns.get_loc("sy")
        sz = reader.controls.columns.get_loc("sz")

        assert not reader.controls.empty

        assert np.isnan(reader.controls.iloc[0, x])
        assert np.isnan(reader.controls.iloc[1, y])
        assert np.isnan(reader.controls.iloc[2, z])
        assert np.isnan(reader.controls.iloc[0, sx])
        assert np.isnan(reader.controls.iloc[2, sx])
        assert np.isnan(reader.controls.iloc[2, sy])
        assert np.isnan(reader.controls.iloc[3, sy])
        assert np.isnan(reader.controls.iloc[0, sz])
        assert np.isnan(reader.controls.iloc[2, sz])

    def test_controls_data_validation_raise(
        self, valid_measurements_file, invalid_controls_file
    ):
        reader = CSVReader(valid_measurements_file, invalid_controls_file)
        with pytest.raises(InvalidDataError):
            reader.read_controls()

    def test_measurements_to_float(self, valid_measurements_file, valid_controls_file):
        reader = CSVReader(valid_measurements_file, valid_controls_file)
        reader.read_measurements()

        for col in reader.measurements.columns:
            if col in ["stn_pk", "trg_id"]:
                continue
            assert reader.measurements[col].dtype == float

    def test_controls_to_float(self, valid_measurements_file, valid_controls_file):
        reader = CSVReader(valid_measurements_file, valid_controls_file)
        reader.read_controls()

        for col in reader.controls.columns:
            if col == "id":
                continue
            assert reader.controls[col].dtype == float

    def test_import_empty_measurements_file_raise(
        self, empty_file, valid_controls_file
    ):
        reader = CSVReader(empty_file, valid_controls_file)
        with pytest.raises(EmptyDatasetError):
            reader.read_measurements()

    def test_import_empty_measurements_file_skip(self, empty_file, valid_controls_file):
        reader = CSVReader(empty_file, valid_controls_file, validation_mode="skip")
        with pytest.raises(EmptyDatasetError):
            reader.read_measurements()

    def test_import_empty_measurements_file_none(self, empty_file, valid_controls_file):
        reader = CSVReader(empty_file, valid_controls_file, validation_mode=None)
        reader.read_measurements()
        assert reader.measurements.empty

    def test_import_empty_controls_file_raise(
        self, valid_measurements_file, empty_file
    ):
        reader = CSVReader(valid_measurements_file, empty_file)
        with pytest.raises(EmptyDatasetError):
            reader.read_controls()

    def test_import_empty_controls_file_skip(self, valid_measurements_file, empty_file):
        reader = CSVReader(valid_measurements_file, empty_file, validation_mode="skip")
        with pytest.raises(EmptyDatasetError):
            reader.read_controls()

    def test_import_empty_controls_none(self, valid_measurements_file, empty_file):
        reader = CSVReader(valid_measurements_file, empty_file, validation_mode=None)
        reader.read_controls()
        assert reader.controls.empty
