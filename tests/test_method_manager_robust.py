# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

import pytest

from pysurv.adjustment import MethodManagerRobust
from pysurv.adjustment.matrices import Matrices
from pysurv.exceptions import InvalidMethodError


def test_init_obs_method() -> None:
    """Test init observation method and retreiving its default constant."""
    method_manager = MethodManagerRobust(observations="huber")
    obs_c = method_manager.obs_tuning_constants.get("c")

    assert method_manager.observations == "huber"
    assert obs_c == 1.345


def test_init_free_adj_method() -> None:
    """Test init free adjustment method and retreiving its default constants."""
    method_manager = MethodManagerRobust(free_adjustment="tukey")
    free_c = method_manager.free_adj_tuning_constants.get("c")
    free_n = method_manager.free_adj_tuning_constants.get("n")

    assert method_manager.free_adjustment == "tukey"
    assert free_c == 4.685
    assert free_n == 2.0


def test_ivalid_obs_method() -> None:
    """Test invalid observation method raises error."""
    with pytest.raises(InvalidMethodError):
        MethodManagerRobust(observations="invalid_method")


def test_ivalid_free_method() -> None:
    """Test invalid free adjustment method raises error."""
    with pytest.raises(InvalidMethodError):
        MethodManagerRobust(free_adjustment="invalid_method")


def test_custom_obs_tuning_constants() -> None:
    """Test init with user's observation tuning constants."""
    method_manager = MethodManagerRobust(
        observations="huber",
        obs_tuning_constants={"c": 2.5},
    )
    obs_c = method_manager.obs_tuning_constants.get("c")

    assert method_manager.observations == "huber"
    assert obs_c == 2.5


def test_custom_free_adj_tuning_constants() -> None:
    """Test init with user's free adjustment tuning constants."""
    method_manager = MethodManagerRobust(
        free_adjustment="tukey",
        free_adj_tuning_constants={"c": 5.0},
    )
    free_c = method_manager.free_adj_tuning_constants.get("c")
    free_n = method_manager.free_adj_tuning_constants.get("n")

    assert method_manager.free_adjustment == "tukey"
    assert free_c == 5.0
    assert free_n == 2.0


def test_from_simple_to_robust_obs_method() -> None:
    """Test switching from simple to robust observation method updates constants."""
    method_manager = MethodManagerRobust(
        observations="ordinary",
    )

    assert method_manager.observations == "ordinary"
    assert method_manager.obs_tuning_constants is None

    method_manager.observations = "jacobi"
    obs_c = method_manager.obs_tuning_constants.get("c")
    obs_n = method_manager.obs_tuning_constants.get("n")

    assert method_manager.observations == "jacobi"
    assert obs_c == 4.687
    assert obs_n == 1.0


def test_from_simple_to_robust_free_adj_method() -> None:
    """Test switching from simple to robust free adjustment method updates constants."""
    method_manager = MethodManagerRobust(
        free_adjustment="weighted",
    )

    assert method_manager.free_adjustment == "weighted"
    assert method_manager.free_adj_tuning_constants is None

    method_manager.free_adjustment = "hampel"
    free_a = method_manager.free_adj_tuning_constants.get("a")
    free_b = method_manager.free_adj_tuning_constants.get("b")
    free_c = method_manager.free_adj_tuning_constants.get("c")

    assert method_manager.free_adjustment == "hampel"
    assert free_a == 1.7
    assert free_b == 3.4
    assert free_c == 8.5


def test_from_robust_to_simple_obs_method() -> None:
    """Test switching from robust to simple observation method clears constants."""
    method_manager = MethodManagerRobust(
        observations="cauchy",
    )
    obs_c = method_manager.obs_tuning_constants.get("c")
    obs_n = method_manager.obs_tuning_constants.get("n")

    assert method_manager.observations == "cauchy"
    assert obs_c == 2.385
    assert obs_n == 2.0

    method_manager.observations = "weighted"

    assert method_manager.observations == "weighted"
    assert method_manager.obs_tuning_constants is None


def test_from_robust_to_simple_free_adj_method() -> None:
    """Test switching from robust to simple free adjustment method clears constants."""
    method_manager = MethodManagerRobust(
        free_adjustment="andrews",
    )
    free_c = method_manager.free_adj_tuning_constants.get("c")

    assert method_manager.free_adjustment == "andrews"
    assert free_c == 4.207

    method_manager.free_adjustment = None

    assert method_manager.free_adjustment is None
    assert method_manager.free_adj_tuning_constants is None


def test_update_obs_tuning_constants() -> None:
    """Test updating observation tuning constants will keep other parameter."""
    method_manager = MethodManagerRobust(
        observations="epanechnikov", obs_tuning_constants={"c": 1.0, "n": 3.0}
    )
    method_manager.obs_tuning_constants["c"] = 2.0

    obs_c = method_manager.obs_tuning_constants.get("c")
    obs_n = method_manager.obs_tuning_constants.get("n")

    assert obs_c == 2.0
    assert obs_n == 3.0


def test_update_free_adj_tuning_constants() -> None:
    """Test updating free adjustment tuning constants will keep other parameter."""
    method_manager = MethodManagerRobust(
        free_adjustment="epanechnikov", free_adj_tuning_constants={"c": 1.0, "n": 3.0}
    )
    method_manager.free_adj_tuning_constants["c"] = 2.0

    free_c = method_manager.free_adj_tuning_constants.get("c")
    free_n = method_manager.free_adj_tuning_constants.get("n")

    assert free_c == 2.0
    assert free_n == 3.0


def test_reassign_obs_tuning_constants() -> None:
    """Test reassigning observation tuning constants dictionary."""
    method_manager = MethodManagerRobust(
        observations="epanechnikov", obs_tuning_constants={"c": 1.0, "n": 3.0}
    )
    method_manager.obs_tuning_constants = {"c": 5.0, "n": 7.0}

    obs_c = method_manager.obs_tuning_constants.get("c")
    obs_n = method_manager.obs_tuning_constants.get("n")

    assert obs_c == 5.0
    assert obs_n == 7.0


def test_reassign_free_adj_tuning_constants() -> None:
    """Test reassigning free adjustment tuning constants dictionary."""
    method_manager = MethodManagerRobust(
        free_adjustment="epanechnikov", free_adj_tuning_constants={"c": 1.0, "n": 3.0}
    )
    method_manager.free_adj_tuning_constants = {"c": 5.0, "n": 7.0}

    free_c = method_manager.free_adj_tuning_constants.get("c")
    free_n = method_manager.free_adj_tuning_constants.get("n")

    assert free_c == 5.0
    assert free_n == 7.0


def test_obs_tuning_constants_invalid_key() -> None:
    """Test invalid key in observation tuning constants is ignored."""
    method_manager = MethodManagerRobust(
        observations="huber", obs_tuning_constants={"a": 1.0}
    )
    obs_c = method_manager.obs_tuning_constants.get("c")

    assert obs_c == 1.345
    assert "a" not in method_manager.obs_tuning_constants.keys()


def test_free_adj_tuning_constants_invalid_key() -> None:
    """Test invalid key in free adjustment tuning constants is ignored."""
    method_manager = MethodManagerRobust(
        free_adjustment="huber", free_adj_tuning_constants={"a": 1.0}
    )
    free_c = method_manager.free_adj_tuning_constants.get("c")

    assert free_c == 1.345
    assert "a" not in method_manager.free_adj_tuning_constants.keys()


def test_cra_method_tuning_constants_before_injection() -> None:
    """Test CRA method tuning constants before matrix injection is None."""
    method_manager = MethodManagerRobust(observations="cra")
    sigma_sq = method_manager.obs_tuning_constants.get("sigma_sq")

    assert sigma_sq is None
    assert "sigma_sq" in method_manager.obs_tuning_constants.keys()


def test_t_method_tuning_constants_before_injection() -> None:
    """Test t method tuning constants before matrix injection is None."""
    method_manager = MethodManagerRobust(observations="t")
    k = method_manager.obs_tuning_constants.get("k")

    assert k is None
    assert "k" in method_manager.obs_tuning_constants.keys()


def test_t_method_tuning_constants_after_injection(MatricesTester: Matrices) -> None:
    """Test t method tuning constants after matrix injection has value."""
    method_manager = MethodManagerRobust(observations="t")
    MatricesTester(methods=method_manager)
    k = method_manager.obs_tuning_constants.get("k")

    assert k == 20


def test_refresh_degress_of_freedom_after_injection(MatricesTester: Matrices) -> None:
    """Test refreshing degrees of freedom after matrix injection."""
    method_manager = MethodManagerRobust(observations="t")
    MatricesTester(methods=method_manager)

    obs_k = method_manager.obs_tuning_constants.get("k")

    assert obs_k == 20

    method_manager.free_adjustment = "t"

    obs_k = method_manager.obs_tuning_constants.get("k")
    free_k = method_manager.free_adj_tuning_constants.get("k")

    assert obs_k == 24
    assert free_k == 24
