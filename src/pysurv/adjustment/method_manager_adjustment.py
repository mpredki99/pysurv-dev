# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

from abc import ABC, abstractmethod

from pysurv.validators._validators import validate_method


class MethodManagerAdjustment(ABC):
    """
    Abstract base class for managing adjustment methods and tuning constants in least squares adjustment.

    This class provides a features for validating and setting the adjustment method for observation
    and inner constraint reweighting and managing a tuning constants values for robust or weighted
    adjustment methods.
    """

    def __init__(
        self,
        observations: str = "weighted",
        obs_tuning_constants: dict | None = None,
        free_adjustment: str | None = None,
        free_adj_tuning_constants: dict | None = None,
    ) -> None:
        # Matrices object - will be injected during matrices initializaton
        self._matrices = None

        self._observations = validate_method(observations)
        self._obs_tuning_constants = self._get_tuning_constants(
            obs_tuning_constants, self._observations
        )
        self._free_adjustment = self._get_free_adjustment(free_adjustment)
        self._free_adj_tuning_constants = self._get_tuning_constants(
            free_adj_tuning_constants, self._free_adjustment
        )

    @property
    def observations(self) -> str:
        """Return method used for observation weight matrix reweight."""
        return self._observations

    @observations.setter
    def observations(self, value: str) -> None:
        """Set method used for observation weight matrix reweight."""
        self._observations = validate_method(value)
        self._refresh_obs_tuning_constants()

    @property
    def obs_tuning_constants(self) -> dict:
        """Return tuning constants used for observation weight matrix reweight."""
        return self._obs_tuning_constants

    @obs_tuning_constants.setter
    def obs_tuning_constants(self, value: dict | None) -> None:
        """Set tuning constants used for observation weight matrix reweight."""
        self._obs_tuning_constants = self._get_tuning_constants(
            value, self._observations
        )

    @property
    def free_adjustment(self) -> str:
        """Return method used for inner constraint matrix reweight."""
        return self._free_adjustment

    @free_adjustment.setter
    def free_adjustment(self, value: str | None) -> None:
        """Set method used for inner constraint matrix reweight."""
        self._free_adjustment = self._get_free_adjustment(value)
        self._refresh_degrees_of_freedom()
        self._refresh_tuning_constants()

    @property
    def free_adj_tuning_constants(self) -> dict:
        """Return tuning constants used for inner constraint matrix reweight."""
        return self._free_adj_tuning_constants

    @free_adj_tuning_constants.setter
    def free_adj_tuning_constants(self, value: dict | None) -> None:
        """Return tuning constants used for inner constraint matrix reweight."""
        self._free_adj_tuning_constants = self._get_tuning_constants(
            value, self._free_adjustment
        )

    def _get_free_adjustment(self, free_adjustment: str | None) -> str | None:
        """Returns validated free adjustment method or None."""
        if free_adjustment is None:
            return None
        else:
            return validate_method(free_adjustment)

    def _refresh_degrees_of_freedom(self):
        """Refresh degrees of freedom on matrices object if injected."""
        if self._matrices is not None:
            self._matrices._refresh_degrees_of_freedom()

    def _refresh_tuning_constants(self):
        """Set new values of obs tuinig constants and free tuning constants."""
        self._refresh_obs_tuning_constants()
        self._refresh_free_tuning_constants()

    def _refresh_obs_tuning_constants(self):
        """Set new values of obs tuning constants."""
        self._obs_tuning_constants = self._get_tuning_constants(
            self._obs_tuning_constants, self._observations
        )

    def _refresh_free_tuning_constants(self):
        """Set new values of free tuning constants."""
        self._free_adj_tuning_constants = self._get_tuning_constants(
            self._free_adj_tuning_constants, self._free_adjustment
        )

    def _inject_matrices(self, matrices):
        """Inject parent matrices object."""
        self._matrices = matrices
        self._refresh_tuning_constants()

    @abstractmethod
    def _get_tuning_constants(
        self, tuning_constants: dict | None, method: str | None
    ) -> dict | None:
        """Determine and return tuning constants used for updating weight matrices."""
        pass
