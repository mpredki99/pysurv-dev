from inspect import signature

from . import robust
from .method_manager_adjustment import AdjustmentMethodManager


class RobustMethodManager(AdjustmentMethodManager):
    def _get_tuning_constants(
        self, tuning_constants: dict | None, method: str | None
    ) -> dict | None:
        """Determine and return tuning constants used for updating weight matrices."""
        if self._no_tuning_constant_required(method):
            return None

        tuning_constants = self._get_default_tuning_constants(tuning_constants, method)
        return self._add_method_specific_params(tuning_constants, method)

    def _no_tuning_constant_required(self, method: str | None) -> bool:
        """Check if method requires no tuning constants."""
        return method in {None, "ordinary", "weighted"}

    def _get_default_tuning_constants(
        self, tuning_constants: dict | None, method: str
    ) -> dict:
        """Build the complete constants dictionary with defaults and user overrides."""
        default_constants = self._get_func_kwargs(method)

        if not tuning_constants:
            return default_constants

        return {
            key: tuning_constants.get(key, default_constants[key])
            for key in default_constants.keys()
        }

    def _get_func_kwargs(self, method: str) -> dict:
        """Get kwargs with default values of robust method used for updating weights."""
        func = getattr(robust, method)
        sig = signature(func)
        return {
            key: value.default
            for key, value in sig.parameters.items()
            if isinstance(value.default, (int, float))
        }

    def _add_method_specific_params(self, constants: dict, method: str) -> dict:
        """Apply method-specific constant overrides using strategy pattern."""
        methods = {
            "cra": self._get_cra_param,
            "t": self._get_t_param,
        }
        update_func = methods.get(method)
        if update_func:
            constants.update(update_func())

        return constants

    def _get_cra_param(self) -> dict:
        """Get cra method specific param."""
        return {"sigma_sq": None}

    def _get_t_param(self) -> dict:
        """Get t method specific param."""
        if self._matrices is not None:
            return {"k": self._matrices.degrees_of_freedom}
        return {"k": None}
