from inspect import signature, unwrap

from . import robust
from .adjustment_method_manager import AdjustmentMethodManager


class MethodManager(AdjustmentMethodManager):
    def _get_tuning_constants(
        self, tuning_constants: dict | None, method: str | None, type: str
    ) -> dict | None:
        """Determine and return tuning constants used for updating weight matrices."""
        if not self.is_robust(method):
            return None

        tuning_constants = self._get_default_tuning_constants(tuning_constants, method)
        return self._add_method_specific_params(tuning_constants, method, type)

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
        wrapped_func = unwrap(func)
        sig = signature(wrapped_func)
        return {
            key: value.default
            for key, value in sig.parameters.items()
            if isinstance(value.default, (int, float))
        }

    def _add_method_specific_params(
        self, constants: dict, method: str, type: str
    ) -> dict:
        """Apply method-specific constant overrides using strategy pattern."""
        methods = {
            "cra": self._get_cra_param,
            "t": self._get_t_param,
        }
        update_func = methods.get(method)
        if update_func:
            constants.update(update_func(type))

        return constants

    def _get_cra_param(self, type: str) -> dict:
        """Get cra method specific param."""
        if self._solver is not None:
            return {"res_var": self._get_solver_res_var(type)}
        return {"res_var": None}

    def _get_solver_res_var(self, type: str):
        res_var = {
            "obs": self._solver.residual_variance,
            "free": self._solver.coord_cor_variance,
        }
        return res_var.get(type)

    def _get_t_param(self, type: str) -> dict:
        """Get t method specific param."""
        if self._solver is not None:
            return {"k": self._get_solver_k(type)}
        if self._matrices is not None:
            return {"k": self._matrices.degrees_of_freedom}
        return {"k": None}

    def _get_solver_k(self, type: str):
        k = {
            "obs": self._matrices.degrees_of_freedom,
            "free": self._solver.n_movable_tie_points,
        }
        return k.get(type)
