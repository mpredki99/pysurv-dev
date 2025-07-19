# Coding: UTF-8

# Copyright (C) 2025 Michał Prędki
# Licensed under the GNU General Public License v3.0.
# Full text of the license can be found in the LICENSE and COPYING files in the repository.

"""
This module provides a modular and extensible framework for constructing the various matrices
required in adjustment computations, such as those used in least squares adjustment of
surveying cotrol network.

**Structure and Relationships:**

- **Base Constructors:**

  - MatrixConstructor - base abstract class that enforces to implementation `build` method;
  - InnerConstraintsConstructor - abstract class that extends base MatrixConstructor adding `_matrix_x_n_col` attribute;
  - MatrixXYWsWConstructor - abstract class that extends base MatrixConstructor adding `_default_sigmas` attribute;
  - MatrixXYWConstructor - abstract class that extends MatrixXYWsWConstructor adding `_initialize_xyw_matrices` and `_apply_observation_function` methods;
  - MatrixSWConstructor - abstract class that extends MatrixXYWsWConstructor adding `_initialize_sw_matrix` method;

- **Matrix Constructors:**
  - MatrixRConstructor - subclass of InnerConstraintsConstructor for constructing inner constraints R matrix;
  - MatrixSXConstructor - subclass of InnerConstraintsConstructor for constructing control point corrections sX matrix;

  - SpeedXYWConstructor - subclass of MatrixXYWConstructor that implements speed strategy vectorized approach for constructing X, Y and W matrices;
  - SpeedSWConstructor - subclass of MatrixSWConstructor that implements speed strategy vectorized approach for constructing sW matrix;
  - MemoryXYWConstructor - subclass of MatrixXYWConstructor that implements memory-safe strategy row-wise approach for constructing X, Y and W matrices;
  - MemorySWConstructor - subclass of MatrixSWConstructor that implements memory-safe strategy row-wise approach for constructing sW matrix;

- **Strategy Pattern:**

  - MatrixXYWsWStrategy - base abstract class that enforces implementation `xyw_constructor` and `sw_constructor` via composition;
  - SpeedStrategy - subclass of MatrixXYWsWStrategy that implements speed strategy vectorized approach for `xyw_constructor` and `sw_constructor`;
  - MemoryStrategy - subclass of MatrixXYWsWStrategy that implements memory-safe strategy row-wise approach for `xyw_constructor` and `sw_constructor`;

- **Factory Pattern:**

  - `_matrix_xyw_sw_strategy_factory.py` - module provides get_strategy function that returns strategy object based on given name or dataset size;

- **Adapters and Utilities:**

  - MatrixXIndexer - helper class that maps control point coordinates and station orientations to matrix X column indices;
  - `_obs_equations_adapter` - module provides `obs_eqations_adapter` dict with lambda functions that enables assigning observation
    equations coefficients into X and Y matrix;
"""
