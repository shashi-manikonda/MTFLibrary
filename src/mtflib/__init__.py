# mtflib/__init__.py
"""
mtflib: A Python Library for Multivariate Taylor Functions
===========================================================

This library provides a robust framework for working with multivariate
Taylor series expansions, based on the principles of Differential Algebra (DA).
It allows for the creation, manipulation, and analysis of functions
represented by their Taylor series, supporting both real and complex
coefficients.

Core Features:
- **MultivariateTaylorFunction**: The fundamental class for representing
  a function as a DA vector of its Taylor coefficients.
- **ComplexMultivariateTaylorFunction**: An extension for functions with
  complex coefficients.
- **TaylorMap**: Represents vector-valued functions (maps from R^n to R^m)
  for coordinate transformations and systems of equations.
- **Elementary Functions**: A rich set of elementary functions (sin, cos, exp,
  log, etc.) defined for Taylor series objects.
- **Differential Operators**: `derivative` and `integrate` functions that
  act as the derivation operators of the Differential Algebra.

Example:
    >>> from mtflib import MultivariateTaylorFunction, sin_taylor, Var
    >>>
    >>> # It is crucial to initialize the library's global settings first.
    >>> MultivariateTaylorFunction.initialize_mtf(max_order=4, max_dimension=2)
    >>>
    >>> # Create variables x and y
    >>> x = Var(1)
    >>> y = Var(2)
    >>>
    >>> # Create a function f(x, y) = sin(x + y)
    >>> f = sin_taylor(x + y)
    >>>
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (0, 1)
    1  1.000000e+00      1    (1, 0)
    2 -1.666667e-01      3    (0, 3)
    3 -5.000000e-01      3    (1, 2)
    4 -5.000000e-01      3    (2, 1)
    5 -1.666667e-01      3    (3, 0)

"""

import pandas as pd

from .taylor_function import (
    MultivariateTaylorFunction,
    convert_to_mtf,
    isqrt_taylor,
    Var,
    mtfarray,
)
from .elementary_functions import (
    cos_taylor,
    sin_taylor,
    tan_taylor,
    exp_taylor,
    gaussian_taylor,
    log_taylor,
    arctan_taylor,
    sinh_taylor,
    cosh_taylor,
    tanh_taylor,
    arcsin_taylor,
    arccos_taylor,
    arctanh_taylor,
    integrate,
    derivative,
    sqrt_taylor,
)
from .complex_taylor_function import (
    ComplexMultivariateTaylorFunction,
    convert_to_cmtf,
)
from .elementary_coefficients import load_precomputed_coefficients
from .taylor_map import TaylorMap

# Set the display format for floats
pd.options.display.float_format = "{:.12e}".format
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)


MTF = MultivariateTaylorFunction


# Defines the public API for the library.
__all__ = [
    # Core Classes
    "MultivariateTaylorFunction",
    "ComplexMultivariateTaylorFunction",
    "TaylorMap",
    "MTF",  # Alias for MultivariateTaylorFunction
    "Var",  # Factory for creating variables
    # Elementary Functions
    "sin_taylor",
    "cos_taylor",
    "tan_taylor",
    "arcsin_taylor",
    "arccos_taylor",
    "arctan_taylor",
    "sinh_taylor",
    "cosh_taylor",
    "tanh_taylor",
    "arctanh_taylor",
    "exp_taylor",
    "log_taylor",
    "sqrt_taylor",
    "isqrt_taylor",
    "gaussian_taylor",
    # Core Operators
    "integrate",
    "derivative",
    # Utility Functions
    "mtfarray",
    "load_precomputed_coefficients",
    "convert_to_mtf",
    "convert_to_cmtf",
]
