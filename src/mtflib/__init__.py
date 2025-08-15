# mtflib/__init__.py

"""
A library for working with Multivariate Taylor Functions.

This package provides tools for creating, manipulating, and analyzing
Multivariate Taylor Series expansions. It includes classes for both
real and complex-valued functions, elementary function implementations,
and extended functionalities.
"""

import pandas as pd

# Set the display format for floats
pd.options.display.float_format = '{:.12e}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

from . import elementary_coefficients
from . import taylor_function
from . import elementary_functions
from . import MTFExtended
from . import complex_taylor_function

from .taylor_function import (initialize_mtf_globals, get_global_max_order,
    get_global_max_dimension, set_global_max_order, set_global_etol,
    get_global_etol,
    MultivariateTaylorFunctionBase,
    convert_to_mtf,
    get_mtf_initialized_status,
    _split_constant_polynomial_part,
    sqrt_taylor,
    isqrt_taylor,
    get_precomputed_coefficients) # Added remaining names from taylor_function
from .elementary_functions import (cos_taylor, sin_taylor, tan_taylor,
    exp_taylor, gaussian_taylor, log_taylor, arctan_taylor,
    sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor,
    arctanh_taylor, integrate, derivative)
from .complex_taylor_function import (ComplexMultivariateTaylorFunction,
    convert_to_cmtf)
from .MTFExtended import Var, MultivariateTaylorFunction, compose, MTF, mtfarray
from .elementary_coefficients import load_precomputed_coefficients
from .taylor_map import TaylorMap



'''
Purpose of __all__: The __all__ list defines what names are considered public
when a user does from mtflib import *. It's crucial for controlling the
public API of your library.
'''
__all__ = [
    'taylor_function',
    'load_precomputed_coefficients',
    'get_precomputed_coefficients',
    'elementary_functions',
    'MultivariateTaylorFunctionBase',
    'MultivariateTaylorFunction',
    'initialize_mtf_globals',
    'set_global_etol',
    'get_global_etol', # Added get_global_etol - although already present, ensuring
    'get_global_max_order',
    'get_global_max_dimension',
    'set_global_max_order', # Added set_global_max_order
    'Var',
    'mtfarray',
    'compose',
    'MTF',
    'convert_to_mtf',
    'get_mtf_initialized_status', # Added get_mtf_initialized_status - although already present, ensuring
    'integrate',
    'derivative',
    'cos_taylor',
    'sin_taylor',
    'tan_taylor',
    'exp_taylor',
    'gaussian_taylor',
    'sqrt_taylor',
    'isqrt_taylor',
    'log_taylor',
    'arctan_taylor',
    'sinh_taylor',
    'cosh_taylor',
    'tanh_taylor',
    'arcsin_taylor',
    'arccos_taylor',
    'arctanh_taylor',
    'ComplexMultivariateTaylorFunction',
    'convert_to_cmtf',
    'TaylorMap'
]
