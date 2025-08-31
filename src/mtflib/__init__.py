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
from . import complex_taylor_function

from .taylor_function import (MultivariateTaylorFunction,
    convert_to_mtf,
    _split_constant_polynomial_part,
    sqrt_taylor,
    isqrt_taylor,
    Var,
    mtfarray)
from .elementary_functions import (cos_taylor, sin_taylor, tan_taylor,
    exp_taylor, gaussian_taylor, log_taylor, arctan_taylor,
    sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor,
    arctanh_taylor, integrate, derivative)
from .complex_taylor_function import (ComplexMultivariateTaylorFunction,
    convert_to_cmtf)
from .elementary_coefficients import load_precomputed_coefficients
from .taylor_map import TaylorMap

MTF = MultivariateTaylorFunction


'''
Purpose of __all__: The __all__ list defines what names are considered public
when a user does from mtflib import *. It's crucial for controlling the
public API of your library.
'''
__all__ = [
    'taylor_function',
    'load_precomputed_coefficients',
    'elementary_functions',
    'MultivariateTaylorFunction',
    'Var',
    'mtfarray',
    'MTF',
    'convert_to_mtf',
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
