# elementary_functions.py
"""
This module provides implementations of Taylor series expansions for various elementary functions
around zero, using the MultivariateTaylorFunction class.

It includes functions for cosine, sine, exponential, square root (around 1), logarithm (around 1),
arctan, sinh, cosh, tanh, arcsin, arccos, arctanh, and Gaussian functions.

Each function takes a variable (which can be a MultivariateTaylorFunction, Var, or a scalar)
and an optional order for the Taylor expansion.
"""
import math
import numpy as np
from MTFLibrary.taylor_function import MultivariateTaylorFunction, get_global_max_order, convert_to_mtf
from MTFLibrary.taylor_function import Var  # Correct import of Var from taylor_function - v9.9.6


def _generate_exponent(order, var_index, dimension):
    """Generates an exponent tuple of given dimension, with order at var_index and 0 elsewhere."""
    exponent = [0] * dimension
    exponent[var_index] = order
    return tuple(exponent)

def cos_taylor(variable, order=None):
    """
    Taylor expansion of cosine function around 0.
    cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 1 # Determine var_id, default to 1 if not Var - v9.9.8
    coeffs = {}
    for n in range(0, order + 1, 2):
        coeff = (-1)**(n/2) / math.factorial(n)
        coeffs[_generate_exponent(n, var_id_for_exponent - 1, dimension)] = np.array([coeff]) # Use determined var_id - v9.9.8
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension)


def sin_taylor(variable, order=None):
    """
    Taylor expansion of sine function around 0.
    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 2 # Determine var_id, default to 2 if not Var - v9.9.8, assuming sin_taylor is usually for y (Var ID 2)
    coeffs = {}
    for n in range(1, order + 1, 2):
        coeff = (-1)**((n-1)/2) / math.factorial(n)
        coeffs[_generate_exponent(n, var_id_for_exponent - 1, dimension)] = np.array([coeff]) # Use determined var_id - v9.9.8
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension)


def exp_taylor(variable, order=None):
    """
    Taylor expansion of exponential function around 0.
    exp(x) = 1 + x + x^2/2! + x^3/3! + x^4/4! + ...
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 1 # Default to 1 for exp - v9.9.8
    coeffs = {}
    for n in range(0, order + 1):
        coeff = 1.0 / math.factorial(n)
        coeffs[_generate_exponent(n, var_id_for_exponent - 1, dimension)] = np.array([coeff])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension)


def gaussian_taylor(variable, order=None):
    """
    Taylor expansion of Gaussian function exp(-x^2) around 0.
    exp(-x^2) = 1 - x^2 + x^4/2! - x^6/3! + x^8/4! - ...
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 1 # Default to 1 for gaussian - v9.9.8
    coeffs = {}
    for n in range(0, order + 1, 2):
        coeff = (-1)**(n/2) / math.factorial(n//2)
        coeffs[_generate_exponent(n, var_id_for_exponent - 1, dimension)] = np.array([coeff])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension)


def sqrt_taylor(variable, order=None):
    """
    Taylor expansion of sqrt(1+x) around 0.
    sqrt(1+x) = 1 + 1/2*x - 1/8*x^2 + 1/16*x^3 - 5/128*x^4 + ... for |x| < 1
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 1 # Default to 1 for sqrt - v9.9.8
    coeffs = {}
    coeffs[_generate_exponent(0, var_id_for_exponent - 1, dimension)] = np.array([1.0])
    coeffs[_generate_exponent(1, var_id_for_exponent - 1, dimension)] = np.array([0.5])
    coeffs[_generate_exponent(2, var_id_for_exponent - 1, dimension)] = np.array([-0.125])
    coeffs[_generate_exponent(3, var_id_for_exponent - 1, dimension)] = np.array([0.0625])
    coeffs[_generate_exponent(4, var_id_for_exponent - 1, dimension)] = np.array([-0.0390625])

    truncated_coeffs = {}
    for exponent, coefficient in coeffs.items():
        if sum(exponent) <= order:
            truncated_coeffs[exponent] = coefficient
    return MultivariateTaylorFunction(coefficients=truncated_coeffs, dimension=dimension)


def log_taylor(variable, order=None):
    """
    Taylor expansion of log(1+y) around 0.
    log(1+y) = y - y^2/2 + y^3/3 - y^4/4 + ... for |y| < 1
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 2 # Default to 2 for log - v9.9.8
    coeffs = {}
    for n in range(1, order + 1):
        coeff = (-1)**(n-1) / float(n)
        coeffs[_generate_exponent(n, var_id_for_exponent - 1, dimension)] = np.array([coeff])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension)


def arctan_taylor(variable, order=None):
    """
    Taylor expansion of arctan(x) around 0.
    arctan(x) = x - x^3/3 + x^5/5 - x^7/7 + ... for |x| <= 1
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 1 # Default to 1 for arctan - v9.9.8
    coeffs = {}
    for n in range(1, order + 1, 2):
        coeff = (-1)**((n-1)/2) / float(n)
        coeffs[_generate_exponent(n, var_id_for_exponent - 1, dimension)] = np.array([coeff])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension)


def sinh_taylor(variable, order=None):
    """
    Taylor expansion of sinh(y) around 0.
    sinh(y) = y + y^3/3! + y^5/5! + y^7/7! + ...
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 2 # Default to 2 for sinh - v9.9.8
    coeffs = {}
    for n in range(1, order + 1, 2):
        coeff = 1.0 / math.factorial(n)
        coeffs[_generate_exponent(n, var_id_for_exponent - 1, dimension)] = np.array([coeff])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension)


def cosh_taylor(variable, order=None):
    """
    Taylor expansion of cosh(x) around 0.
    cosh(x) = 1 + x^2/2! + x^4/4! + x^6/6! + ...
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 1 # Default to 1 for cosh - v9.9.8
    coeffs = {}
    for n in range(0, order + 1, 2):
        coeff = 1.0 / math.factorial(n)
        coeffs[_generate_exponent(n, var_id_for_exponent - 1, dimension)] = np.array([coeff])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension)


def tanh_taylor(variable, order=None):
    """
    Taylor expansion of tanh(y) around 0.
    tanh(y) = y - y^3/3 + 2*y^5/15 - 17*y^7/315 + ... for |y| < pi/2
    (up to order 4 for this example)
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 2 # Default to 2 for tanh - v9.9.8
    coeffs = {}
    coeffs[_generate_exponent(1, var_id_for_exponent - 1, dimension)] = np.array([1.0])
    coeffs[_generate_exponent(3, var_id_for_exponent - 1, dimension)] = np.array([-1.0/3.0])
    coeffs[_generate_exponent(5, var_id_for_exponent - 1, dimension)] = np.array([2.0/15.0])
    coeffs[_generate_exponent(7, var_id_for_exponent - 1, dimension)] = np.array([-17.0/315.0])

    truncated_coeffs = {}
    for exponent, coefficient in coeffs.items():
        if sum(exponent) <= order:
            truncated_coeffs[exponent] = coefficient
    return MultivariateTaylorFunction(coefficients=truncated_coeffs, dimension=dimension)


def arcsin_taylor(variable, order=None):
    """
    Taylor expansion of arcsin(x) around 0.
    arcsin(x) = x + 1/6*x^3 + 3/40*x^5 + 5/112*x^7 + ... for |x| <= 1
    (up to order 4 for this example)
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 1 # Default to 1 for arcsin - v9.9.8
    coeffs = {}
    coeffs[_generate_exponent(1, var_id_for_exponent - 1, dimension)] = np.array([1.0])
    coeffs[_generate_exponent(3, var_id_for_exponent - 1, dimension)] = np.array([1.0/6.0])
    coeffs[_generate_exponent(5, var_id_for_exponent - 1, dimension)] = np.array([3.0/40.0])
    coeffs[_generate_exponent(7, var_id_for_exponent - 1, dimension)] = np.array([5.0/112.0])

    truncated_coeffs = {}
    for exponent, coefficient in coeffs.items():
        if sum(exponent) <= order:
            truncated_coeffs[exponent] = coefficient
    return MultivariateTaylorFunction(coefficients=truncated_coeffs, dimension=dimension)


def arccos_taylor(variable, order=None):
    """
    Taylor expansion of arccos(y) around 0.
    arccos(y) = pi/2 - (y + 1/6*y^3 + 3/40*y^5 + 5/112*y^7 + ...) for |y| <= 1
    (up to order 4 for this example)
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 2 # Default to 2 for arccos - v9.9.8
    coeffs = {}
    coeffs[_generate_exponent(0, var_id_for_exponent - 1, dimension)] = np.array([math.pi/2.0])
    coeffs[_generate_exponent(1, var_id_for_exponent - 1, dimension)] = np.array([-1.0])
    coeffs[_generate_exponent(3, var_id_for_exponent - 1, dimension)] = np.array([-1.0/6.0])
    coeffs[_generate_exponent(5, var_id_for_exponent - 1, dimension)] = np.array([-3.0/40.0])
    coeffs[_generate_exponent(7, var_id_for_exponent - 1, dimension)] = np.array([-5.0/112.0])

    truncated_coeffs = {}
    for exponent, coefficient in coeffs.items():
        if sum(exponent) <= order:
            truncated_coeffs[exponent] = coefficient
    return MultivariateTaylorFunction(coefficients=truncated_coeffs, dimension=dimension)


def arctanh_taylor(variable, order=None):
    """
    Taylor expansion of arctanh(x) around 0.
    arctanh(x) = x + x^3/3 + x^5/5 + x^7/7 + ... for |x| < 1
    """
    if order is None:
        order = get_global_max_order()
    variable_mtf = convert_to_mtf(variable)
    dimension = variable_mtf.dimension
    var_id_for_exponent = variable.var_id if isinstance(variable, Var) else 1 # Default to 1 for arctanh - v9.9.8
    coeffs = {}
    for n in range(1, order + 1, 2):
        coeff = 1.0 / float(n)
        coeffs[_generate_exponent(n, var_id_for_exponent - 1, dimension)] = np.array([coeff])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension)


# Modified _generate_exponent to take var_index explicitly - v9.9.8
# In elementary functions, determine var_id_for_exponent:
# If input variable is Var, use variable.var_id, else default to 1 or 2 based on function context (x or y usually) - v9.9.8