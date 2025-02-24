# MTFLibrary/elementary_functions.py
"""
This module provides implementations of Taylor series expansions for various elementary functions
around zero, using the MultivariateTaylorFunction class and the Var function.

... (rest of the docstring remains the same) ...
"""
import math
import numpy as np
from MTFLibrary.taylor_function import MultivariateTaylorFunction, get_global_max_order, convert_to_mtf  # Correct import of Var function
# from MTFLibrary.complex_taylor_function import ComplexMultivariateTaylorFunction

def _generate_exponent(order, var_index, dimension):
    """Generates an exponent tuple of given dimension, with order at var_index and 0 elsewhere."""
    exponent = [0] * dimension
    exponent[var_index] = order
    return tuple(exponent)

def cos_taylor(variable, order=None):
    """
    Taylor expansion of cosine function, now supporting composition.
    cos(variable), where variable can be a scalar, Var object, or MTF.
    ... (docstring from previous response) ...
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D cosine Taylor series (cos_taylor_1d) in variable 'u'
    cos_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for cos(u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    for n in range(0, order + 1, 2):
        coeff = (-1)**(n/2) / math.factorial(n)
        cos_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    cos_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=cos_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose cos_taylor_1d(g_mtf) - substitute g_mtf into cos_taylor_1d
    composed_mtf = cos_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def sin_taylor(variable, order=None):
    """
    Taylor expansion of sine function around 0.
    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of sin(variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D sine Taylor series (sin_taylor_1d) in variable 'u'
    sin_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for sin(u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    for n in range(1, order + 1, 2):
        coeff = (-1)**((n-1)/2) / math.factorial(n)
        sin_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    sin_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=sin_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose sin_taylor_1d(g_mtf) - substitute g_mtf into sin_taylor_1d
    composed_mtf = sin_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def exp_taylor(variable, order=None):
    """
    Taylor expansion of exponential function around 0.
    exp(x) = 1 + x + x^2/2! + x^3/3! + x^4/4! + ...

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of exp(variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D exponential Taylor series (exp_taylor_1d) in variable 'u'
    exp_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for exp(u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    for n in range(0, order + 1):
        coeff = 1.0 / math.factorial(n)
        exp_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    exp_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=exp_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose exp_taylor_1d(g_mtf) - substitute g_mtf into exp_taylor_1d
    composed_mtf = exp_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def gaussian_taylor(variable, order=None):
    """
    Taylor expansion of Gaussian function exp(-x^2) around 0.
    exp(-x^2) = 1 - x^2 + x^4/2! - x^6/3! + x^8/4! - ...

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of gaussian(variable) [exp(-variable^2)].
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D Gaussian Taylor series (gaussian_taylor_1d) in variable 'u'
    gaussian_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for gaussian(u) = exp(-u^2)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    for n in range(0, order + 1, 2):
        coeff = (-1)**(n/2) / math.factorial(n//2)
        gaussian_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    gaussian_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=gaussian_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose gaussian_taylor_1d(g_mtf) - substitute g_mtf into gaussian_taylor_1d
    composed_mtf = gaussian_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def sqrt_taylor(variable, order=None):
    """
    Taylor expansion of sqrt(1+x) around 0.
    sqrt(1+x) = 1 + 1/2*x - 1/8*x^2 + 1/16*x^3 - 5/128*x^4 + ... for |x| < 1

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of sqrt(1+variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D sqrt Taylor series (sqrt_taylor_1d) in variable 'u' - around 0 (for 1+u)
    sqrt_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for sqrt(1+u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    sqrt_taylor_1d_coeffs[_generate_exponent(0, var_index_1d, taylor_dimension_1d)] = np.array([1.0]).reshape(1)
    sqrt_taylor_1d_coeffs[_generate_exponent(1, var_index_1d, taylor_dimension_1d)] = np.array([0.5]).reshape(1)
    sqrt_taylor_1d_coeffs[_generate_exponent(2, var_index_1d, taylor_dimension_1d)] = np.array([-0.125]).reshape(1)
    sqrt_taylor_1d_coeffs[_generate_exponent(3, var_index_1d, taylor_dimension_1d)] = np.array([0.0625]).reshape(1)
    sqrt_taylor_1d_coeffs[_generate_exponent(4, var_index_1d, taylor_dimension_1d)] = np.array([-0.0390625]).reshape(1)

    sqrt_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=sqrt_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose sqrt_taylor_1d(g_mtf) - substitute g_mtf into sqrt_taylor_1d
    composed_mtf = sqrt_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def log_taylor(variable, order=None):
    """
    Taylor expansion of log(1+y) around 0.
    log(1+y) = y - y^2/2 + y^3/3 - y^4/4 + ... for |y| < 1

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of log(1+variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D log Taylor series (log_taylor_1d) in variable 'u' - around 0 (for 1+u)
    log_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for log(1+u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    for n in range(1, order + 1):
        coeff = (-1)**(n-1) / float(n)
        log_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    log_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=log_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose log_taylor_1d(g_mtf) - substitute g_mtf into log_taylor_1d
    composed_mtf = log_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def arctan_taylor(variable, order=None):
    """
    Taylor expansion of arctan(x) around 0.
    arctan(x) = x - x^3/3 + x^5/5 - x^7/7 + ... for |x| <= 1

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of arctan(variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D arctan Taylor series (arctan_taylor_1d) in variable 'u'
    arctan_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for arctan(u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    for n in range(1, order + 1, 2):
        coeff = (-1)**((n-1)/2) / float(n)
        arctan_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    arctan_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=arctan_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose arctan_taylor_1d(g_mtf) - substitute g_mtf into arctan_taylor_1d
    composed_mtf = arctan_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def sinh_taylor(variable, order=None):
    """
    Taylor expansion of sinh(y) around 0.
    sinh(y) = y + y^3/3! + y^5/5! + y^7/7! + ...

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of sinh(variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D sinh Taylor series (sinh_taylor_1d) in variable 'u'
    sinh_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for sinh(u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    for n in range(1, order + 1, 2):
        coeff = 1.0 / math.factorial(n)
        sinh_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    sinh_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=sinh_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose sinh_taylor_1d(g_mtf) - substitute g_mtf into sinh_taylor_1d
    composed_mtf = sinh_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def cosh_taylor(variable, order=None):
    """
    Taylor expansion of cosh(x) around 0.
    cosh(x) = 1 + x^2/2! + x^4/4! + x^6/6! + ...

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of cosh(variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D cosh Taylor series (cosh_taylor_1d) in variable 'u'
    cosh_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for cosh(u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    for n in range(0, order + 1, 2):
        coeff = 1.0 / math.factorial(n)
        cosh_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    cosh_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=cosh_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose cosh_taylor_1d(g_mtf) - substitute g_mtf into cosh_taylor_1d
    composed_mtf = cosh_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def tanh_taylor(variable, order=None):
    """
    Taylor expansion of tanh(y) around 0.
    tanh(y) = y - y^3/3 + 2*y^5/15 - 17*y^7/315 + ... for |y| < pi/2
    (up to order max_order for this example)

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of tanh(variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D tanh Taylor series (tanh_taylor_1d) in variable 'u'
    tanh_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for tanh(u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    # tanh coefficients up to order 'order' - use Bernoulli numbers or pre-calculate
    # For simplicity, let's hardcode a few terms, or calculate Bernoulli numbers if available
    tanh_coeffs_list = [0, 1, 0, -1/3.0, 0, 2/15.0, 0, -17/315.0, 0, 62/2835.0, 0, -1382/155925.0] # Up to order 11
    for n in range(1, order + 1, 2):
        if n < len(tanh_coeffs_list):
            coeff = tanh_coeffs_list[n]
            tanh_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)

    tanh_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=tanh_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose tanh_taylor_1d(g_mtf) - substitute g_mtf into tanh_taylor_1d
    composed_mtf = tanh_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def arcsin_taylor(variable, order=None):
    """
    Taylor expansion of arcsin(x) around 0.
    arcsin(x) = x + 1/6*x^3 + 3/40*x^5 + 5/112*x^7 + ... for |x| <= 1
    (up to max_order for this example)

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of arcsin(variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D arcsin Taylor series (arcsin_taylor_1d) in variable 'u'
    arcsin_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for arcsin(u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    # arcsin coefficients - calculate or pre-calculate as needed for higher orders
    arcsin_coeffs_list = [0, 1, 0, 1/6.0, 0, 3/40.0, 0, 5/112.0, 0, 35/1152.0, 0, 63/2816.0] # Up to order 11
    for n in range(1, order + 1, 2):
        if n < len(arcsin_coeffs_list):
            coeff = arcsin_coeffs_list[n]
            arcsin_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)

    arcsin_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=arcsin_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose arcsin_taylor_1d(g_mtf) - substitute g_mtf into arcsin_taylor_1d
    composed_mtf = arcsin_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


def arccos_taylor(variable, order=None):
    """
    Taylor expansion of arccos(y) around 0.
    arccos(y) = pi/2 - (y + 1/6*y^3 + 3/40*y^5 + 5/112*y^7 + ...) for |y| <= 1
    (up to max_order for this example)

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of arccos(variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D arcsin Taylor series (arcsin_taylor_1d) - reuse arcsin series, then adjust for arccos
    arcsin_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for arcsin(u) (will reuse coeffs)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    # arcsin coefficients - same as arcsin_taylor
    arcsin_coeffs_list = [0, 1, 0, 1/6.0, 0, 3/40.0, 0, 5/112.0, 0, 35/1152.0, 0, 63/2816.0] # Up to order 11
    for n in range(1, get_global_max_order() + 1, 2): # Generate up to max order for arcsin part
        if n < len(arcsin_coeffs_list):
            coeff = arcsin_coeffs_list[n]
            arcsin_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    arcsin_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=arcsin_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # For arccos, it's pi/2 - arcsin(y)
    arccos_taylor_1d_coeffs = {}
    arccos_taylor_1d_coeffs[_generate_exponent(0, var_index_1d, taylor_dimension_1d)] = np.array([math.pi/2.0]).reshape(1) # Constant term pi/2
    # Copy coefficients from -arcsin_taylor_1d_mtf for the rest (negate non-constant terms)
    for exponents, coeff_value in arcsin_taylor_1d_mtf.coefficients.items():
        if sum(exponents) > 0: # Negate only non-constant terms of arcsin series
            arccos_taylor_1d_coeffs[exponents] = -coeff_value
        else: # Constant term from arcsin is 0, so no constant term to copy (or negate), pi/2 already set
            pass # Or equivalently: arccos_taylor_1d_coeffs[exponents] = coeff_value * 0

    arccos_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=arccos_taylor_1d_coeffs, dimension=taylor_dimension_1d)


    # (c) Compose arccos_taylor_1d(g_mtf) - substitute g_mtf into arccos_taylor_1d
    composed_mtf = arccos_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf



def arctanh_taylor(variable, order=None):
    """
    Taylor expansion of arctanh(x) around 0.
    arctanh(x) = x + x^3/3 + x^5/5 + x^7/7 + ... for |x| < 1

    Input:
        variable: Scalar, Var object, or MultivariateTaylorFunction.
        order (optional): Order of Taylor expansion. If None, uses global max order.

    Returns:
        MultivariateTaylorFunction: Taylor expansion of arctanh(variable).
    """
    if order is None:
        order = get_global_max_order()

    # (a) Convert input to MTF (g_mtf) - preserve its dimension
    g_mtf = convert_to_mtf(variable)

    # (b) Create 1D arctanh Taylor series (arctanh_taylor_1d) in variable 'u'
    arctanh_taylor_1d_coeffs = {}
    taylor_dimension_1d = 1 # 1D Taylor series for arctanh(u)
    var_index_1d = 0 # Variable index for 1D series (only one variable, u or x_1)
    for n in range(1, order + 1, 2):
        coeff = 1.0 / float(n)
        arctanh_taylor_1d_coeffs[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff]).reshape(1)
    arctanh_taylor_1d_mtf = MultivariateTaylorFunction(coefficients=arctanh_taylor_1d_coeffs, dimension=taylor_dimension_1d)

    # (c) Compose arctanh_taylor_1d(g_mtf) - substitute g_mtf into arctanh_taylor_1d
    composed_mtf = arctanh_taylor_1d_mtf.compose(g_mtf)

    # (d) Truncate the result to the requested order (or global max order)
    truncated_mtf = composed_mtf.truncate(order)

    return truncated_mtf


