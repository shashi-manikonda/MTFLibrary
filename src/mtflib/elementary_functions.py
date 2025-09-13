"""
Taylor series for elementary functions, integration, and differentiation.

This module provides functions for computing the Taylor series expansions of
common elementary functions (e.g., sin, exp, log). It also provides the
core operators for differentiation and integration, which, together with the
arithmetic operations in `MultivariateTaylorFunction`, form a complete
Differential Algebra (DA).
"""

import math

import numpy as np

from . import (
    elementary_coefficients,
)  # Import the new module with loaded coefficients
from .complex_taylor_function import ComplexMultivariateTaylorFunction
from .taylor_function import (
    MultivariateTaylorFunction,
    _generate_exponent,
    _split_constant_polynomial_part,
    _sqrt_taylor,
)


def _apply_constant_factoring(
    input_mtf: MultivariateTaylorFunction,
    constant_factor_function,
    expansion_function_around_zero,
    combine_operation,
) -> MultivariateTaylorFunction:
    """Helper: Applies constant factoring for Taylor expansion."""
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_factor_value = constant_factor_function(constant_term_C_value)
    polynomial_expansion_mtf = expansion_function_around_zero(polynomial_part_mtf)

    if combine_operation == "*":
        result_mtf = polynomial_expansion_mtf * constant_factor_value
    elif combine_operation == "+":
        result_mtf = polynomial_expansion_mtf + constant_factor_value
    elif combine_operation == "-":
        result_mtf = polynomial_expansion_mtf - constant_factor_value
    else:
        raise ValueError(
            f"Unsupported combine_operation: {combine_operation}. "
            "Must be '*', '+', or '-'."
        )

    return result_mtf


def _sin_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `sin(x)`.

    Uses the identity `sin(C + p) = sin(C)cos(p) + cos(C)sin(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `sin(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.sin(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1 -1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_sin_C = math.sin(constant_term_C_value)
    constant_cos_C = math.cos(constant_term_C_value)

    term1_mtf = cos_taylor_around_zero(polynomial_part_mtf) * constant_sin_C
    term2_mtf = sin_taylor_around_zero(polynomial_part_mtf) * constant_cos_C

    result_mtf = term1_mtf + term2_mtf
    return result_mtf.truncate(order)


def _cos_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `cos(x)`.

    Uses the identity `cos(C + p) = cos(C)cos(p) - sin(C)sin(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `cos(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.cos(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      0    (0,)
    1 -5.000000e-01      2    (2,)
    2  4.166667e-02      4    (4,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_cos_C = math.cos(constant_term_C_value)
    constant_sin_C = math.sin(constant_term_C_value)

    term1_mtf = cos_taylor_around_zero(polynomial_part_mtf) * constant_cos_C
    term2_mtf = sin_taylor_around_zero(polynomial_part_mtf) * constant_sin_C

    result_mtf = term1_mtf - term2_mtf
    return result_mtf.truncate(order)


def sin_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunction:
    """Helper: Taylor expansion of sin(u) around zero, precomputed coefficients."""
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    sin_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get("sin")
    if precomputed_coeffs is None:
        raise ValueError(
            "Precomputed coefficients for 'sin' function not found. "
            "Ensure coefficients are loaded."
        )

    for n_order in range(1, max_precomputed_order + 1, 2):
        coefficient_val = precomputed_coeffs[n_order]
        sin_taylor_1d_coefficients[
            _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
        ] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(
            f"Warning: Requested order {order} exceeds precomputed order "
            f"{elementary_coefficients.MAX_PRECOMPUTED_ORDER}."
            " Calculations may be slower for higher orders."
        )
        for n_order in range(
            elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1, 2
        ):  # Calculate dynamically for higher orders
            coefficient_val = (-1) ** ((n_order - 1) // 2) / math.factorial(n_order)
            sin_taylor_1d_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val

    sin_taylor_1d_mtf = type(input_mtf)(
        coefficients=sin_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = sin_taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)


def cos_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunction:
    """Helper: Taylor expansion of cos(u) around zero, precomputed coefficients."""
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    cos_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get("cos")
    if precomputed_coeffs is None:
        raise ValueError(
            "Precomputed coefficients for 'cos' function not found. "
            "Ensure coefficients are loaded."
        )

    for n_order in range(0, max_precomputed_order + 1, 2):
        coefficient_val = precomputed_coeffs[n_order]
        cos_taylor_1d_coefficients[
            _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
        ] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        warning_msg = (
            f"Warning: Requested order {order} exceeds precomputed order "
            f"{elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
            "Calculations may be slower for higher orders."
        )
        print(warning_msg)
        for n_order in range(
            elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1, 2
        ):  # Calculate dynamically for higher orders
            coefficient_val = (-1) ** (n_order // 2) / math.factorial(n_order)
            cos_taylor_1d_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val

    cos_taylor_1d_mtf = type(input_mtf)(
        coefficients=cos_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = cos_taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)


def _tan_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `tan(x)`.

    Calculated as `sin(x) / cos(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `tan(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.tan(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1  3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    sin_mtf = _sin_taylor(variable, order=order)
    cos_mtf = _cos_taylor(variable, order=order)
    tan_mtf = sin_mtf / cos_mtf
    return tan_mtf.truncate(order)


def _exp_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `exp(x)`.

    Uses the identity `exp(C + p) = exp(C) * exp(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `exp(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.exp(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      0    (0,)
    1  1.000000e+00      1    (1,)
    2  5.000000e-01      2    (2,)
    3  1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    return _apply_constant_factoring(
        input_mtf, math.exp, exp_taylor_around_zero, "*"
    ).truncate(order)


def exp_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunction:
    """Helper: Taylor expansion of exp(u) around zero, precomputed coefficients."""
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    exp_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get("exp")
    if precomputed_coeffs is None:
        raise ValueError(
            "Precomputed coefficients for 'exp' function not found. "
            "Ensure coefficients are loaded."
        )

    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        exp_taylor_1d_coefficients[
            _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
        ] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        warning_msg = (
            f"Warning: Requested order {order} exceeds precomputed order "
            f"{elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
            "Calculations may be slower for higher orders."
        )
        print(warning_msg)
        for n_order in range(
            elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1
        ):  # Calculate dynamically for higher orders
            coefficient_val = 1.0 / math.factorial(n_order)
            exp_taylor_1d_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val

    exp_taylor_1d_mtf = type(input_mtf)(
        coefficients=exp_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = exp_taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)


def _gaussian_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of a Gaussian function, `exp(-x^2)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `exp(-x^2)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.gaussian(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      0    (0,)
    1 -1.000000e+00      2    (2,)
    2  5.000000e-01      4    (4,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    return _exp_taylor(-(input_mtf**2), order=order)


def _log_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `log(x)`.

    Uses the identity `log(C + p) = log(C) + log(1 + p/C)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`. Must have a positive constant term.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `log(x)`.

    Raises
    ------
    ValueError
        If the constant term of the input is not positive.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> # We compute log(1+x) since log(x) is singular at x=0
    >>> f = mtf.log(1 + x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1 -5.000000e-01      2    (2,)
    2  3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)

    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(
        input_mtf
    )

    if constant_term_C_value <= 1e-9:
        raise ValueError(
            "Constant part of input to log_taylor is too close to zero or "
            "negative. Logarithm is not defined for non-positive values, "
            "and this method requires a positive constant term."
        )
    if constant_term_C_value < 0:  # Explicit check for negative constant part
        raise ValueError(
            "Constant part of input to log_taylor is negative. Logarithm is "
            "not defined for negative values."
        )

    constant_factor_log_C = math.log(constant_term_C_value)
    polynomial_part_x_mtf = polynomial_part_B_mtf / constant_term_C_value
    log_1_plus_x_mtf = log_taylor_1D_expansion(polynomial_part_x_mtf, order=order)
    result_mtf = log_1_plus_x_mtf + constant_factor_log_C
    return result_mtf.truncate(order)


def log_taylor_1D_expansion(variable, order: int = None) -> MultivariateTaylorFunction:
    """Helper: 1D Taylor expansion of log(1+u) around zero, precomputed coefficients."""
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    log_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get("log")
    if precomputed_coeffs is None:
        raise ValueError(
            "Precomputed coefficients for 'log' function not found. "
            "Ensure coefficients are loaded."
        )

    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        log_taylor_1d_coefficients[
            _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
        ] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        warning_msg = (
            f"Warning: Requested order {order} exceeds precomputed order "
            f"{elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
            "Calculations may be slower for higher orders."
        )
        print(warning_msg)
        for n_order in range(
            elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1
        ):  # Calculate dynamically for higher orders
            if n_order == 0:
                coefficient_val = 0.0
            elif n_order >= 1:
                coefficient_val = ((-1) ** (n_order - 1)) / n_order
            log_taylor_1d_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val

    log_taylor_1d_mtf = type(input_mtf)(
        coefficients=log_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = log_taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)


def _arctan_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `arctan(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `arctan(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.arctan(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1 -3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)

    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(
        input_mtf
    )

    constant_arctan_C = math.atan(constant_term_C_value)
    denominator_mtf = MultivariateTaylorFunction.from_constant(
        1.0 + constant_term_C_value**2
    ) + (float(constant_term_C_value) * polynomial_part_B_mtf)
    argument_mtf = polynomial_part_B_mtf / denominator_mtf
    arctan_argument_mtf = arctan_taylor_1D_expansion(argument_mtf, order=order)
    result_mtf = constant_arctan_C + arctan_argument_mtf
    return result_mtf.truncate(order)


def arctan_taylor_1D_expansion(
    variable, order: int = None
) -> MultivariateTaylorFunction:
    """
    Helper: 1D Taylor expansion of arctan(u) around zero, precomputed coefficients.
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    arctan_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get("arctan")
    if precomputed_coeffs is None:
        raise ValueError(
            "Precomputed coefficients for 'arctan' function not found. "
            "Ensure coefficients are loaded."
        )

    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        arctan_taylor_1d_coefficients[
            _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
        ] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        warning_msg = (
            f"Warning: Requested order {order} exceeds precomputed order "
            f"{elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
            "Calculations may be slower for higher orders."
        )
        print(warning_msg)
        for n_order in range(
            elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1
        ):  # Calculate dynamically for higher orders
            if n_order == 0:
                coefficient_val = 0.0
            elif n_order % 2 != 0:  # Arctan series has only odd terms
                term_index = (n_order - 1) // 2
                coefficient_val = ((-1) ** term_index) / n_order
            else:
                coefficient_val = 0.0
            arctan_taylor_1d_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val

    arctan_taylor_1d_mtf = type(input_mtf)(
        coefficients=arctan_taylor_1d_coefficients,
        dimension=taylor_dimension_1d,
    )
    composed_mtf = arctan_taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)


def _sinh_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `sinh(x)`.

    Uses the identity `sinh(C + p) = sinh(C)cosh(p) + cosh(C)sinh(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `sinh(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.sinh(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1  1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_sinh_C = math.sinh(constant_term_C_value)
    constant_cosh_C = math.cosh(constant_term_C_value)

    term1_mtf = cosh_taylor_around_zero(polynomial_part_mtf) * constant_sinh_C
    term2_mtf = sinh_taylor_around_zero(polynomial_part_mtf) * constant_cosh_C

    result_mtf = term1_mtf + term2_mtf
    return result_mtf.truncate(order)


def _cosh_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `cosh(x)`.

    Uses the identity `cosh(C + p) = cosh(C)cosh(p) + sinh(C)sinh(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `cosh(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.cosh(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      0    (0,)
    1  5.000000e-01      2    (2,)
    2  4.166667e-02      4    (4,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_cosh_C = math.cosh(constant_term_C_value)
    constant_sinh_C = math.sinh(constant_term_C_value)

    term1_mtf = cosh_taylor_around_zero(polynomial_part_mtf) * constant_cosh_C
    term2_mtf = sinh_taylor_around_zero(polynomial_part_mtf) * constant_sinh_C

    result_mtf = term1_mtf + term2_mtf
    return result_mtf.truncate(order)


def sinh_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunction:
    """Helper: Taylor expansion of sinh(u) around zero, precomputed coefficients."""
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    sinh_taylor_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get("sinh")
    if precomputed_coeffs is None:
        raise ValueError(
            "Precomputed coefficients for 'sinh' function not found. "
            "Ensure coefficients are loaded."
        )

    for n_order in range(0, max_precomputed_order + 1):
        if n_order % 2 != 0 and n_order <= max_precomputed_order:
            coefficient_val = precomputed_coeffs[n_order]
            sinh_taylor_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val
        elif n_order % 2 == 0:
            coefficient_val = precomputed_coeffs[
                n_order
            ]  # Should be 0.0, already precomputed.
            sinh_taylor_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        warning_msg = (
            f"Warning: Requested order {order} exceeds precomputed order "
            f"{elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
            "Calculations may be slower for higher orders."
        )
        print(warning_msg)
        for n_order in range(
            elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1
        ):  # Calculate dynamically for higher orders
            if n_order % 2 != 0 and n_order <= order:  # Odd orders for sinh
                coefficient_val = 1 / math.factorial(n_order)
                sinh_taylor_coefficients[
                    _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
                ] = coefficient_val
            elif n_order % 2 == 0:  # Even orders for sinh are 0
                coefficient_val = 0.0
                sinh_taylor_coefficients[
                    _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
                ] = coefficient_val

    sinh_taylor_1d_mtf = type(input_mtf)(
        coefficients=sinh_taylor_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = sinh_taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)


def cosh_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunction:
    """Helper: Taylor expansion of cosh(u) around zero, precomputed coefficients."""
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    cosh_taylor_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get("cosh")
    if precomputed_coeffs is None:
        raise ValueError(
            "Precomputed coefficients for 'cosh' function not found. "
            "Ensure coefficients are loaded."
        )

    for n_order in range(0, max_precomputed_order + 1):
        if n_order % 2 == 0 and n_order <= max_precomputed_order:
            coefficient_val = precomputed_coeffs[n_order]
            cosh_taylor_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val
        elif n_order % 2 != 0:
            coefficient_val = precomputed_coeffs[
                n_order
            ]  # Should be 0.0, already precomputed.
            cosh_taylor_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val

    cosh_taylor_1d_mtf = type(input_mtf)(
        coefficients=cosh_taylor_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = cosh_taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)


def _tanh_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `tanh(x)`.

    Calculated as `sinh(x) / cosh(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `tanh(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.tanh(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1 -3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    sinh_mtf = _sinh_taylor(variable, order=order)
    cosh_mtf = _cosh_taylor(variable, order=order)
    tanh_mtf = sinh_mtf / cosh_mtf
    return tanh_mtf.truncate(order)


def _arctanh_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `arctanh(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `arctanh(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.arctanh(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1  3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)

    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(
        input_mtf
    )  # Corrected variable name here and below
    constant_arctanh_C = math.atanh(
        constant_term_C_value
    )  # Note: math.atanh for scalar input

    denominator_mtf = MultivariateTaylorFunction.from_constant(
        1.0 - constant_term_C_value**2
    ) - (float(constant_term_C_value) * polynomial_part_B_mtf)  # Corrected for arctanh
    argument_mtf = polynomial_part_B_mtf / denominator_mtf
    arctanh_argument_mtf = arctanh_taylor_1D_expansion(argument_mtf, order=order)
    result_mtf = constant_arctanh_C + arctanh_argument_mtf  # Correct scalar addition
    return result_mtf.truncate(order)


def arctanh_taylor_1D_expansion(
    variable, order: int = None
) -> MultivariateTaylorFunction:
    """
    Helper: 1D Taylor expansion of arctanh(u) around zero, precomputed coefficients.
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    arctanh_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get("arctanh")
    if precomputed_coeffs is None:
        raise ValueError(
            "Precomputed coefficients for 'arctanh' function not found. "
            "Ensure coefficients are loaded."
        )

    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        arctanh_taylor_1d_coefficients[
            _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
        ] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        warning_msg = (
            f"Warning: Requested order {order} exceeds precomputed order "
            f"{elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
            "Calculations may be slower for higher orders."
        )
        print(warning_msg)
        for n_order in range(
            elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1
        ):  # Calculate dynamically for higher orders
            if n_order == 0:
                coefficient_val = 0.0
            elif n_order % 2 != 0:  # Arctanh series has only odd terms
                coefficient_val = 1.0 / float(n_order)
            else:
                coefficient_val = 0.0
            arctanh_taylor_1d_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = coefficient_val

    arctanh_taylor_1d_mtf = type(input_mtf)(
        coefficients=arctanh_taylor_1d_coefficients,
        dimension=taylor_dimension_1d,
    )
    composed_mtf = arctanh_taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)


def _arcsin_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `arcsin(x)`.

    Uses the identity `arcsin(x) = arctan(x / sqrt(1 - x^2))`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `arcsin(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.arcsin(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1  1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    x_mtf = MultivariateTaylorFunction.to_mtf(variable)
    x_squared_mtf = x_mtf * x_mtf
    one_minus_x_squared_mtf = 1.0 - x_squared_mtf
    sqrt_of_one_minus_x_squared_mtf = _sqrt_taylor(one_minus_x_squared_mtf)
    argument_mtf = x_mtf / sqrt_of_one_minus_x_squared_mtf
    arcsin_mtf = _arctan_taylor(argument_mtf, order=order)
    return arcsin_mtf.truncate(order)


def _arccos_taylor(variable, order: int = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `arccos(x)`.

    Uses the identity `arccos(x) = pi/2 - arcsin(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `arccos(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> import numpy as np
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.arccos(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.570796e+00      0    (0,)
    1 -1.000000e+00      1    (1,)
    2 -1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    arcsin_mtf = _arcsin_taylor(variable, order=order)  # Get arcsin_taylor expansion
    pi_over_2_constant = np.pi / 2.0
    arccos_mtf = (
        MultivariateTaylorFunction.to_mtf(pi_over_2_constant) - arcsin_mtf
    )  # Perform MTF subtraction
    return arccos_mtf.truncate(order)  # Truncate to the desired order


def _integrate(
    mtf_instance,
    integration_variable_index,
    lower_limit=None,
    upper_limit=None,
):
    r"""
    Performs definite or indefinite integration of an MTF.

    This function corresponds to the inverse derivation operator
    :math:`\partial_{\bigcirc}^{-1}` of the Differential Algebra. It integrates
    the Taylor series with respect to one of its variables.

    Parameters
    ----------
    mtf_instance : MultivariateTaylorFunction
        The function to integrate.
    integration_variable_index : int
        The 1-based index of the variable to integrate with respect to.
    lower_limit : float, optional
        The lower limit for definite integration.
    upper_limit : float, optional
        The upper limit for definite integration.

    Returns
    -------
    MultivariateTaylorFunction
        If an indefinite integral, a new MTF representing the integral.
        If a definite integral, a new MTF representing the result after
        integrating and substituting the bounds.

    Raises
    ------
    ValueError
        If limits are partially provided or if the variable index is invalid.
    TypeError
        If inputs have incorrect types.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=2)
    >>> x, y = mtf.var(1), mtf.var(2)
    >>> f = x*y + y**2
    >>>
    >>> # Indefinite integral with respect to x
    >>> F_indef = f.integrate(1)
    >>> print(F_indef.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  5.000000e-01      2    (2, 1)
    1  1.000000e+00      3    (1, 2)
    >>>
    >>> # Definite integral with respect to y from 0 to 2
    >>> F_def = f.integrate(2, lower_limit=0.0, upper_limit=2.0)
    >>> print(F_def.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  2.666667e+00      0    (0, 0)
    1  2.000000e+00      1    (1, 0)
    """
    if not isinstance(
        mtf_instance,
        (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction),
    ):
        raise TypeError(
            "mtf_instance must be a MultivariateTaylorFunction or "
            "ComplexMultivariateTaylorFunction."
        )
    if not isinstance(integration_variable_index, int):
        raise ValueError(
            "integration_variable_index must be an integer dimension index (1-based)."
        )
    if not (1 <= integration_variable_index <= mtf_instance.dimension):
        raise ValueError(
            f"integration_variable_index must be between 1 and "
            f"{mtf_instance.dimension}, inclusive."
        )
    if lower_limit is not None and upper_limit is None:
        raise ValueError(
            "If lower_limit is provided, upper_limit must also be provided for "
            "definite integration."
        )
    if upper_limit is not None and lower_limit is None:
        raise ValueError(
            "If upper_limit is provided, upper_limit must also be provided for "
            "definite integration."
        )
    if lower_limit is not None and not isinstance(lower_limit, (int, float)):
        raise TypeError("lower_limit must be a number (int or float).")
    if upper_limit is not None and not isinstance(upper_limit, (int, float)):
        raise TypeError("upper_limit must be a number (int or float).")

    # Handle empty mtf_instance
    if mtf_instance.exponents.size == 0:
        return type(mtf_instance)(
            (
                np.empty((0, mtf_instance.dimension), dtype=np.int32),
                np.empty((0,), dtype=mtf_instance.coeffs.dtype),
            ),
            mtf_instance.dimension,
        )

    original_max_order = MultivariateTaylorFunction.get_max_order()
    MultivariateTaylorFunction.set_max_order(
        original_max_order + 1
    )  # Step 1: Increment computation order by one.

    new_exponents = mtf_instance.exponents.copy()
    new_coeffs = mtf_instance.coeffs.copy()

    p = new_exponents[:, integration_variable_index - 1]
    new_coeffs /= p + 1
    new_exponents[:, integration_variable_index - 1] += 1

    indefinite_integral_mtf = type(mtf_instance)(
        (new_exponents, new_coeffs), dimension=mtf_instance.dimension
    )

    if lower_limit is not None and upper_limit is not None:
        # Step 3: substitute variable that integration was performed on with
        # upper and lower limit
        upper_limit_mtf = indefinite_integral_mtf.substitute_variable(
            integration_variable_index, upper_limit
        )  # upper Taylor function
        lower_limit_mtf = indefinite_integral_mtf.substitute_variable(
            integration_variable_index, lower_limit
        )  # lower taylor function

        # Step 4: Take difference of the two
        definite_integral_mtf_full_order = upper_limit_mtf - lower_limit_mtf

        # Step 5: reduce order by one and truncate
        MultivariateTaylorFunction.set_max_order(
            original_max_order
        )  # reduce order by one (restore original)
        definite_integral_mtf = definite_integral_mtf_full_order.truncate()  # truncate
        return definite_integral_mtf  # Definite integral as MTF
    else:
        MultivariateTaylorFunction.set_max_order(original_max_order)
        return indefinite_integral_mtf


def _derivative(mtf_instance, deriv_dim):
    r"""
    Computes the partial derivative of an MTF.

    This function corresponds to the derivation operator
    :math:`\partial_{\bigcirc}` of the Differential Algebra. It differentiates
    the Taylor series with respect to one of its variables.

    Parameters
    ----------
    mtf_instance : MultivariateTaylorFunction
        The function to differentiate.
    deriv_dim : int
        The 1-based index of the variable to differentiate with respect to.

    Returns
    -------
    MultivariateTaylorFunction
        A new MTF representing the partial derivative.

    Raises
    ------
    TypeError
        If `mtf_instance` is not a `MultivariateTaylorFunction`.
    ValueError
        If `deriv_dim` is not a valid dimension index.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=2)
    >>> x, y = mtf.var(1), mtf.var(2)
    >>> f = x**3 * y**2
    >>>
    >>> # Differentiate with respect to x
    >>> df_dx = f.derivative(1)
    >>> print(df_dx.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  3.000000e+00      4    (2, 2)
    >>>
    >>> # Differentiate with respect to y
    >>> df_dy = f.derivative(2)
    >>> print(df_dy.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  2.000000e+00      4    (3, 1)
    """
    if not isinstance(mtf_instance, MultivariateTaylorFunction):
        raise TypeError("mtf_instance must be a MultivariateTaylorFunction object.")
    if (
        not isinstance(deriv_dim, int)
        or deriv_dim < 1
        or deriv_dim > mtf_instance.dimension
    ):
        raise ValueError(
            f"deriv_dim must be an integer between 1 and "
            f"{mtf_instance.dimension} inclusive."
        )

    deriv_dim_index = deriv_dim - 1  # Convert 1-based to 0-based index

    # Filter out terms where the exponent in the derivative dimension is 0
    mask = mtf_instance.exponents[:, deriv_dim_index] > 0

    if not np.any(mask):
        return type(mtf_instance)(
            (
                np.empty((0, mtf_instance.dimension), dtype=np.int32),
                np.empty((0,), dtype=mtf_instance.coeffs.dtype),
            ),
            mtf_instance.dimension,
        )

    new_exponents = mtf_instance.exponents[mask].copy()
    new_coeffs = mtf_instance.coeffs[mask].copy()

    p = new_exponents[:, deriv_dim_index]
    new_coeffs *= p
    new_exponents[:, deriv_dim_index] -= 1

    return type(mtf_instance)((new_exponents, new_coeffs), mtf_instance.dimension)
