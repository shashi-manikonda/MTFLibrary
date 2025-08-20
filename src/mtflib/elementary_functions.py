"""
Taylor series expansions for elementary functions around zero, using MultivariateTaylorFunction.

Includes functions for sine, cosine, tangent, exponential, Gaussian, square root,
inverse square root, logarithm, arctangent, arcsin, arccos, sinh, cosh, arctanh Taylor series and integration.

Leverages Taylor series composition and constant factoring for efficiency.

Refactored for reduced redundancy and using precomputed coefficients.
"""

import math
import numpy as np
from collections import defaultdict
from .taylor_function import (get_global_max_order, set_global_max_order,
    convert_to_mtf, MultivariateTaylorFunctionBase, _split_constant_polynomial_part,
    sqrt_taylor, isqrt_taylor)
from .complex_taylor_function import ComplexMultivariateTaylorFunction
from . import elementary_coefficients  # Import the new module with loaded coefficients
# from .MTFExtended import MultivariateTaylorFunction

def _generate_exponent(order_val: int, var_index: int, dimension: int) -> tuple:
    """Helper: Generates exponent tuples."""
    exponent_tuple = [0] * dimension
    if dimension > 0:
        exponent_tuple[var_index] = order_val
    return tuple(exponent_tuple)


def _split_constant_polynomial_part(input_mtf: MultivariateTaylorFunctionBase) -> tuple[float, MultivariateTaylorFunctionBase]:
    """Helper: Splits MTF into constant and polynomial parts."""
    dimension = input_mtf.dimension
    const_exp = np.zeros(dimension, dtype=np.int32)

    match = np.all(input_mtf.exponents == const_exp, axis=1)
    const_idx = np.where(match)[0]

    if const_idx.size > 0:
        constant_term_C_value = input_mtf.coeffs[const_idx[0]]
        poly_mask = ~match
        poly_exponents = input_mtf.exponents[poly_mask]
        poly_coeffs = input_mtf.coeffs[poly_mask]
        polynomial_part_mtf = type(input_mtf)((poly_exponents, poly_coeffs), dimension)
    else:
        constant_term_C_value = 0.0
        polynomial_part_mtf = input_mtf

    return constant_term_C_value, polynomial_part_mtf


def _apply_constant_factoring(
    input_mtf: MultivariateTaylorFunctionBase,
    constant_factor_function,
    expansion_function_around_zero,
    combine_operation
) -> MultivariateTaylorFunctionBase:
    """Helper: Applies constant factoring for Taylor expansion."""
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(input_mtf)
    constant_factor_value = constant_factor_function(constant_term_C_value)
    polynomial_expansion_mtf = expansion_function_around_zero(polynomial_part_mtf)

    if combine_operation == '*':
        result_mtf = polynomial_expansion_mtf * constant_factor_value
    elif combine_operation == '+':
        result_mtf = polynomial_expansion_mtf + constant_factor_value
    elif combine_operation == '-':
        result_mtf = polynomial_expansion_mtf - constant_factor_value
    else:
        raise ValueError(f"Unsupported combine_operation: {combine_operation}. Must be '*', '+', or '-'.")

    return result_mtf

def sin_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of sin(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(input_mtf)
    constant_sin_C = math.sin(constant_term_C_value)
    constant_cos_C = math.cos(constant_term_C_value)

    term1_mtf = cos_taylor_around_zero(polynomial_part_mtf) * constant_sin_C
    term2_mtf = sin_taylor_around_zero(polynomial_part_mtf) * constant_cos_C

    result_mtf = term1_mtf + term2_mtf
    return result_mtf.truncate(order)


def cos_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of cos(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(input_mtf)
    constant_cos_C = math.cos(constant_term_C_value)
    constant_sin_C = math.sin(constant_term_C_value)

    term1_mtf = cos_taylor_around_zero(polynomial_part_mtf) * constant_cos_C
    term2_mtf = sin_taylor_around_zero(polynomial_part_mtf) * constant_sin_C

    result_mtf = term1_mtf - term2_mtf
    return result_mtf.truncate(order)

def sin_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: Taylor expansion of sin(u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    sin_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('sin')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'sin' function not found. Ensure coefficients are loaded.")

    for n_order in range(1, max_precomputed_order + 1, 2):
        coefficient_val = precomputed_coeffs[n_order]
        sin_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(f"Warning: Requested order {order} exceeds precomputed order {elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
              "Calculations may be slower for higher orders.")
        for n_order in range(elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1, 2): # Calculate dynamically for higher orders
            coefficient_val = (-1)**((n_order - 1) // 2) / math.factorial(n_order)
            sin_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val

    sin_taylor_1d_mtf = type(input_mtf)(
        coefficients=sin_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = sin_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)

def cos_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: Taylor expansion of cos(u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    cos_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('cos')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'cos' function not found. Ensure coefficients are loaded.")

    for n_order in range(0, max_precomputed_order + 1, 2):
        coefficient_val = precomputed_coeffs[n_order]
        cos_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(f"Warning: Requested order {order} exceeds precomputed order {elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
              "Calculations may be slower for higher orders.")
        for n_order in range(elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1, 2): # Calculate dynamically for higher orders
            coefficient_val = (-1)**(n_order // 2) / math.factorial(n_order)
            cos_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val

    cos_taylor_1d_mtf = type(input_mtf)(
        coefficients=cos_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = cos_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)


def tan_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of tan(x) using tan(x) = sin(x) / cos(x)."""
    if order is None:
        order = get_global_max_order()
    sin_mtf = sin_taylor(variable, order=order)
    cos_mtf = cos_taylor(variable, order=order)
    tan_mtf = sin_mtf / cos_mtf
    return tan_mtf.truncate(order)


def exp_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of exp(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    return _apply_constant_factoring(
        input_mtf,
        math.exp,
        exp_taylor_around_zero,
        '*'
    ).truncate(order)


def exp_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: Taylor expansion of exp(u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    exp_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('exp')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'exp' function not found. Ensure coefficients are loaded.")


    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        exp_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(f"Warning: Requested order {order} exceeds precomputed order {elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
              "Calculations may be slower for higher orders.")
        for n_order in range(elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1): # Calculate dynamically for higher orders
            coefficient_val = 1.0 / math.factorial(n_order)
            exp_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val

    exp_taylor_1d_mtf = type(input_mtf)(
        coefficients=exp_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = exp_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)


def gaussian_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of Gaussian exp(-x^2) around zero."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    return exp_taylor(-(input_mtf**2), order=order)




def log_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of log(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)

    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(input_mtf)

    if constant_term_C_value <= 1e-9:
        raise ValueError(
            "Constant part of input to log_taylor is too close to zero or negative. "
            "Logarithm is not defined for non-positive values, and this method "
            "requires a positive constant term."
        )
    if constant_term_C_value < 0:  # Explicit check for negative constant part
        raise ValueError(
            "Constant part of input to log_taylor is negative. "
            "Logarithm is not defined for negative values."
        )

    constant_factor_log_C = math.log(constant_term_C_value)
    polynomial_part_x_mtf = polynomial_part_B_mtf / constant_term_C_value
    log_1_plus_x_mtf = log_taylor_1D_expansion(polynomial_part_x_mtf, order=order)
    result_mtf = log_1_plus_x_mtf + constant_factor_log_C
    return result_mtf.truncate(order)


def log_taylor_1D_expansion(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: 1D Taylor expansion of log(1+u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    log_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('log')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'log' function not found. Ensure coefficients are loaded.")

    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        log_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(f"Warning: Requested order {order} exceeds precomputed order {elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
              "Calculations may be slower for higher orders.")
        for n_order in range(elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1): # Calculate dynamically for higher orders
            if n_order == 0:
                coefficient_val = 0.0
            elif n_order >= 1:
                coefficient_val = ((-1)**(n_order - 1)) / n_order
            log_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val


    log_taylor_1d_mtf = type(input_mtf)(
        coefficients=log_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = log_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)


def arctan_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of arctan(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)

    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(input_mtf)

    constant_arctan_C = math.atan(constant_term_C_value)
    denominator_mtf = 1.0 + (constant_term_C_value**2) + (constant_term_C_value * polynomial_part_B_mtf)
    argument_mtf = polynomial_part_B_mtf / denominator_mtf
    arctan_argument_mtf = arctan_taylor_1D_expansion(argument_mtf, order=order)
    result_mtf = constant_arctan_C + arctan_argument_mtf
    return result_mtf.truncate(order)


def arctan_taylor_1D_expansion(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: 1D Taylor expansion of arctan(u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    arctan_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('arctan')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'arctan' function not found. Ensure coefficients are loaded.")

    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        arctan_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(f"Warning: Requested order {order} exceeds precomputed order {elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
              "Calculations may be slower for higher orders.")
        for n_order in range(elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1): # Calculate dynamically for higher orders
            if n_order == 0:
                coefficient_val = 0.0
            elif n_order % 2 != 0:  # Arctan series has only odd terms
                term_index = (n_order - 1) // 2
                coefficient_val = ((-1)**term_index) / n_order
            else:
                coefficient_val = 0.0
            arctan_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val


    arctan_taylor_1d_mtf = type(input_mtf)(
        coefficients=arctan_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = arctan_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)

def sinh_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of sinh(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(input_mtf)
    constant_sinh_C = math.sinh(constant_term_C_value)
    constant_cosh_C = math.cosh(constant_term_C_value)

    term1_mtf = cosh_taylor_around_zero(polynomial_part_mtf) * constant_sinh_C
    term2_mtf = sinh_taylor_around_zero(polynomial_part_mtf) * constant_cosh_C

    result_mtf = term1_mtf + term2_mtf
    return result_mtf.truncate(order)


def cosh_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of cosh(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(input_mtf)
    constant_cosh_C = math.cosh(constant_term_C_value)
    constant_sinh_C = math.sinh(constant_term_C_value)

    term1_mtf = cosh_taylor_around_zero(polynomial_part_mtf) * constant_cosh_C
    term2_mtf = sinh_taylor_around_zero(polynomial_part_mtf) * constant_sinh_C

    result_mtf = term1_mtf + term2_mtf
    return result_mtf.truncate(order)

def sinh_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: Taylor expansion of sinh(u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    sinh_taylor_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('sinh')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'sinh' function not found. Ensure coefficients are loaded.")

    for n_order in range(0, max_precomputed_order + 1):
        if n_order % 2 != 0 and n_order <= max_precomputed_order:
            coefficient_val = precomputed_coeffs[n_order]
            sinh_taylor_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val
        elif n_order % 2 == 0:
            coefficient_val = precomputed_coeffs[n_order] # Should be 0.0, already precomputed.
            sinh_taylor_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val

    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(f"Warning: Requested order {order} exceeds precomputed order {elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
              "Calculations may be slower for higher orders.")
        for n_order in range(elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1): # Calculate dynamically for higher orders
            if n_order % 2 != 0 and n_order <= order:  # Odd orders for sinh
                coefficient_val = 1 / math.factorial(n_order)
                sinh_taylor_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val
            elif n_order % 2 == 0: # Even orders for sinh are 0
                coefficient_val = 0.0
                sinh_taylor_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val


    sinh_taylor_1d_mtf = type(input_mtf)(
        coefficients=sinh_taylor_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = sinh_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)


def cosh_taylor_around_zero(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: Taylor expansion of cosh(u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    cosh_taylor_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('cosh')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'cosh' function not found. Ensure coefficients are loaded.")

    for n_order in range(0, max_precomputed_order + 1):
        if n_order % 2 == 0 and n_order <= max_precomputed_order:
            coefficient_val = precomputed_coeffs[n_order]
            cosh_taylor_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val
        elif n_order % 2 != 0:
            coefficient_val = precomputed_coeffs[n_order] # Should be 0.0, already precomputed.
            cosh_taylor_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val


    cosh_taylor_1d_mtf = type(input_mtf)(
        coefficients=cosh_taylor_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = cosh_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)


def tanh_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of tanh(x) using tanh(x) = sinh(x) / cosh(x)."""
    if order is None:
        order = get_global_max_order()
    sinh_mtf = sinh_taylor(variable, order=order)
    cosh_mtf = cosh_taylor(variable, order=order)
    tanh_mtf = sinh_mtf / cosh_mtf
    return tanh_mtf.truncate(order)

def arctanh_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of arctanh(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)

    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(input_mtf) # Corrected variable name here and below
    constant_arctanh_C = math.atanh(constant_term_C_value) # Note: math.atanh for scalar input

    denominator_mtf = 1.0 - (constant_term_C_value**2) - (constant_term_C_value * polynomial_part_B_mtf) # Corrected for arctanh
    argument_mtf = polynomial_part_B_mtf / denominator_mtf
    arctanh_argument_mtf = arctanh_taylor_1D_expansion(argument_mtf, order=order)
    result_mtf = constant_arctanh_C + arctanh_argument_mtf # Correct scalar addition
    return result_mtf.truncate(order)


def arctanh_taylor_1D_expansion(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: 1D Taylor expansion of arctanh(u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    arctanh_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('arctanh')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'arctanh' function not found. Ensure coefficients are loaded.")


    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        arctanh_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val


    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(f"Warning: Requested order {order} exceeds precomputed order {elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
              "Calculations may be slower for higher orders.")
        for n_order in range(elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1): # Calculate dynamically for higher orders
            if n_order == 0:
                coefficient_val = 0.0
            elif n_order % 2 != 0:  # Arctanh series has only odd terms
                coefficient_val = 1.0 / float(n_order)
            else:
                coefficient_val = 0.0
            arctanh_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = coefficient_val


    arctanh_taylor_1d_mtf = type(input_mtf)(
        coefficients=arctanh_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = arctanh_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)

def arcsin_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """
    Taylor expansion of arcsine function, using arctan_taylor and relation arcsin(x) = arctan(x/sqrt(1-x^2)).
    """
    if order is None:
        order = get_global_max_order()
    x_mtf = convert_to_mtf(variable)
    x_squared_mtf = x_mtf * x_mtf
    one_minus_x_squared_mtf = 1.0 - x_squared_mtf
    sqrt_of_one_minus_x_squared_mtf = sqrt_taylor(one_minus_x_squared_mtf)
    argument_mtf = x_mtf / sqrt_of_one_minus_x_squared_mtf
    arcsin_mtf = arctan_taylor(argument_mtf, order=order)
    return arcsin_mtf.truncate(order)

def arccos_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """
    Taylor expansion of arccosine function using the relation arccos(x) = pi/2 - arcsin(x).
    """
    if order is None:
        order = get_global_max_order()
    arcsin_mtf = arcsin_taylor(variable, order=order)  # Get arcsin_taylor expansion
    pi_over_2_constant = np.pi / 2.0
    arccos_mtf = convert_to_mtf(pi_over_2_constant) - arcsin_mtf  # Perform MTF subtraction
    return arccos_mtf.truncate(order)  # Truncate to the desired order

def integrate(mtf_instance, integration_variable_index, lower_limit=None, upper_limit=None):
    """
    Performs definite or indefinite integration of an MTF with respect to a variable,
    following these steps for definite integration in the specified order:
    (1) Increment computation order by one.
    (2) Perform indefinite integral.
    (3) substitute variable that integration was performed on with upper and lower limit resulting in upper Taylor function and lower taylor function
    (4) Take difference of the two
    (5) reduce order by one and truncate

    Args:
        mtf_instance (MultivariateTaylorFunction): The MTF object to integrate.
        integration_variable_index (int): Dimension index (1-based integer) of the variable to integrate with respect to.
        lower_limit (float, optional): Lower limit for definite integration.
        upper_limit (float, optional): Upper limit for definite integration.

    Returns:
        MultivariateTaylorFunction or np.ndarray:
            - If lower_limit and upper_limit are None: Returns indefinite integral MTF.
            - If limits are provided: Returns definite integral MTF (as a Taylor function).
    """
    if not isinstance(mtf_instance, (MultivariateTaylorFunctionBase, ComplexMultivariateTaylorFunction)):
        raise TypeError("mtf_instance must be a MultivariateTaylorFunction or ComplexMultivariateTaylorFunction.")
    if not isinstance(integration_variable_index, int):
        raise ValueError("integration_variable_index must be an integer dimension index (1-based).")
    if not (1 <= integration_variable_index <= mtf_instance.dimension):
        raise ValueError(f"integration_variable_index must be between 1 and {mtf_instance.dimension}, inclusive.")
    if lower_limit is not None and upper_limit is None:
        raise ValueError("If lower_limit is provided, upper_limit must also be provided for definite integration.")
    if upper_limit is not None and lower_limit is None:
        raise ValueError("If upper_limit is provided, upper_limit must also be provided for definite integration.")
    if lower_limit is not None and not isinstance(lower_limit, (int, float)):
        raise TypeError("lower_limit must be a number (int or float).")
    if upper_limit is not None and not isinstance(upper_limit, (int, float)):
        raise TypeError("upper_limit must be a number (int or float).")

    original_max_order = get_global_max_order()
    set_global_max_order(original_max_order + 1)  # Step 1: Increment computation order by one.

    new_exponents = mtf_instance.exponents.copy()
    new_coeffs = mtf_instance.coeffs.copy()

    p = new_exponents[:, integration_variable_index - 1]
    new_coeffs /= (p + 1)
    new_exponents[:, integration_variable_index - 1] += 1

    indefinite_integral_mtf = type(mtf_instance)((new_exponents, new_coeffs), dimension=mtf_instance.dimension)

    if lower_limit is not None and upper_limit is not None:
        # Step 3: substitute variable that integration was performed on with upper and lower limit
        upper_limit_mtf = indefinite_integral_mtf.substitute_variable(integration_variable_index, upper_limit) # upper Taylor function
        lower_limit_mtf = indefinite_integral_mtf.substitute_variable(integration_variable_index, lower_limit) # lower taylor function

        # Step 4: Take difference of the two
        definite_integral_mtf_full_order = upper_limit_mtf - lower_limit_mtf

        # Step 5: reduce order by one and truncate
        set_global_max_order(original_max_order) # reduce order by one (restore original)
        definite_integral_mtf = definite_integral_mtf_full_order.truncate() # truncate
        return definite_integral_mtf # Definite integral as MTF
    else:
        set_global_max_order(original_max_order)
        return indefinite_integral_mtf

def derivative(mtf_instance, deriv_dim):
    """
    Calculates the derivative of a MultivariateTaylorFunction with respect to a specified dimension.

    Args:
        mtf_instance (MultivariateTaylorFunctionBase): The MultivariateTaylorFunction to differentiate.
        deriv_dim (int): The dimension index (1-based) with respect to which to differentiate.
                         Values should be from 1 to mtf_instance.dimension.

    Returns:
        MultivariateTaylorFunctionBase: A new MTF representing the derivative.

    Raises:
        TypeError: if mtf_instance is not a MultivariateTaylorFunctionBase.
        ValueError: if deriv_dim is not a valid dimension index for the MTF.
    """
    if not isinstance(mtf_instance, MultivariateTaylorFunctionBase):
        raise TypeError("mtf_instance must be a MultivariateTaylorFunctionBase object.")
    if not isinstance(deriv_dim, int) or deriv_dim < 1 or deriv_dim > mtf_instance.dimension:
        raise ValueError(f"deriv_dim must be an integer between 1 and {mtf_instance.dimension} inclusive.")

    deriv_dim_index = deriv_dim - 1 # Convert 1-based to 0-based index

    # Filter out terms where the exponent in the derivative dimension is 0
    mask = mtf_instance.exponents[:, deriv_dim_index] > 0

    if not np.any(mask):
        return type(mtf_instance)((np.empty((0, mtf_instance.dimension), dtype=np.int32), np.empty((0,), dtype=mtf_instance.coeffs.dtype)), mtf_instance.dimension)

    new_exponents = mtf_instance.exponents[mask].copy()
    new_coeffs = mtf_instance.coeffs[mask].copy()

    p = new_exponents[:, deriv_dim_index]
    new_coeffs *= p
    new_exponents[:, deriv_dim_index] -= 1

    return type(mtf_instance)((new_exponents, new_coeffs), mtf_instance.dimension)
