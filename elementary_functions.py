# elementary_functions.py
from taylor_function import MultivariateTaylorFunction, get_global_max_order
from variables import Var # Import Var if not already present
import numpy as np
import math

def cos_taylor(variable, order=None):
    """Taylor expansion of cosine around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(0, order + 1, 2):
        index = (n,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        coeffs[index] = np.array([((-1)**(n/2)) / math.factorial(n)])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension

def sin_taylor(variable, order=None):
    """Taylor expansion of sine around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(1, order + 1, 2):
        index = (n,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        coeffs[index] = np.array([((-1)**((n-1)/2)) / math.factorial(n)])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension

def exp_taylor(variable, order=None):
    """Taylor expansion of exponential function around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(order + 1):
        index = (n,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        coeffs[index] = np.array([1.0 / math.factorial(n)])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension

def sqrt_taylor(variable, order=None):
    """Taylor expansion of sqrt(1+x) around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(order + 1):
        index = (n,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        coeffs[index] = np.array([math.sqrt(1) if n == 0 else (-1)**(n+1) * math.factorial(2*n-2) / ((math.factorial(n-1)**2) * (2**(2*n-1)) ) if n >= 1 else 0.0])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension

def log_taylor(variable, order=None):
    """Taylor expansion of ln(1+x) around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(1, order + 1):
        index = (n,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        coeffs[index] = np.array([((-1)**(n+1)) / n])
    coeffs[()] = np.array([0.0]) # ln(1) = 0, constant term is 0, index corrected to empty tuple for scalar input
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension

def arctan_taylor(variable, order=None):
    """Taylor expansion of arctan(x) around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(1, order + 1, 2):
        index = (n,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        coeffs[index] = np.array([((-1)**((n-1)/2)) / n])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension

def sinh_taylor(variable, order=None):
    """Taylor expansion of sinh(x) around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(1, order + 1, 2):
        index = (n,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        coeffs[index] = np.array([1.0 / math.factorial(n)])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension

def cosh_taylor(variable, order=None):
    """Taylor expansion of cosh(x) around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(0, order + 1, 2):
        index = (n,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        coeffs[index] = np.array([1.0 / math.factorial(n)])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension

def tanh_taylor(variable, order=None):
    """Taylor expansion of tanh(x) around 0. (Requires higher order for accuracy)"""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    # Bernoulli numbers (B_n), tanh(x) = sum_{n=1}^inf (B_{2n} * 4^n * (4^n - 1) / (2n)!) * x^(2n-1)
    bernoulli_numbers = {2: 1/6, 4: -1/30, 6: 1/42, 8: -1/30, 10: 5/66, 12: -691/2730, 14: 7/6, 16: -3617/510, 18: 43867/798, 20: -174611/330} # Up to B_20
    for n_double in bernoulli_numbers: # n_double represents 2n
        if n_double - 1 <= order:
            n = n_double // 2
            index = (n_double - 1,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
            coeffs[index] = np.array([(bernoulli_numbers[n_double] * (4**n) * (4**n - 1)) / math.factorial(n_double)])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension


def arcsin_taylor(variable, order=None):
    """Taylor expansion of arcsin(x) around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(0, order + 1, 2): # n starts from 0, step 2, so we use (n//2) in formula to get sequence 0, 1, 2, ...
        index = (n+1,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        n_half = n // 2 # We are iterating over 0, 2, 4, ..., so n_half will be 0, 1, 2, ...
        coeffs[index] = np.array([(math.factorial(n) / ((2**n_half) * math.factorial(n_half) * math.factorial(n_half))) * (1 / (n + 1))])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension


def arccos_taylor(variable, order=None):
    """Taylor expansion of arccos(x) around 0. (Using arcsin and pi/2 - arcsin(x) relation)"""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    arcsin_exp = arcsin_taylor(variable, order=order)
    constant_term = MultivariateTaylorFunction(coefficients={(0,): np.array([math.pi/2])}, dimension=1, order=0) # Dimension corrected to 1 for scalar input
    arccos_exp = constant_term - arcsin_exp # arccos(x) = pi/2 - arcsin(x)
    arccos_exp.order = order # Ensure order is set to requested order after operations
    return arccos_exp


def arctanh_taylor(variable, order=None):
    """Taylor expansion of arctanh(x) around 0."""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}
    for n in range(0, order + 1, 2):
        index = (n+1,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        coeffs[index] = np.array([1.0 / (n + 1)])
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension


def gaussian_taylor(variable, order=None, mu=0, sigma=1):
    """Taylor expansion of Gaussian function around mu. (default mu=0, sigma=1)"""
    if order is None:
        order = get_global_max_order()
    else:
        order = min(order, get_global_max_order())
    coeffs = {}

    if sigma <= 0:
        raise ValueError("Sigma (standard deviation) must be positive for Gaussian function.")

    for n in range(order + 1):
        index = (n,) if isinstance(variable, MultivariateTaylorFunction) else (0,) * (variable.dimension if isinstance(variable, Var) else 1) if isinstance(variable, Var) else () # Corrected index
        # Use series expansion for Gaussian around mu. Here we use mu=0 in series formula and shift variable by mu later in composition.
        coeff_val = (1 / (sigma * math.sqrt(2 * math.pi))) * ((-1)**n / (sigma**n * math.factorial(n))) # Coefficient for expansion around 0

        coeffs[index] = np.array([coeff_val])

    exp_around_zero = MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order) # Corrected dimension

    if mu != 0: # Shift expansion point if mu is not zero
        mu_shift_mtf = MultivariateTaylorFunction(coefficients={(1,): np.array([1.0]), (0,): np.array([-mu])}, dimension=1) if variable.dimension == 1 else MultivariateTaylorFunction(coefficients={(0,): np.array([-mu])}, dimension=variable.dimension) #  MTF representing (x - mu) for composition. For multi-dimension, assume shift applies to first variable

        composed_gaussian = exp_around_zero.compose({Var(1, variable.dimension): mu_shift_mtf}) # Compose to shift expansion point to mu
        composed_gaussian.order = order # Ensure order is correct after composition
        return composed_gaussian
    else:
        return exp_around_zero # No shift needed if mu=0