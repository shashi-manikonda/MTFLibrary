import numpy as np
import math # ADD THIS LINE
from taylor_function import MultivariateTaylorFunction
from variables import Var
from taylor_operations import set_global_max_order

def cos_taylor(input_value, order):
    """Taylor expansion of cosine function around 0."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        if n % 2 == 0:
            index = (n,) * dimension
            coefficients[index] = np.array([((-1)**(n/2)) / math.factorial(n)]) # use math.factorial
    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, order=order)

def sin_taylor(input_value, order):
    """Taylor expansion of sine function around 0."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        if n % 2 != 0:
            index = (n,) * dimension
            coefficients[index] = np.array([((-1)**((n-1)/2)) / math.factorial(n)]) # use math.factorial
    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, order=order)

def exp_taylor(input_value, order):
    """Taylor expansion of exponential function around 0."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        coefficients[index] = np.array([1.0 / math.factorial(n)]) # use math.factorial
    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, order=order)

def gaussian_taylor(input_value, order, mean=0.0, std_dev=1.0):
    """Taylor expansion of Gaussian function around mean."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        if n % 2 == 0:
            coefficients[index] = np.array([(math.factorial(n) / (math.factorial(n//2) * (2**(n//2)))) * ((-1)**(n/2)) / (std_dev**n) * (1/(std_dev * np.sqrt(2*np.pi)))]) # use math.factorial
        else:
            coefficients[index] = np.array([0.0])
    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, expansion_point=np.array([mean]), order=order)

def sqrt_taylor(input_value, order):
    """Taylor expansion of sqrt function around 1."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        if n == 0:
            coefficients[index] = np.array([np.sqrt(1)])
        elif n == 1:
            coefficients[index] = np.array([1/2])
        elif n == 2:
            coefficients[index] = np.array([-1/8])
        elif n == 3:
            coefficients[index] = np.array([1/16])
        elif n == 4:
            coefficients[index] = np.array([-5/128])
        elif n >= 5:
             coefficients[index] = np.array([np.nan]) # Series becomes more complex, needs general term if higher orders needed.
    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, expansion_point=np.ones(dimension), order=order)


def log_taylor(input_value, order):
    """Taylor expansion of natural logarithm function around 1."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        if n == 0:
            coefficients[index] = np.array([np.log(1)]) # ln(1) = 0
        elif n >= 1:
            coefficients[index] = np.array([((-1)**(n+1)) / n]) # Coefficient for ln(1+x) Taylor series (around 1, it's ln(x) around 1, which is ln(1+(x-1)))

    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, expansion_point=np.ones(dimension), order=order)


def arctan_taylor(input_value, order):
    """Taylor expansion of arctan function around 0."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        if n % 2 != 0:
            coefficients[index] = np.array([((-1)**((n-1)/2)) / n])

    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, expansion_point=np.zeros(dimension), order=order)

def sinh_taylor(input_value, order):
    """Taylor expansion of hyperbolic sine function around 0."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        if n % 2 != 0:
            coefficients[index] = np.array([1.0 / math.factorial(n)]) # use math.factorial

    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, expansion_point=np.zeros(dimension), order=order)

def cosh_taylor(input_value, order):
    """Taylor expansion of hyperbolic cosine function around 0."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        if n % 2 == 0:
            coefficients[index] = np.array([1.0 / math.factorial(n)]) # use math.factorial

    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, order=order)

def tanh_taylor(input_value, order):
    """Taylor expansion of hyperbolic tangent function around 0."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        if n == 1:
            coefficients[index] = np.array([1.0])
        elif n == 3:
            coefficients[index] = np.array([-1/3])
        elif n == 5:
            coefficients[index] = np.array([2/15])
        elif n == 7:
            coefficients[index] = np.array([-17/315])
        elif n >= 9:
            coefficients[index] = np.array([np.nan]) # Series becomes complex

    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, order=order)


def arcsin_taylor(input_value, order):
    """Taylor expansion of arcsin function around 0."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        if n % 2 != 0:
            if n == 1:
                coefficients[index] = np.array([1.0])
            elif n == 3:
                coefficients[index] = np.array([1/6])
            elif n == 5:
                coefficients[index] = np.array([3/40])
            elif n == 7:
                coefficients[index] = np.array([5/112])
            elif n >= 9:
                coefficients[index] = np.array([np.nan]) # Series becomes complex

    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, order=order)

def arccos_taylor(input_value, order):
    """Taylor expansion of arccos function around 0. Using arccos(x) = pi/2 - arcsin(x)"""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")

    arcsin_mtf = arcsin_taylor(input_value, order)
    constant_pi_over_2 = MultivariateTaylorFunction.from_constant(np.pi/2.0, arcsin_mtf.dimension)
    return constant_pi_over_2 - arcsin_mtf


def arctanh_taylor(input_value, order):
    """Taylor expansion of arctanh function around 0."""
    if not isinstance(input_value, (MultivariateTaylorFunction, Var, int, float)):
        raise TypeError("Input must be Var, MultivariateTaylorFunction, int, or float.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    dimension = _get_dimension(input_value)
    coefficients = {}
    for n in range(order + 1):
        index = (n,) * dimension
        if n % 2 != 0:
            coefficients[index] = np.array([1.0 / n])

    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, order=order)



# --- Helper functions ---
def _get_dimension(input_value):
    """Helper to get dimension from input."""
    if isinstance(input_value, Var):
        return input_value.dimension
    elif isinstance(input_value, MultivariateTaylorFunction):
        return input_value.dimension
    else:
        return 1 # default dimension for scalar input

def _input_to_mtf(input_value, dimension):
    """Helper to convert input to MTF if necessary."""
    if isinstance(input_value, MultivariateTaylorFunction):
        return input_value
    elif isinstance(input_value, Var):
        return input_value._create_taylor_function_from_var()
    else: # int or float
        return scalar_to_mtf(input_value, dimension)

def scalar_to_mtf(scalar, dimension):
    """Convert a scalar value to a constant MultivariateTaylorFunction."""
    return MultivariateTaylorFunction.from_constant(scalar, dimension)

def _composition_dict(input_value, mtf_input):
    """Helper to create composition dictionary if input is Var."""
    if isinstance(input_value, Var):
        return {input_value: mtf_input}
    return {} # Empty dict for scalar or MTF inputs


