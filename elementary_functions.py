# elementary_functions.py
"""
This module provides implementations of Taylor series expansions for various elementary functions
around zero, using the MultivariateTaylorFunction class.

It includes functions for cosine, sine, exponential, square root (around 1), logarithm (around 1),
arctan, sinh, cosh, tanh, arcsin, arccos, arctanh, and Gaussian functions.

Each function takes a variable (which can be a MultivariateTaylorFunction, Var, or a scalar)
and an optional order for the Taylor expansion.
"""
from taylor_function import MultivariateTaylorFunction, get_global_max_order, convert_to_mtf
from variables import Var  # Import Var class for variable handling
import numpy as np
import math

def cos_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of the cosine function (cos(x)) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the cosine function.
                                                              It can be a MultivariateTaylorFunction object, a Var object, or a scalar.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of cos(variable) as a MultivariateTaylorFunction object.

    Formula:
        cos(x) = sum_{n=0, 2, 4, ...}  (-1)^(n/2) / n! * x^n  = 1 - x^2/2! + x^4/4! - ...
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through even powers up to the specified order (cosine series has only even terms)
    for n in range(0, order + 1, 2):
        # Determine the index for the coefficient.
        # For scalar input or Var, it's based on dimension. For MTF input, it's based on term order.
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n,) # For MTF input, index is a tuple representing the order of the term
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1) # For Var, index is a tuple of zeros with length = dimension
        else: # scalar input
            index = () # For scalar input, index is an empty tuple (representing constant term if n=0, and higher order terms are for variable 'x')

        # Calculate the coefficient for the x^n term in cosine Taylor series
        coefficient_value = np.array([((-1)**(n/2)) / math.factorial(n)])
        coeffs[index] = coefficient_value # Store the coefficient in the dictionary with its index

    # Determine the dimension of the output MTF. Inherit dimension from input variable if it's MTF or Var, otherwise default to 1 for scalar input.
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    # Construct and return the MultivariateTaylorFunction object
    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)


def sin_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of the sine function (sin(x)) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the sine function.
                                                              It can be a MultivariateTaylorFunction object, a Var object, or a scalar.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of sin(variable) as a MultivariateTaylorFunction object.

    Formula:
        sin(x) = sum_{n=1, 3, 5, ...}  (-1)^((n-1)/2) / n! * x^n  = x - x^3/3! + x^5/5! - ...
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through odd powers up to the specified order (sine series has only odd terms)
    for n in range(1, order + 1, 2):
        # Determine the index for the coefficient. Logic is similar to cos_taylor.
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n,)
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        # Calculate the coefficient for the x^n term in sine Taylor series
        coefficient_value = np.array([((-1)**((n-1)/2)) / math.factorial(n)])
        coeffs[index] = coefficient_value # Store the coefficient

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)


def exp_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of the exponential function (exp(x)) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the exponential function.
                                                              It can be a MultivariateTaylorFunction object, a Var object, or a scalar.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of exp(variable) as a MultivariateTaylorFunction object.

    Formula:
        exp(x) = sum_{n=0, 1, 2, ...}  1 / n! * x^n  = 1 + x + x^2/2! + x^3/3! + ...
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through powers from 0 to order (exponential series has all terms)
    for n in range(order + 1):
        # Determine the index for the coefficient. Logic is similar to cos_taylor.
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n,)
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        # Calculate the coefficient for the x^n term in exponential Taylor series
        coefficient_value = np.array([1.0 / math.factorial(n)])
        coeffs[index] = coefficient_value # Store the coefficient

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)


def sqrt_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of sqrt(1+x) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the sqrt function.
                                                              It should represent 'x' in sqrt(1+x).
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of sqrt(1+variable) as a MultivariateTaylorFunction object.

    Formula:
        sqrt(1+x) = 1 + sum_{n=1, 2, 3, ...}  (-1)^(n+1) * (2n-2)! / ((n-1)!^2 * 2^(2n-1)) * x^n
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through powers from 0 to order
    for n in range(order + 1):
        # Determine the index for the coefficient. Logic is similar to cos_taylor.
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n,)
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        # Calculate the coefficient for the x^n term in sqrt(1+x) Taylor series
        if n == 0:
            coefficient_value = np.array([math.sqrt(1)]) # Constant term is sqrt(1) = 1
        elif n >= 1:
            coefficient_value = np.array([((-1)**(n+1) * math.factorial(2*n-2)) / ((math.factorial(n-1)**2) * (2**(2*n-1)) )])
        else: # Should not reach here, but for completeness
            coefficient_value = np.array([0.0])
        coeffs[index] = coefficient_value # Store the coefficient

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)


def log_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of natural logarithm ln(1+x) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the log function.
                                                              It should represent 'x' in ln(1+x).
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of ln(1+variable) as a MultivariateTaylorFunction object.

    Formula:
        ln(1+x) = sum_{n=1, 2, 3, ...}  (-1)^(n+1) / n * x^n  = x - x^2/2 + x^3/3 - ...
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through powers from 1 to order (log series starts from n=1)
    for n in range(1, order + 1):
        # Determine the index for the coefficient. Logic is similar to cos_taylor.
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n,)
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        # Calculate the coefficient for the x^n term in ln(1+x) Taylor series
        coefficient_value = np.array([((-1)**(n+1)) / n])
        coeffs[index] = coefficient_value # Store the coefficient

    coeffs[()] = np.array([0.0]) # ln(1) = 0, constant term is 0 for expansion around 0, index is empty tuple for scalar input

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)


def arctan_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of arctan(x) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the arctan function.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of arctan(variable) as a MultivariateTaylorFunction object.

    Formula:
        arctan(x) = sum_{n=1, 3, 5, ...}  (-1)^((n-1)/2) / n * x^n  = x - x^3/3 + x^5/5 - ...
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through odd powers from 1 to order (arctan series has only odd terms)
    for n in range(1, order + 1, 2):
        # Determine the index for the coefficient. Logic is similar to cos_taylor.
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n,)
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        # Calculate the coefficient for the x^n term in arctan(x) Taylor series
        coefficient_value = np.array([((-1)**((n-1)/2)) / n])
        coeffs[index] = coefficient_value # Store the coefficient

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)


def sinh_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of hyperbolic sine function (sinh(x)) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the sinh function.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of sinh(variable) as a MultivariateTaylorFunction object.

    Formula:
        sinh(x) = sum_{n=1, 3, 5, ...}  1 / n! * x^n  = x + x^3/3! + x^5/5! + ...
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through odd powers from 1 to order (sinh series has only odd terms)
    for n in range(1, order + 1, 2):
        # Determine the index for the coefficient. Logic is similar to cos_taylor.
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n,)
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        # Calculate the coefficient for the x^n term in sinh(x) Taylor series
        coefficient_value = np.array([1.0 / math.factorial(n)])
        coeffs[index] = coefficient_value # Store the coefficient

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)


def cosh_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of hyperbolic cosine function (cosh(x)) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the cosh function.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of cosh(variable) as a MultivariateTaylorFunction object.

    Formula:
        cosh(x) = sum_{n=0, 2, 4, ...}  1 / n! * x^n  = 1 + x^2/2! + x^4/4! + ...
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through even powers from 0 to order (cosh series has only even terms)
    for n in range(0, order + 1, 2):
        # Determine the index for the coefficient. Logic is similar to cos_taylor.
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n,)
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        # Calculate the coefficient for the x^n term in cosh(x) Taylor series
        coefficient_value = np.array([1.0 / math.factorial(n)])
        coeffs[index] = coefficient_value # Store the coefficient

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)


def tanh_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of hyperbolic tangent function (tanh(x)) around 0.
    Note: Taylor series for tanh(x) involves Bernoulli numbers and converges slower, especially for |x| > pi/2.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the tanh function.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of tanh(variable) as a MultivariateTaylorFunction object.

    Formula:
        tanh(x) = sum_{n=1}^inf (B_{2n} * 4^n * (4^n - 1) / (2n)!) * x^(2n-1)  where B_{2n} are Bernoulli numbers.
        We use pre-calculated Bernoulli numbers for terms up to x^19.
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series
    # Bernoulli numbers for tanh(x) Taylor series. Keys are 2n, values are B_{2n}.
    bernoulli_numbers = {
        2: 1/6, 4: -1/30, 6: 1/42, 8: -1/30, 10: 5/66, 12: -691/2730,
        14: 7/6, 16: -3617/510, 18: 43867/798, 20: -174611/330
    } # Bernoulli numbers up to B_20

    # Iterate through Bernoulli numbers to compute tanh coefficients
    for n_double in bernoulli_numbers: # n_double represents 2n in Bernoulli number index B_{2n}
        if n_double - 1 <= order: # Only include terms up to the requested order
            n = n_double // 2 # Calculate n from 2n
            # Determine the index for the coefficient. Logic is similar to cos_taylor.
            if isinstance(variable, MultivariateTaylorFunction):
                index = (n_double - 1,) # tanh series has odd powers, index is (2n-1)
            elif isinstance(variable, Var):
                index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
            else: # scalar input
                index = ()

            # Calculate coefficient using Bernoulli numbers formula
            coefficient_value = np.array([(bernoulli_numbers[n_double] * (4**n) * (4**n - 1)) / math.factorial(n_double)])
            coeffs[index] = coefficient_value # Store the coefficient

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)



def arcsin_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of arcsin(x) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the arcsin function.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of arcsin(variable) as a MultivariateTaylorFunction object.

    Formula:
        arcsin(x) = sum_{n=0, 1, 2, ...}  (2n)! / ((2^n * n!)^2 * (2n+1)) * x^(2n+1)
                  = sum_{n=0, 2, 4, ...}  (n)! / ((2^(n/2) * (n/2)!)^2 * (n+1)) * x^(n+1)  (re-indexed using even n for loop)
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through even powers n, but use index n+1 for arcsin series (odd powers in x)
    for n in range(0, order + 1, 2): # n = 0, 2, 4, ... corresponds to powers 1, 3, 5, ... in arcsin series
        # Determine the index for the coefficient. Logic is similar to cos_taylor, but index is (n+1,)
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n+1,) # arcsin series has odd powers, index is (n+1) because loop iterates n by 2.
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        n_half = n // 2 # Calculate n/2 for the formula, since loop iterates over even n
        # Calculate coefficient for x^(n+1) term
        coefficient_value = np.array([(math.factorial(n) / ((2**n_half) * math.factorial(n_half) * math.factorial(n_half))) * (1 / (n + 1))])
        coeffs[index] = coefficient_value # Store the coefficient

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)


def arccos_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of arccos(x) around 0, using the relation arccos(x) = pi/2 - arcsin(x).

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the arccos function.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of arccos(variable) as a MultivariateTaylorFunction object.

    Relation Used:
        arccos(x) = pi/2 - arcsin(x)
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    arcsin_exp = arcsin_taylor(variable, order=order) # Calculate arcsin Taylor expansion

    # Create a constant MTF for pi/2. Dimension is set to 1 for scalar case, but will be broadcasted if needed in operations.
    constant_term = MultivariateTaylorFunction(coefficients={(0,): np.array([math.pi/2])}, dimension=1, order=0)

    arccos_exp = constant_term - arcsin_exp # Compute arccos as pi/2 - arcsin(x)
    arccos_exp.order = order # Ensure the order of the result is set to the requested order
    return arccos_exp



def arctanh_taylor(variable, order=None):
    """
    Compute the Taylor series expansion of arctanh(x) around 0.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the arctanh function.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of arctanh(variable) as a MultivariateTaylorFunction object.

    Formula:
        arctanh(x) = sum_{n=0, 1, 2, ...}  1 / (2n+1) * x^(2n+1)
                   = sum_{n=0, 2, 4, ...}  1 / (n+1) * x^(n+1)  (re-indexed using even n for loop)
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    # Iterate through even powers n, but use index n+1 for arctanh series (odd powers in x)
    for n in range(0, order + 1, 2): # n = 0, 2, 4, ... corresponds to powers 1, 3, 5, ... in arctanh series
        # Determine the index for the coefficient. Logic is similar to cos_taylor, but index is (n+1,)
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n+1,) # arctanh series has odd powers, index is (n+1) because loop iterates n by 2.
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        # Calculate coefficient for x^(n+1) term
        coefficient_value = np.array([1.0 / (n + 1)])
        coeffs[index] = coefficient_value # Store coefficient

    # Determine dimension similar to cos_taylor
    dimension = 1
    if isinstance(variable, MultivariateTaylorFunction):
        dimension = variable.dimension
    elif isinstance(variable, Var):
        dimension = variable.dimension

    return MultivariateTaylorFunction(coefficients=coeffs, dimension=dimension, order=order)



def gaussian_taylor(variable, order=None, mu=0, sigma=1):
    """
    Compute the Taylor series expansion of the Gaussian function around mu.

    Args:
        variable (MultivariateTaylorFunction | Var | scalar): The input variable for the Gaussian function.
        order (int, optional): The order of the Taylor expansion. If None, it defaults to the global maximum order.
                                 The order will be capped at the global maximum order if it exceeds it.
        mu (float, optional): The mean (center) of the Gaussian function. Defaults to 0.
        sigma (float, optional): The standard deviation of the Gaussian function. Defaults to 1. Must be positive.

    Returns:
        MultivariateTaylorFunction: The Taylor series expansion of Gaussian(variable, mu, sigma) as a MultivariateTaylorFunction object.

    Formula (around mu=0, then shifted):
        Gaussian(x; mu, sigma) = (1 / (sigma * sqrt(2*pi))) * exp(-(x-mu)^2 / (2*sigma^2))
        Expansion around 0 (for simplicity then shift): (1 / (sigma * sqrt(2*pi))) * sum_{n=0}^inf  ((-1)^n / (sigma^n * n!)) * x^n
    """
    # Determine the order of the Taylor expansion
    if order is None:
        order = get_global_max_order() # Use global max order if order is not specified
    else:
        order = min(order, get_global_max_order()) # Cap order at global max order

    coeffs = {} # Dictionary to store coefficients of the Taylor series

    if sigma <= 0:
        raise ValueError("Sigma (standard deviation) must be positive for Gaussian function.")

    # Compute Taylor expansion of Gaussian around 0 first.
    for n in range(order + 1):
        # Determine the index for the coefficient. Logic is similar to cos_taylor.
        if isinstance(variable, MultivariateTaylorFunction):
            index = (n,)
        elif isinstance(variable, Var):
            index = (0,) * (variable.dimension if variable.dimension > 0 else 1)
        else: # scalar input
            index = ()

        # Calculate the coefficient for the x^n term in Gaussian Taylor series (around 0)
        coeff_val = (1 / (sigma * math.sqrt(2 * math.pi))) * ((-1)**n / (sigma**n * math.factorial(n)))
        coeffs[index] = np.array([coeff_val]) # Store the coefficient

    # Create MTF for Gaussian expanded around 0
    exp_around_zero = MultivariateTaylorFunction(coefficients=coeffs, dimension=1 if not isinstance(variable, (MultivariateTaylorFunction, Var)) else variable.dimension if isinstance(variable, Var) else variable.dimension, order=order)

    # Shift the expansion point if mu is not zero to get expansion around mu
    if mu != 0:
        # Create an MTF representing (x - mu) to perform composition for shifting expansion point.
        # If input variable is multi-dimensional, assume shift applies to the first variable (index 1).
        if isinstance(variable, Var) and variable.dimension > 1:
            mu_shift_mtf = MultivariateTaylorFunction(coefficients={(0,): np.array([-mu]), (tuple([1] + [0]*(variable.dimension-1))): np.array([1.0])}, dimension=variable.dimension) # MTF for (x - mu) in multi-dimensional case (shift first var)
        else: # scalar or 1D Var or MTF
            mu_shift_mtf = MultivariateTaylorFunction(coefficients={(0,): np.array([-mu]), (1,): np.array([1.0])}, dimension=1) # MTF for (x - mu) in 1D case

        # Compose exp_around_zero with mu_shift_mtf to shift the expansion point to mu: Gaussian(x-mu; 0, sigma) = Gaussian(x; mu, sigma)
        composed_gaussian = exp_around_zero.compose({Var(1, variable.dimension if isinstance(variable, Var) else 1): mu_shift_mtf}) # Compose to shift expansion point to mu
        composed_gaussian.order = order # Ensure order is correct after composition
        return composed_gaussian # Return shifted Gaussian MTF
    else:
        return exp_around_zero # If mu=0, no shift needed, return expansion around 0
    
