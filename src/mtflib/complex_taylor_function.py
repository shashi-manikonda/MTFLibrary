# mtflib/complex_taylor_function.py
import numpy as np
from collections import defaultdict
from .taylor_function import (initialize_mtf_globals, get_global_max_order,
        get_global_max_dimension, set_global_max_order, set_global_etol,
        get_global_etol, convert_to_mtf, get_mtf_initialized_status,
        MultivariateTaylorFunctionBase)

class ComplexMultivariateTaylorFunction(MultivariateTaylorFunctionBase):
    """
    Represents a multivariate Taylor function with complex coefficients.

    This class extends the concept of MultivariateTaylorFunction to handle
    Taylor expansions where coefficients are complex numbers. It supports
    similar operations as MultivariateTaylorFunction, adapted for complex arithmetic.
    """

    def __init__(self, coefficients, dimension=None, var_name=None, implementation='python'):
        """
        Initializes a ComplexMultivariateTaylorFunction object.
        """
        super().__init__(coefficients, dimension, var_name, implementation)
        if self.coeffs.dtype != np.complex128:
            self.coeffs = self.coeffs.astype(np.complex128)

    @classmethod
    def from_constant(cls, constant_value, dimension=None, implementation='python'):
        """
        Creates a ComplexMultivariateTaylorFunction representing a constant value.
        """
        if dimension is None:
            dimension = get_global_max_dimension()
        constant_value_complex = np.array([complex(constant_value)], dtype=np.complex128)
        coeffs = {(0,) * dimension: constant_value_complex}
        return cls(coefficients=coeffs, dimension=dimension, implementation=implementation)

    @classmethod
    def from_variable(cls, var_index, dimension, implementation='python'):
        """
        Creates a ComplexMultivariateTaylorFunction representing a single variable.
        """
        if not (1 <= var_index <= dimension):
            raise ValueError(f"Variable index must be in range [1, dimension], got {var_index} for dimension {dimension}.")

        exponent = [0] * dimension
        exponent[var_index - 1] = 1
        coeffs = {tuple(exponent): np.array([1.0 + 0.0j], dtype=np.complex128)}
        return cls(coefficients=coeffs, dimension=dimension, implementation=implementation)

    def conjugate(self):
        """
        Returns the complex conjugate of the ComplexMultivariateTaylorFunction.
        """
        return ComplexMultivariateTaylorFunction((self.exponents, np.conjugate(self.coeffs)), self.dimension)

    def real_part(self):
        """
        Returns the real part of the ComplexMultivariateTaylorFunction as a MultivariateTaylorFunction.
        """
        return MultivariateTaylorFunctionBase((self.exponents, np.real(self.coeffs)), self.dimension)

    def imag_part(self):
        """
        Returns the imaginary part of the ComplexMultivariateTaylorFunction as a MultivariateTaylorFunction.
        """
        return MultivariateTaylorFunctionBase((self.exponents, np.imag(self.coeffs)), self.dimension)

    def __repr__(self):
        """Returns a detailed string representation of the MTF (for debugging)."""
        df = self.get_tabular_dataframe()
        return f'{df}\n'

    def __str__(self):
        if self.var_name: # Use var_name if available for concise representation
            return f"ComplexMultivariateTaylorFunction({self.var_name})"
        df = self.get_tabular_dataframe()
        return f'\n{df}'

    def magnitude(self):
        """
        Raises NotImplementedError as magnitude is not directly representable as a CMTF.
        """
        raise NotImplementedError("Magnitude of a ComplexMultivariateTaylorFunction is generally not a ComplexMultivariateTaylorFunction.")

    def phase(self):
        """
        Raises NotImplementedError as phase is not directly representable as a CMTF.
        """
        raise NotImplementedError("Phase of a ComplexMultivariateTaylorFunction is generally not a ComplexMultivariateTaylorFunction.")

def _generate_exponent_combinations(dimension, order):
    """
    Generates all combinations of exponents for a given dimension and order.

    Args:
        dimension (int): The dimension of the multivariate function.
        order (int): The maximum order of terms to generate exponents for.

    Returns:
        list of tuples: A list of exponent tuples.
    """
    if dimension <= 0 or order < 0:
        return []
    if dimension == 1:
        return [(o,) for o in range(order + 1)]
    if order == 0:
        return [(0,) * dimension]

    exponent_combinations = []
    for o in range(order + 1):
        for comb in _generate_exponent_combinations(dimension - 1, o):
            remaining_order = order - o
            for i in range(remaining_order + 1):
                exponent_combinations.append(comb + (i,))
    return exponent_combinations


def _add_coefficient_dicts(dict1, dict2, subtract=False):
    """
    Adds two coefficient dictionaries.

    Args:
        dict1 (defaultdict): First coefficient dictionary.
        dict2 (defaultdict): Second coefficient dictionary.
        subtract (bool, optional): If True, subtract dict2 from dict1. Defaults to False (add).

    Returns:
        defaultdict: A new coefficient dictionary with the sum (or difference) of coefficients.
    """
    sum_coeffs = defaultdict(lambda: np.array([0.0j]).reshape(1) if any(isinstance(coeff[0], complex) for coeff in dict1.values()) or any(isinstance(coeff[0], complex) for coeff in dict2.values()) else np.array([0.0]).reshape(1)) #Default to complex 0 if either dict is complex
    for exponents in set(dict1.keys()) | set(dict2.keys()): # Iterate over all unique exponents
        coeff1 = dict1.get(exponents, sum_coeffs.default_factory()) # Get coeff1, default to zero array if missing
        coeff2 = dict2.get(exponents, sum_coeffs.default_factory()) # Get coeff2, default to zero array if missing
        if subtract:
            sum_coeffs[exponents] = np.array(coeff1).flatten() - np.array(coeff2).flatten() # Flatten for subtraction
        else:
            sum_coeffs[exponents] = np.array(coeff1).flatten() + np.array(coeff2).flatten() # Flatten for addition
    return sum_coeffs


def convert_to_cmtf(variable):
    """
    Converts a variable into a ComplexMultivariateTaylorFunction object.

    Handles input types:
    - ComplexMultivariateTaylorFunction: Returns the input as is.
    - MultivariateTaylorFunction: Converts to ComplexMultivariateTaylorFunction.
    - Scalar (int, float, complex, np.number): Creates a constant ComplexMultivariateTaylorFunction.

    Args:
        variable: The variable to convert. Can be a ComplexMultivariateTaylorFunction,
                  MultivariateTaylorFunction, or a scalar.

    Returns:
        ComplexMultivariateTaylorFunction: The converted ComplexMultivariateTaylorFunction object.
    """
    if isinstance(variable, ComplexMultivariateTaylorFunction):
        return variable  # Already a CMTF, return as is
    elif isinstance(variable, MultivariateTaylorFunctionBase): # Var function returns MTF now
        # Convert MTF to CMTF: just need to ensure coefficients are complex type
        exponents = variable.exponents
        coeffs = variable.coeffs.astype(np.complex128)
        return ComplexMultivariateTaylorFunction((exponents, coeffs), variable.dimension)
    elif isinstance(variable, (int, float, complex, np.number)):
        # Create constant CMTF from scalar
        dim = 1
        if hasattr(variable, 'dimension'):
            dim = variable.dimension
        return ComplexMultivariateTaylorFunction.from_constant(variable, dimension=dim)
    else:
        raise TypeError("Unsupported type for conversion to ComplexMultivariateTaylorFunction.")
