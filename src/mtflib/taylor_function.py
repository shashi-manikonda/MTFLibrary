# mtflib/taylor_function.py
import numpy as np
from collections import defaultdict
import math
from .elementary_coefficients import load_precomputed_coefficients
import time
import numbers
import pandas as pd

from . import elementary_coefficients

# Try to import the C++ backend
try:
    from .backends.cpp import mtf_cpp
    _CPP_BACKEND_AVAILABLE = True
except ImportError:
    _CPP_BACKEND_AVAILABLE = False


_GLOBAL_MAX_ORDER = None
_GLOBAL_MAX_DIMENSION = None
_INITIALIZED = False
_DEFAULT_ETOL = 1e-16
_TRUNCATE_AFTER_OPERATION = True
precomputed_coefficients = {}

def get_max_coefficient_count(max_order=None, max_dimension=None):
    """Calculates max coefficient count for given order/dimension."""
    effective_max_order = max_order if max_order is not None else _GLOBAL_MAX_ORDER
    effective_max_dimension = max_dimension if max_dimension is not None else _GLOBAL_MAX_DIMENSION
    if effective_max_order is None or effective_max_dimension is None:
        raise ValueError("Global max_order or max_dimension not initialized and no defaults provided.")
    return math.comb(effective_max_order + effective_max_dimension, effective_max_dimension)

def initialize_mtf_globals(max_order=None, max_dimension=None):
    """Initializes global settings and loads precomputed coefficients. Must be called once."""
    global _GLOBAL_MAX_ORDER, _GLOBAL_MAX_DIMENSION, _INITIALIZED, precomputed_coefficients
    if _INITIALIZED:
        raise RuntimeError("MTF globals already initialized. Cannot re-initialize.")
    if max_order is not None:
        if not isinstance(max_order, int) or max_order <= 0:
            raise ValueError("max_order must be a positive integer.")
        _GLOBAL_MAX_ORDER = max_order
    if max_dimension is not None:
        if not isinstance(max_dimension, int) or max_dimension <= 0:
            raise ValueError("max_dimension must be a positive integer.")
        _GLOBAL_MAX_DIMENSION = max_dimension
    print(f"Initializing MTF globals with: _GLOBAL_MAX_ORDER={_GLOBAL_MAX_ORDER}, _GLOBAL_MAX_DIMENSION={_GLOBAL_MAX_DIMENSION}")
    precomputed_coefficients_dict = load_precomputed_coefficients(max_order_config=_GLOBAL_MAX_ORDER)
    precomputed_coefficients = precomputed_coefficients_dict
    _INITIALIZED = True
    print(f"MTF globals initialized: _GLOBAL_MAX_ORDER={_GLOBAL_MAX_ORDER}, _GLOBAL_MAX_DIMENSION={_GLOBAL_MAX_DIMENSION}, _INITIALIZED={_INITIALIZED}")
    print(f"Max coefficient count (order={_GLOBAL_MAX_ORDER}, nvars={_GLOBAL_MAX_DIMENSION}): {get_max_coefficient_count()}")
    print(f"Precomputed coefficients loaded and ready for use.")


# Add this function to expose the precomputed coefficients
def get_precomputed_coefficients():
    """Returns the precomputed Taylor coefficients for elementary functions."""
    global precomputed_coefficients, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals are not initialized.")
    return precomputed_coefficients

def get_mtf_initialized_status():
    """Returns initialization status of MTF globals."""
    return _INITIALIZED

def set_global_max_order(order):
    """Sets the global maximum order for Taylor series."""
    global _GLOBAL_MAX_ORDER, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals must be initialized before setting max order.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    _GLOBAL_MAX_ORDER = order

def get_global_max_order():
    """Returns the global maximum order for Taylor series."""
    global _GLOBAL_MAX_ORDER, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals are not initialized.")
    return _GLOBAL_MAX_ORDER

def get_global_max_dimension():
    """Returns the global maximum dimension (number of variables)."""
    global _GLOBAL_MAX_DIMENSION, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals are not initialized.")
    return _GLOBAL_MAX_DIMENSION

def set_global_etol(etol):
    """Sets the global error tolerance (etol) for `mtflib`."""
    global _DEFAULT_ETOL, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals must be initialized before setting error tolerance.")
    if not isinstance(etol, float) or etol <= 0:
        raise ValueError("Error tolerance (etol) must be a positive float.")
    _DEFAULT_ETOL = etol

def get_global_etol():
    """Returns the global error tolerance (etol)."""
    global _DEFAULT_ETOL, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals are not initialized.")
    return _DEFAULT_ETOL

def set_truncate_after_operation(enable: bool):
    """
    Sets the global flag to enable or disable automatic coefficient cleanup.
    """
    global _TRUNCATE_AFTER_OPERATION
    if not isinstance(enable, bool):
        raise ValueError("Input 'enable' must be a boolean value (True or False).")
    _TRUNCATE_AFTER_OPERATION = enable

def _generate_exponent(order, var_index, dimension):
    """Generates an exponent tuple for a monomial term."""
    exponent = [0] * dimension
    exponent[var_index] = order
    return tuple(exponent)

class MultivariateTaylorFunctionBase:
    """Represents a multivariate Taylor function."""
    def __init__(self, coefficients, dimension=None, var_name=None, implementation='cpp'):
        """
        Initializes a MultivariateTaylorFunction object.

        :param coefficients: A dictionary mapping exponent tuples to coefficient values,
                             or a tuple of (exponents, coeffs) numpy arrays.
        :param dimension: The number of variables in the function.
        :param var_name: The name of the variable (optional).
        :param implementation: The backend to use for calculations ('python', 'cpp', or 'cython').
        """
        self.var_name = var_name
        if implementation == 'cpp' and not _CPP_BACKEND_AVAILABLE:
            # Fallback to python if cpp is not available
            self.implementation = 'python'
        else:
            self.implementation = implementation

        # Fast path for tuple of (exponents, coeffs)
        if isinstance(coefficients, tuple) and len(coefficients) == 2 and isinstance(coefficients[0], np.ndarray) and isinstance(coefficients[1], np.ndarray):
            self.exponents, self.coeffs = coefficients
            if dimension is None:
                self.dimension = self.exponents.shape[1] if self.exponents.size > 0 else get_global_max_dimension()
            else:
                self.dimension = dimension

            if self.exponents.size > 0 and self.exponents.shape[1] != self.dimension:
                 raise ValueError(f"Provided dimension {self.dimension} does not match exponent dimension {self.exponents.shape[1]}.")
            if self.coeffs.ndim != 1 or self.coeffs.shape[0] != self.exponents.shape[0]:
                raise ValueError("Coefficients array has incorrect shape.")

        # Path for dictionary
        elif isinstance(coefficients, dict):
            if not coefficients:
                self.dimension = dimension if dimension is not None else get_global_max_dimension()
                self.exponents = np.empty((0, self.dimension), dtype=np.int32)
                self.coeffs = np.empty((0,), dtype=np.float64)
            else:
                first_exp = next(iter(coefficients.keys()))
                inferred_dim = len(first_exp)
                if dimension is None:
                    self.dimension = inferred_dim
                elif dimension != inferred_dim:
                    raise ValueError(f"Provided dimension {dimension} does not match inferred dimension {inferred_dim} from coefficients.")
                else:
                    self.dimension = dimension

                # Optimized dict conversion
                num_items = len(coefficients)
                self.exponents = np.empty((num_items, self.dimension), dtype=np.int32)
                is_complex = any(np.iscomplexobj(v) for v in coefficients.values())
                dtype = np.complex128 if is_complex else np.float64
                self.coeffs = np.empty(num_items, dtype=dtype)

                for i, (exp, coeff) in enumerate(coefficients.items()):
                    self.exponents[i] = exp
                    self.coeffs[i] = coeff

                # Sort both arrays based on exponents
                sorted_indices = np.lexsort(self.exponents.T)
                self.exponents = self.exponents[sorted_indices]
                self.coeffs = self.coeffs[sorted_indices]
        else:
            raise TypeError("Unsupported type for 'coefficients'. Must be a dict or a tuple of (exponents, coeffs) arrays.")

    @classmethod
    def from_constant(cls, constant_value, dimension=None, implementation='python'):
        """Creates an MTF representing a constant value."""
        if dimension is None:
            dimension=get_global_max_dimension()
        # Ensure the value is a scalar float, not a numpy array
        coeffs = {(0,) * dimension: float(constant_value)}
        return cls(coefficients=coeffs, dimension=dimension, implementation=implementation)

    @classmethod
    def from_variable(cls, var_index, dimension, implementation='python'):
        """Creates an MTF representing a single variable."""
        if not (1 <= var_index <= dimension):
            raise ValueError(f"Variable index must be between 1 and {dimension}, inclusive.")
        exponent = [0] * dimension
        exponent[var_index - 1] = 1
        # Use a scalar float for the coefficient
        coeffs = {tuple(exponent): 1.0}
        return cls(coefficients=coeffs, dimension=dimension, var_name=f"x_{var_index}", implementation=implementation)

    def __call__(self, evaluation_point):
        """Alias for `eval()`."""
        return self.eval(evaluation_point)

    def eval(self, evaluation_point):
        """Evaluates the MTF at a given evaluation point."""
        if len(evaluation_point) != self.dimension:
            raise ValueError(f"Evaluation point dimension must match MTF dimension ({self.dimension}).")
        evaluation_point = np.array(evaluation_point)

        if self.coeffs.size == 0:
            return np.array([0.0]).reshape(1)

        # Optimized evaluation using np.power and np.einsum
        term_values = np.prod(np.power(evaluation_point, self.exponents), axis=1)
        result = np.einsum('i,i->', self.coeffs, term_values)
        return np.array([result])

    def __add__(self, other):
        """Defines addition (+) for MultivariateTaylorFunction objects."""
        if isinstance(other, (int, float, complex, np.number)):
            if isinstance(other, complex):
                from .complex_taylor_function import ComplexMultivariateTaylorFunction
                const_mtf = ComplexMultivariateTaylorFunction.from_constant(other, dimension=self.dimension, implementation=self.implementation)
                return self + const_mtf
            else: # int or float
                const_mtf = type(self).from_constant(other, dimension=self.dimension, implementation=self.implementation)
                return self + const_mtf

        if not isinstance(other, MultivariateTaylorFunctionBase):
            return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for addition.")

        # C++ Backend Dispatch
        if self.implementation == 'cpp' and _CPP_BACKEND_AVAILABLE and not (np.iscomplexobj(self.coeffs) or np.iscomplexobj(other.coeffs)):
            new_exps, new_coeffs = mtf_cpp.add_mtf_cpp(self.exponents, self.coeffs, other.exponents, other.coeffs)
            result_mtf = type(self)((new_exps, new_coeffs), self.dimension, implementation='cpp')
            if _TRUNCATE_AFTER_OPERATION:
                result_mtf._cleanup_after_operation()
            return result_mtf

        # Python Implementation (Optimized with dictionary)
        is_complex = np.iscomplexobj(self.coeffs) or np.iscomplexobj(other.coeffs)
        summed_coeffs_dict = defaultdict(complex) if is_complex else defaultdict(float)

        for i in range(self.coeffs.shape[0]):
            exp_tuple = tuple(self.exponents[i])
            summed_coeffs_dict[exp_tuple] += self.coeffs[i]

        for i in range(other.coeffs.shape[0]):
            exp_tuple = tuple(other.exponents[i])
            summed_coeffs_dict[exp_tuple] += other.coeffs[i]

        if not summed_coeffs_dict:
            unique_exponents = np.empty((0, self.dimension), dtype=np.int32)
            summed_coeffs = np.empty((0,), dtype=np.complex128 if is_complex else np.float64)
        else:
            unique_exponents = np.array(list(summed_coeffs_dict.keys()), dtype=np.int32)
            summed_coeffs = np.array(list(summed_coeffs_dict.values()), dtype=np.complex128 if is_complex else np.float64)

        result_mtf = type(self)((unique_exponents, summed_coeffs), self.dimension, implementation=self.implementation)
        if _TRUNCATE_AFTER_OPERATION:
            result_mtf._cleanup_after_operation()
        return result_mtf

    def __radd__(self, other):
        """Defines reverse addition for commutative property."""
        return self.__add__(other)

    def __sub__(self, other):
        """Defines subtraction (-) for MultivariateTaylorFunction objects."""
        if isinstance(other, (int, float, complex, np.number)):
            return self + (-other)

        if not isinstance(other, MultivariateTaylorFunctionBase):
            return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for subtraction.")

        # C++ Backend Dispatch
        if self.implementation == 'cpp' and _CPP_BACKEND_AVAILABLE and not (np.iscomplexobj(self.coeffs) or np.iscomplexobj(other.coeffs)):
            # C++ add can handle subtraction by negating the second operand's coeffs
            new_exps, new_coeffs = mtf_cpp.add_mtf_cpp(self.exponents, self.coeffs, other.exponents, -other.coeffs)
            result_mtf = type(self)((new_exps, new_coeffs), self.dimension, implementation='cpp')
            if _TRUNCATE_AFTER_OPERATION:
                result_mtf._cleanup_after_operation()
            return result_mtf

        # Python Implementation (Optimized with dictionary)
        is_complex = np.iscomplexobj(self.coeffs) or np.iscomplexobj(other.coeffs)
        summed_coeffs_dict = defaultdict(complex) if is_complex else defaultdict(float)

        for i in range(self.coeffs.shape[0]):
            exp_tuple = tuple(self.exponents[i])
            summed_coeffs_dict[exp_tuple] += self.coeffs[i]

        for i in range(other.coeffs.shape[0]):
            exp_tuple = tuple(other.exponents[i])
            summed_coeffs_dict[exp_tuple] -= other.coeffs[i]

        if not summed_coeffs_dict:
            unique_exponents = np.empty((0, self.dimension), dtype=np.int32)
            summed_coeffs = np.empty((0,), dtype=np.complex128 if is_complex else np.float64)
        else:
            unique_exponents = np.array(list(summed_coeffs_dict.keys()), dtype=np.int32)
            summed_coeffs = np.array(list(summed_coeffs_dict.values()), dtype=np.complex128 if is_complex else np.float64)

        result_mtf = type(self)((unique_exponents, summed_coeffs), self.dimension, implementation=self.implementation)
        if _TRUNCATE_AFTER_OPERATION:
            result_mtf._cleanup_after_operation()
        return result_mtf

    def __rsub__(self, other):
        """Defines reverse subtraction for non-commutative property."""
        return -(self - other)

    def __mul__(self, other):
        """Defines multiplication (*) for MultivariateTaylorFunction objects."""
        if isinstance(other, (int, float, complex, np.number)):
            # Scalar multiplication
            if self.coeffs.size == 0:
                return self.copy()
            return type(self)((self.exponents.copy(), self.coeffs * other), self.dimension, implementation=self.implementation)

        if not isinstance(other, MultivariateTaylorFunctionBase):
            return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for multiplication.")

        if self.coeffs.size == 0 or other.coeffs.size == 0:
            dtype = np.result_type(self.coeffs.dtype, other.coeffs.dtype)
            return type(self)((np.empty((0, self.dimension), dtype=np.int32), np.empty((0,), dtype=dtype)), self.dimension)

        # C++ Backend Dispatch
        if self.implementation == 'cpp' and _CPP_BACKEND_AVAILABLE and not (np.iscomplexobj(self.coeffs) or np.iscomplexobj(other.coeffs)):
            new_exps, new_coeffs = mtf_cpp.multiply_mtf_cpp(self.exponents, self.coeffs, other.exponents, other.coeffs)
            result_mtf = type(self)((new_exps, new_coeffs), self.dimension, implementation='cpp')
            if _TRUNCATE_AFTER_OPERATION:
                result_mtf._cleanup_after_operation()
            return result_mtf

        # Python Implementation (Optimized with dictionary)
        is_complex = np.iscomplexobj(self.coeffs) or np.iscomplexobj(other.coeffs)
        summed_coeffs_dict = defaultdict(complex) if is_complex else defaultdict(float)

        for i in range(self.coeffs.shape[0]):
            exp1 = self.exponents[i]
            coeff1 = self.coeffs[i]
            for j in range(other.coeffs.shape[0]):
                exp2 = other.exponents[j]
                coeff2 = other.coeffs[j]
                new_exp = tuple(exp1 + exp2)
                summed_coeffs_dict[new_exp] += coeff1 * coeff2

        if not summed_coeffs_dict:
            unique_exponents = np.empty((0, self.dimension), dtype=np.int32)
            summed_coeffs = np.empty((0,), dtype=np.complex128 if is_complex else np.float64)
        else:
            unique_exponents = np.array(list(summed_coeffs_dict.keys()), dtype=np.int32)
            summed_coeffs = np.array(list(summed_coeffs_dict.values()), dtype=np.complex128 if is_complex else np.float64)

        result_mtf = type(self)((unique_exponents, summed_coeffs), self.dimension, implementation=self.implementation)
        if _TRUNCATE_AFTER_OPERATION:
            result_mtf._cleanup_after_operation()
        return result_mtf

    def __rmul__(self, other):
        """Defines reverse multiplication for commutative property."""
        return self.__mul__(other)

    def __imul__(self, other):
        """Defines in-place multiplication (*=) with a scalar."""
        if isinstance(other, (int, float, np.number)):
            self.coeffs *= other
            return self
        else:
            return NotImplemented

    def __pow__(self, power):
        """Defines exponentiation (**) for MultivariateTaylorFunction objects."""
        if isinstance(power, numbers.Integral):
            if power < 0:
                if power == -1:
                    return self._inv_mtf_internal(self)
                else:
                    raise ValueError("Power must be a non-negative integer, 0.5, or -0.5.")
            if power == 0:
                return type(self).from_constant(1.0, dimension=self.dimension, implementation=self.implementation)
            if power == 1:
                return self

            # Optimized power using binary exponentiation (exponentiation by squaring)
            result = type(self).from_constant(1.0, dimension=self.dimension, implementation=self.implementation)
            base = self
            while power > 0:
                if power % 2 == 1:
                    result *= base
                base *= base
                power //= 2
            return result

        elif isinstance(power, float):
            if power == 0.5:
                return sqrt_taylor(self)
            elif power == -0.5:
                return isqrt_taylor(self)
            else:
                raise ValueError("Power must be a non-negative integer, 0.5, or -0.5.")
        else:
            raise ValueError("Power must be a non-negative integer, 0.5, or -0.5.")

    def __neg__(self):
        """Defines negation (-) for MultivariateTaylorFunction objects."""
        return type(self)((self.exponents.copy(), -self.coeffs.copy()), self.dimension)

    def __truediv__(self, other):
        """Defines division (/) for MultivariateTaylorFunction objects."""
        if isinstance(other, MultivariateTaylorFunctionBase):
            inverse_other_mtf = self._inv_mtf_internal(other)
            return self * inverse_other_mtf
        elif isinstance(other, (int, float, np.number)):
            result_mtf = type(self)((self.exponents.copy(), self.coeffs / other), self.dimension, implementation=self.implementation)
            if _TRUNCATE_AFTER_OPERATION:
                result_mtf._cleanup_after_operation()
            return result_mtf
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """Defines reverse division for scalar divided by MTF."""
        if isinstance(other, (int, float, np.number)):
            inverse_self_mtf = self._inv_mtf_internal(self)
            return inverse_self_mtf * other
        else:
            return NotImplemented

    def _inv_mtf_internal(self, mtf_instance, order=None):
        """Internal method to calculate Taylor expansion of 1/mtf_instance."""
        if order is None:
            order = get_global_max_order()
        constant_term_coeff = mtf_instance.extract_coefficient(tuple([0] * mtf_instance.dimension))
        c0 = constant_term_coeff.item()
        if abs(c0) < get_global_etol():
            raise ValueError("Cannot invert MTF with zero constant term (or very close to zero).")
        rescaled_mtf = mtf_instance / c0
        inverse_coefficients = precomputed_coefficients.get('inverse')
        if inverse_coefficients is None:
            raise RuntimeError("Precomputed 'inverse' coefficients not loaded.")
        coeffs_to_use = inverse_coefficients[:order+1]
        coeff_items = []
        for i, coeff_val in enumerate(coeffs_to_use):
            exponent_tuple = (i,)
            coeff_items.append((exponent_tuple, coeff_val))
        inverse_series_1d_mtf = type(self)(coefficients=dict(coeff_items), dimension=1)
        composed_mtf = inverse_series_1d_mtf.compose_one_dim(
            rescaled_mtf - type(self).from_constant(1.0, dimension=rescaled_mtf.dimension, implementation=rescaled_mtf.implementation)
        )
        final_mtf = composed_mtf / c0
        truncated_mtf = final_mtf.truncate(order)
        return truncated_mtf

    def substitute_variable(self, var_index, value):
        """Substitutes a variable in the MTF with a numerical value."""
        if not isinstance(var_index, int):
            raise TypeError("var_index must be an integer dimension index (1-based).")
        if not (1 <= var_index <= self.dimension):
            raise ValueError(f"var_index must be between 1 and {self.dimension}, inclusive.")
        if not isinstance(value, (int, float, complex, np.number)):
            raise TypeError("value must be a number.")

        new_coeffs = self.coeffs.copy()
        new_exponents = self.exponents.copy()

        dim_exponents = new_exponents[:, var_index - 1]

        # Calculate the multipliers for each term
        multipliers = np.power(value, dim_exponents)

        # Apply the multipliers
        new_coeffs *= multipliers

        # Zero out the exponents for the substituted variable
        new_exponents[:, var_index - 1] = 0

        # Use np.unique to group the new exponents and sum the corresponding coefficients
        unique_exponents, inverse_indices = np.unique(new_exponents, axis=0, return_inverse=True)
        summed_coeffs = np.bincount(inverse_indices, weights=new_coeffs)

        return type(self)((unique_exponents, summed_coeffs), self.dimension, implementation=self.implementation)

    def truncate_inplace(self, order=None):
        """Truncates the MTF *in place* to a specified order."""
        if order is None:
            order = get_global_max_order()
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")

        if self.exponents.shape[0] == 0:
            # self._max_order = 0
            return self

        term_orders = np.sum(self.exponents, axis=1)
        keep_mask = term_orders <= order

        self.exponents = self.exponents[keep_mask]
        self.coeffs = self.coeffs[keep_mask]

        # if self.coeffs.shape[0] > 0:
        #     self._max_order = np.max(np.sum(self.exponents, axis=1))
        # else:
        #     self._max_order = 0

        return self

    def truncate(self, order=None):
        """Truncates the MultivariateTaylorFunction to a specified order."""
        etol = get_global_etol()
        if order is None:
            order = get_global_max_order()
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")

        if self.coeffs.size == 0:
            return self.copy()

        term_orders = np.sum(self.exponents, axis=1)
        order_mask = term_orders <= order

        etol_mask = np.abs(self.coeffs) > etol

        keep_mask = np.logical_and(order_mask, etol_mask)

        new_exponents = self.exponents[keep_mask]
        new_coeffs = self.coeffs[keep_mask]

        return type(self)((new_exponents, new_coeffs), self.dimension, implementation=self.implementation)

    def substitute_variable_inplace(self, var_dimension, value):
        """Substitutes a variable in the MTF with a numerical value IN-PLACE."""
        if not isinstance(var_dimension, int) or not (1 <= var_dimension <= self.dimension):
            raise ValueError("Invalid var_dimension.")
        if not isinstance(value, (int, float, complex, np.number)):
            raise TypeError("Value must be a number.")

        new_self = self.substitute_variable(var_dimension, value)
        self.exponents = new_self.exponents
        self.coeffs = new_self.coeffs

    def is_zero_mtf(self, mtf, zero_tolerance=None):
        """Checks if an MTF is effectively zero."""
        if zero_tolerance is None:
            zero_tolerance = get_global_etol()
        if mtf.coeffs.size == 0:
            return True
        return np.all(np.abs(mtf.coeffs) < zero_tolerance)

    def _calculate_composed_term(self, self_exp_order, self_coeff, other_mtf):
        """Helper function to calculate a single term in the composition."""
        if self_exp_order == 0:
            return self_coeff

        term_mtf = other_mtf ** self_exp_order
        if self.is_zero_mtf(term_mtf):
            return None

        return term_mtf * self_coeff

    def compose_one_dim(self, other_mtf):
        """Performs function composition: self(other_mtf(x))."""
        if self.dimension != 1:
            raise ValueError("Composition is only supported for 1D MTF as the outer function.")

        # The result starts as a zero MTF, with the dimension of the inner function
        composed_mtf = type(other_mtf)(
            (np.empty((0, other_mtf.dimension), dtype=np.int32), np.empty((0,), dtype=self.coeffs.dtype)),
            other_mtf.dimension,
            implementation=other_mtf.implementation
        )

        # Iterate through the terms of the outer function (self)
        for i in range(self.coeffs.size):
            self_exp_order = self.exponents[i, 0]
            self_coeff = self.coeffs[i]

            composed_term = self._calculate_composed_term(self_exp_order, self_coeff, other_mtf)
            if composed_term is not None:
                composed_mtf += composed_term

        composed_mtf.truncate_inplace()
        return composed_mtf

    def get_tabular_dataframe(self):
        """Returns a pandas DataFrame representation of MTF or CMTF instance."""
        if self.coeffs.size == 0:
            return pd.DataFrame([{'Coefficient': 0.0, 'Order': 0, 'Exponents': (0,) * self.dimension}])

        data = []
        for i in range(self.coeffs.size):
            exponents = tuple(self.exponents[i])
            coeff = self.coeffs[i]
            order = sum(exponents)
            data.append({
                "Coefficient": coeff,
                "Order": order,
                "Exponents": exponents
            })

        df = pd.DataFrame(data)
        df = df.sort_values(by=['Order', 'Exponents'], ascending=[True, False]).reset_index(drop=True)
        return df

    def extract_coefficient(self, exponents):
        """Extracts the coefficient for a given exponent tuple."""
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if len(exponents) != self.dimension:
            raise ValueError(f"Exponents tuple length must match MTF dimension ({self.dimension}).")

        exponent_row = np.array(exponents, dtype=np.int32)
        match = np.all(self.exponents == exponent_row, axis=1)
        match_indices = np.where(match)[0]

        if match_indices.size > 0:
            return np.array([self.coeffs[match_indices[0]]]).reshape(1)
        else:
            return np.array([0.0], dtype=self.coeffs.dtype)

    def set_coefficient(self, exponents, value):
        """Sets the coefficient for a given exponent tuple."""
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if len(exponents) != self.dimension:
            raise ValueError(f"Exponents tuple length must match MTF dimension ({self.dimension}).")
        if not isinstance(value, (int, float, np.number, complex)):
            raise TypeError("Coefficient value must be a number.")

        exponent_row = np.array(exponents, dtype=np.int32)
        match = np.all(self.exponents == exponent_row, axis=1)
        match_indices = np.where(match)[0]

        if match_indices.size > 0:
            # Exponent exists, update the coefficient
            self.coeffs[match_indices[0]] = value
        else:
            # Exponent does not exist, add it
            self.exponents = np.vstack([self.exponents, exponent_row])
            self.coeffs = np.append(self.coeffs, value)

            # Re-sort to maintain a canonical representation
            sorted_indices = np.lexsort(self.exponents.T)
            self.exponents = self.exponents[sorted_indices]
            self.coeffs = self.coeffs[sorted_indices]

    def get_max_coefficient(self):
        """Finds the maximum absolute value among all coefficients."""
        if self.coeffs.size == 0:
            return 0.0
        return np.max(np.abs(self.coeffs))

    def get_min_coefficient(self, tolerance=np.nan):
        """Finds the minimum absolute value among non-negligible coefficients."""
        if tolerance is np.nan:
            tolerance = get_global_etol()

        if self.coeffs.size == 0:
            return 0.0

        non_negligible_coeffs = self.coeffs[np.abs(self.coeffs) > tolerance]
        if non_negligible_coeffs.size == 0:
            return 0.0

        return np.min(np.abs(non_negligible_coeffs))

    def __str__(self):
        """Returns a string representation of the MTF (tabular format)."""
        df = self.get_tabular_dataframe()
        return f'{df}\n'

    def __repr__(self):
        """Returns a detailed string representation of the MTF (for debugging)."""
        df = self.get_tabular_dataframe()
        return f'{df}\n'
    
    def copy(self):
        """Returns a copy of the MTF."""
        return type(self)((self.exponents.copy(), self.coeffs.copy()), self.dimension, var_name=self.var_name, implementation=self.implementation)

    def _cleanup_after_operation(self):
        """
        Removes coefficients smaller than the global error tolerance in-place.
        """
        etol = get_global_etol()

        if self.coeffs.size == 0:
            return

        keep_mask = np.abs(self.coeffs) > etol

        self.exponents = self.exponents[keep_mask]
        self.coeffs = self.coeffs[keep_mask]

    def __eq__(self, other):
        """Defines equality (==) for MultivariateTaylorFunction objects."""
        if not isinstance(other, MultivariateTaylorFunctionBase):
            return False
        if self.dimension != other.dimension:
            return False

        # Create cleaned versions of both MTFs for comparison
        # This handles cases where one MTF is empty and the other is a zero constant
        self_cleaned = self.copy()
        self_cleaned._cleanup_after_operation()
        other_cleaned = other.copy()
        other_cleaned._cleanup_after_operation()

        # If both are empty after cleanup, they are equal
        if self_cleaned.coeffs.shape[0] == 0 and other_cleaned.coeffs.shape[0] == 0:
            return True

        # If number of terms is different after cleanup, they are not equal
        if self_cleaned.coeffs.shape[0] != other_cleaned.coeffs.shape[0]:
            return False

        # If number of terms is the same, compare the arrays
        if not np.array_equal(self_cleaned.exponents, other_cleaned.exponents):
            # Fallback for non-canonical but equivalent forms (should be rare now)
            self_map = {tuple(exp): coeff for exp, coeff in zip(self_cleaned.exponents, self_cleaned.coeffs)}
            other_map = {tuple(exp): coeff for exp, coeff in zip(other_cleaned.exponents, other_cleaned.coeffs)}
            return self_map == other_map

        return np.allclose(self_cleaned.coeffs, other_cleaned.coeffs)

    def __ne__(self, other):
        """Defines inequality (!=) for MultivariateTaylorFunction objects."""
        return not self.__eq__(other)
        

def _generate_exponent_combinations(dimension, order):
    """Generates all combinations of exponents for a given dimension and order."""
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

def convert_to_mtf(input_val, dimension=None):
    """Converts input to MultivariateTaylorFunction or ComplexMultivariateTaylorFunction."""
    if isinstance(input_val, (MultivariateTaylorFunctionBase)):
        return input_val
    elif isinstance(input_val, (int, float)):
        if dimension is None:
            dimension = get_global_max_dimension()
        return MultivariateTaylorFunctionBase.from_constant(input_val, dimension=dimension, implementation='python')
    elif isinstance(input_val, np.ndarray) and input_val.shape == ():
        return convert_to_mtf(input_val.item(), dimension)
    elif isinstance(input_val, np.number):
        return convert_to_mtf(float(input_val), dimension)
    elif callable(input_val) and input_val.__name__ == 'Var':
        return input_val(dimension)
    else:
        raise TypeError(f"Unsupported input type: {type(input_val)}. Cannot convert to MTF/CMTF.")



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

def sqrt_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of sqrt(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(input_mtf)
    if constant_term_C_value <= 0:
        raise ValueError("Constant part of input to sqrt_taylor is non-positive. This method is for sqrt(constant*(1+x)) form, requiring positive constant.")
    constant_factor_sqrt_C = math.sqrt(constant_term_C_value)
    polynomial_part_x_mtf = polynomial_part_B_mtf / constant_term_C_value
    sqrt_1_plus_x_mtf = sqrt_taylor_1D_expansion(polynomial_part_x_mtf, order=order)
    result_mtf = sqrt_1_plus_x_mtf * constant_factor_sqrt_C
    return result_mtf.truncate(order)

def sqrt_taylor_1D_expansion(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: 1D Taylor expansion of sqrt(1+u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    sqrt_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0
    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('sqrt')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'sqrt' function not found. Ensure coefficients are loaded.")
    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        sqrt_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = np.array([coefficient_val]).reshape(1)
    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(f"Warning: Requested order {order} exceeds precomputed order {elementary_coefficients.MAX_PRECOMPUTED_ORDER}. Calculations may be slower for higher orders.")
        for n_order in range(elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1):
            if n_order == 0:
                coefficient_val = 1.0
            elif n_order == 1:
                coefficient_val = 0.5
            else:
                previous_coefficient = sqrt_taylor_1d_coefficients[_generate_exponent(n_order - 1, variable_index_1d, taylor_dimension_1d)][0]
                coefficient_val = previous_coefficient * (0.5 - (n_order - 1)) / n_order
            sqrt_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = np.array([coefficient_val]).reshape(1)
    sqrt_taylor_1d_mtf = type(variable)(
        coefficients=sqrt_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = sqrt_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)

def isqrt_taylor(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Taylor expansion of isqrt(x) using constant factoring."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(input_mtf)
    if abs(constant_term_C_value) < 1e-9:
        raise ValueError("Constant part of input to isqrt_taylor is too close to zero. This method requires a non-zero constant term.")
    constant_factor_isqrt_C = 1.0 / math.sqrt(constant_term_C_value)
    polynomial_part_x_mtf = polynomial_part_B_mtf / constant_term_C_value
    isqrt_1_plus_x_mtf = isqrt_taylor_1D_expansion(polynomial_part_x_mtf, order=order)
    result_mtf = isqrt_1_plus_x_mtf * constant_factor_isqrt_C
    return result_mtf.truncate(order)

def isqrt_taylor_1D_expansion(variable, order: int = None) -> MultivariateTaylorFunctionBase:
    """Helper: 1D Taylor expansion of isqrt(1+u) around zero, precomputed coefficients."""
    if order is None:
        order = get_global_max_order()
    input_mtf = convert_to_mtf(variable)
    isqrt_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0
    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get('isqrt')
    if precomputed_coeffs is None:
        raise ValueError("Precomputed coefficients for 'isqrt' function not found. Ensure coefficients are loaded.")
    for n_order in range(0, max_precomputed_order + 1):
        coefficient_val = precomputed_coeffs[n_order]
        isqrt_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = np.array([coefficient_val]).reshape(1)
    if order > elementary_coefficients.MAX_PRECOMPUTED_ORDER:
        print(f"Warning: Requested order {order} exceeds precomputed order {elementary_coefficients.MAX_PRECOMPUTED_ORDER}. Calculations may be slower for higher orders.")
        for n_order in range(elementary_coefficients.MAX_PRECOMPUTED_ORDER + 1, order + 1):
            if n_order == 0:
                coefficient_val = 1.0
            elif n_order == 1:
                coefficient_val = -0.5
            else:
                previous_coefficient = isqrt_taylor_1d_coefficients[_generate_exponent(n_order - 1, variable_index_1d, taylor_dimension_1d)][0]
                coefficient_val = previous_coefficient * (-0.5 - (n_order - 1)) / n_order
            isqrt_taylor_1d_coefficients[_generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)] = np.array([coefficient_val]).reshape(1)
    isqrt_taylor_1d_mtf = type(variable)(
        coefficients=isqrt_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = isqrt_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)