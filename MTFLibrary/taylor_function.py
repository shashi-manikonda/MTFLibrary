# MTFLibrary/taylor_function.py
import numpy as np
from collections import defaultdict
import math
from MTFLibrary.elementary_coefficients import load_precomputed_coefficients
import time
import numbers
import pandas as pd

from . import elementary_coefficients


_GLOBAL_MAX_ORDER = None
_GLOBAL_MAX_DIMENSION = None
_INITIALIZED = False
_DEFAULT_ETOL = 1e-16
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
    """Sets the global error tolerance (etol) for MTFLibrary."""
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

def _generate_exponent(order, var_index, dimension):
    """Generates an exponent tuple for a monomial term."""
    exponent = [0] * dimension
    exponent[var_index] = order
    return tuple(exponent)

def _create_default_coefficient_array():
    """Creates a default coefficient array."""
    return np.array([0.0]).reshape(1)

class MultivariateTaylorFunctionBase:
    """Represents a multivariate Taylor function."""
    def __init__(self, coefficients, dimension, var_name=None, implementation='python'):
        """
        Initializes a MultivariateTaylorFunction object.

        Args:
            coefficients: Can be one of two things:
                1. A dictionary mapping exponent tuples to coefficient values (as single-element NumPy arrays).
                2. A tuple containing two NumPy arrays: (exponents, coeffs).
                   'exponents' is a 2D array of shape (n_terms, dimension).
                   'coeffs' is a 1D array of shape (n_terms,).
            dimension (int): The number of variables in the function.
            var_name (str, optional): A name for the variable if it's a single-variable function.
            implementation (str): The backend to use ('python' or 'cython').
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")

        self.dimension = dimension
        self.var_name = var_name
        self.implementation = implementation

        if isinstance(coefficients, dict):
            # Convert dictionary to the new NumPy array format
            if not coefficients:
                self.exponents = np.empty((0, self.dimension), dtype=np.int32)
                self.coeffs = np.empty((0,), dtype=np.float64)
            else:
                # Sort by exponents to ensure a canonical representation
                sorted_items = sorted(coefficients.items())
                self.exponents = np.array([item[0] for item in sorted_items], dtype=np.int32)
                # Check if any coefficient is complex to determine dtype
                # We need to handle the case where the value is already a numpy array
                is_complex = any(np.iscomplexobj(v) or (isinstance(v, np.ndarray) and np.iscomplexobj(v.item())) for _, v in sorted_items)
                dtype = np.complex128 if is_complex else np.float64
                self.coeffs = np.array([item[1] for item in sorted_items], dtype=dtype).flatten()

        elif isinstance(coefficients, tuple) and len(coefficients) == 2 and isinstance(coefficients[0], np.ndarray) and isinstance(coefficients[1], np.ndarray):
            # This is the new, preferred way to create an MTF object
            self.exponents, self.coeffs = coefficients
            if self.exponents.ndim != 2 or self.exponents.shape[1] != self.dimension:
                raise ValueError("Exponents array has incorrect shape.")
            if self.coeffs.ndim != 1 or self.coeffs.shape[0] != self.exponents.shape[0]:
                raise ValueError("Coefficients array has incorrect shape.")
        else:
            raise TypeError("Unsupported type for 'coefficients'. Must be a dict or a tuple of (exponents, coeffs) arrays.")

        # The old self.coefficients is now deprecated and should not be used.
        self.truncate_inplace()

    @classmethod
    def from_constant(cls, constant_value):
        """Creates an MTF representing a constant value."""
        dimension=get_global_max_dimension()
        coeffs = {(0,) * dimension: np.array([float(constant_value)]).reshape(1)}
        return cls(coefficients=coeffs, dimension=dimension)

    @classmethod
    def from_variable(cls, var_index, dimension):
        """Creates an MTF representing a single variable."""
        if not (1 <= var_index <= dimension):
            raise ValueError(f"Variable index must be between 1 and {dimension}, inclusive.")
        exponent = [0] * dimension
        exponent[var_index - 1] = 1
        coeffs = {tuple(exponent): np.array([1.0]).reshape(1)}
        return cls(coefficients=coeffs, dimension=dimension, var_name=f"x_{var_index}")

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

        # We can directly use self.exponents and self.coeffs
        powers_matrix = evaluation_point[np.newaxis, :] ** self.exponents
        term_values = self.coeffs * np.prod(powers_matrix, axis=1)
        result = np.sum(term_values).reshape(1,)
        return result

    def __add__(self, other):
        """Defines addition (+) for MultivariateTaylorFunction objects."""
        if isinstance(other, (int, float)):
            new_coeffs = self.coeffs.copy()
            const_exp = np.zeros(self.dimension, dtype=np.int32)
            match = np.all(self.exponents == const_exp, axis=1)
            const_idx = np.where(match)[0]

            if const_idx.size > 0:
                new_coeffs[const_idx[0]] += other
                return MultivariateTaylorFunctionBase((self.exponents, new_coeffs), self.dimension, implementation=self.implementation)
            else:
                new_exponents = np.vstack([self.exponents, const_exp])
                new_coeffs = np.append(self.coeffs, other)
                return MultivariateTaylorFunctionBase((new_exponents, new_coeffs), self.dimension, implementation=self.implementation)

        if not isinstance(other, MultivariateTaylorFunctionBase):
            return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for addition.")

        if self.implementation == 'cython':
            from MTFLibrary import mtf_cython
            new_exps, new_coeffs = mtf_cython.add_mtf_cython(self.exponents, self.coeffs, other.exponents, other.coeffs)
            return MultivariateTaylorFunctionBase((new_exps, new_coeffs), self.dimension, implementation=self.implementation)
        elif self.implementation == 'cpp':
            from MTFLibrary import mtf_cpp
            new_exps, new_coeffs = mtf_cpp.add_mtf_cpp(self.exponents, self.coeffs, other.exponents, other.coeffs)
            return MultivariateTaylorFunctionBase((new_exps, new_coeffs), self.dimension, implementation=self.implementation)
        else: # Default to python
            all_exponents = np.vstack([self.exponents, other.exponents])
            all_coeffs = np.concatenate([self.coeffs, other.coeffs])

            unique_exponents, inverse_indices = np.unique(all_exponents, axis=0, return_inverse=True)
            summed_coeffs = np.bincount(inverse_indices, weights=all_coeffs)

            return MultivariateTaylorFunctionBase((unique_exponents, summed_coeffs), self.dimension, implementation=self.implementation)

    def __radd__(self, other):
        """Defines reverse addition for commutative property."""
        return self.__add__(other)

    def __sub__(self, other):
        """Defines subtraction (-) for MultivariateTaylorFunction objects."""
        if isinstance(other, MultivariateTaylorFunctionBase):
            if self.dimension != other.dimension:
                raise ValueError("MTF dimensions must match for subtraction.")

            # This logic will be moved to a Cython function
            all_exponents = np.vstack([self.exponents, other.exponents])
            negated_other_coeffs = -other.coeffs
            all_coeffs = np.concatenate([self.coeffs, negated_other_coeffs])

            unique_exponents, inverse_indices = np.unique(all_exponents, axis=0, return_inverse=True)
            summed_coeffs = np.bincount(inverse_indices, weights=all_coeffs)

            return MultivariateTaylorFunctionBase((unique_exponents, summed_coeffs), self.dimension, implementation=self.implementation)

        elif isinstance(other, (int, float)):
            return self + (-other)
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Defines reverse subtraction for non-commutative property."""
        return -(self - other)

    def __mul__(self, other):
        """Defines multiplication (*) for MultivariateTaylorFunction objects."""
        if isinstance(other, (int, float, np.number)):
            # Scalar multiplication
            return MultivariateTaylorFunctionBase((self.exponents.copy(), self.coeffs * other), self.dimension, implementation=self.implementation)

        if not isinstance(other, MultivariateTaylorFunctionBase):
            return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for multiplication.")

        if self.coeffs.size == 0 or other.coeffs.size == 0:
            return MultivariateTaylorFunctionBase((np.empty((0, self.dimension), dtype=np.int32), np.empty((0,), dtype=np.float64)), self.dimension, implementation=self.implementation)

        if self.implementation == 'cython':
            from MTFLibrary import mtf_cython
            new_exps, new_coeffs = mtf_cython.multiply_mtf_cython(self.exponents, self.coeffs, other.exponents, other.coeffs)
            return MultivariateTaylorFunctionBase((new_exps, new_coeffs), self.dimension, implementation=self.implementation)
        elif self.implementation == 'cpp':
            from MTFLibrary import mtf_cpp
            new_exps, new_coeffs = mtf_cpp.multiply_mtf_cpp(self.exponents, self.coeffs, other.exponents, other.coeffs)
            return MultivariateTaylorFunctionBase((new_exps, new_coeffs), self.dimension, implementation=self.implementation)
        else: # Default to python
            # This is the core convolution.
            new_exponents = self.exponents[:, np.newaxis, :] + other.exponents[np.newaxis, :, :]
            new_exponents = new_exponents.reshape(-1, self.dimension)

            new_coeffs = self.coeffs[:, np.newaxis] * other.coeffs[np.newaxis, :]
            new_coeffs = new_coeffs.flatten()

            # Group by exponent and sum coefficients
            unique_exponents, inverse_indices = np.unique(new_exponents, axis=0, return_inverse=True)
            summed_coeffs = np.bincount(inverse_indices, weights=new_coeffs)

            return MultivariateTaylorFunctionBase((unique_exponents, summed_coeffs), self.dimension, implementation=self.implementation)

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
                return MultivariateTaylorFunctionBase.from_constant(1.0)
            if power == 1:
                return self
            result_mtf = MultivariateTaylorFunctionBase.from_constant(1.0)
            result_mtf.implementation = self.implementation
            for _ in range(power):
                if self.is_zero_mtf(result_mtf):
                    continue
                result_mtf = result_mtf * self
            return result_mtf
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
        return MultivariateTaylorFunctionBase((self.exponents.copy(), -self.coeffs.copy()), self.dimension, implementation=self.implementation)

    def __truediv__(self, other):
        """Defines division (/) for MultivariateTaylorFunction objects."""
        if isinstance(other, MultivariateTaylorFunctionBase):
            inverse_other_mtf = self._inv_mtf_internal(other)
            return self * inverse_other_mtf
        elif isinstance(other, (int, float, np.number)):
            return MultivariateTaylorFunctionBase((self.exponents.copy(), self.coeffs / other), self.dimension, implementation=self.implementation)
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
        inverse_series_1d_mtf = MultivariateTaylorFunctionBase(coefficients=dict(coeff_items), dimension=1, implementation=self.implementation)
        composed_mtf = inverse_series_1d_mtf.compose_one_dim(
            rescaled_mtf - MultivariateTaylorFunctionBase.from_constant(1.0)
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
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number (int or float).")
        substituted_coefficients = defaultdict(_create_default_coefficient_array)
        for exponent_tuple, coeff_value in self.coefficients.items():
            exponent_to_substitute = exponent_tuple[var_index - 1]
            substitution_factor = value**exponent_to_substitute
            modified_coefficient = coeff_value * substitution_factor
            new_exponent_list = list(exponent_tuple)
            new_exponent_list[var_index - 1] = 0
            new_exponent_tuple = tuple(new_exponent_list)
            substituted_coefficients[new_exponent_tuple] += modified_coefficient
        return MultivariateTaylorFunctionBase(coefficients=substituted_coefficients, dimension=self.dimension, implementation=self.implementation)

    def truncate_inplace(self, order=None):
        """Truncates the MTF *in place* to a specified order."""
        if order is None:
            order = get_global_max_order()
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")

        if self.exponents.shape[0] == 0:
            self._max_order = 0
            return self

        term_orders = np.sum(self.exponents, axis=1)
        keep_mask = term_orders <= order

        self.exponents = self.exponents[keep_mask]
        self.coeffs = self.coeffs[keep_mask]

        if self.coeffs.shape[0] > 0:
            self._max_order = np.max(np.sum(self.exponents, axis=1))
        else:
            self._max_order = 0

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

        return MultivariateTaylorFunctionBase((new_exponents, new_coeffs), self.dimension, implementation=self.implementation)

    def substitute_variable_inplace(self, var_dimension, value):
        """Substitutes a variable in the MTF with a numerical value IN-PLACE."""
        if not isinstance(var_dimension, int) or not (1 <= var_dimension <= self.dimension):
            raise ValueError("Invalid var_dimension.")
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be an integer or float.")
        dimension_index = var_dimension - 1
        original_coefficients = self.coefficients.copy()
        self.coefficients = defaultdict(lambda: np.array([0.0]).reshape(1))
        for exponent_tuple, coeff_value in original_coefficients.items():
            exponent_for_var = exponent_tuple[dimension_index]
            multiplier = value**exponent_for_var
            new_exponent_list = list(exponent_tuple)
            new_exponent_list[dimension_index] = 0
            new_exponent_tuple = tuple(new_exponent_list)
            self.coefficients[new_exponent_tuple] += coeff_value * multiplier

    def is_zero_mtf(self, mtf, zero_tolerance=None):
        """Checks if an MTF is effectively zero."""
        if zero_tolerance is None:
            zero_tolerance = get_global_etol()
        if mtf.coeffs.size == 0:
            return True
        return np.all(np.abs(mtf.coeffs) < zero_tolerance)

    def compose_one_dim(self, other_mtf):
        """Performs function composition: self(other_mtf(x))."""
        if self.dimension != 1:
            raise ValueError("Composition is only supported for 1D MTF as the outer function.")

        if self.implementation in ['cython', 'cpp']:
            from MTFLibrary import mtf_cython
            new_exps, new_coeffs = mtf_cython.compose_one_dim_cython(
                self.exponents, self.coeffs,
                other_mtf.exponents, other_mtf.coeffs,
                other_mtf.dimension
            )
            return MultivariateTaylorFunctionBase((new_exps, new_coeffs), other_mtf.dimension, implementation=self.implementation)
        else: # Default to python
            # The result starts as a zero MTF, with the dimension of the inner function
            composed_mtf = MultivariateTaylorFunctionBase(
                (np.empty((0, other_mtf.dimension), dtype=np.int32), np.empty((0,), dtype=np.float64)),
                other_mtf.dimension,
                implementation=self.implementation
            )

            # Iterate through the terms of the outer function (self)
            for i in range(self.coeffs.size):
                self_exp_order = self.exponents[i, 0]
                self_coeff = self.coeffs[i]

                if self_exp_order == 0:
                    composed_mtf += self_coeff
                else:
                    term_mtf = other_mtf ** self_exp_order
                    if self.is_zero_mtf(term_mtf):
                        continue

                    term_mtf_scaled = term_mtf * self_coeff
                    composed_mtf += term_mtf_scaled

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
            return np.array([0.0]).reshape(1)

    def set_coefficient(self, exponents, value):
        """Sets the coefficient for a given exponent tuple."""
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if len(exponents) != self.dimension:
            raise ValueError(f"Exponents tuple length must match MTF dimension ({self.dimension}).")
        if not isinstance(value, (int, float, np.number)):
            raise TypeError("Coefficient value must be a real number.")

        exponent_row = np.array(exponents, dtype=np.int32)
        match = np.all(self.exponents == exponent_row, axis=1)
        match_indices = np.where(match)[0]

        if match_indices.size > 0:
            # Exponent exists, update the coefficient
            self.coeffs[match_indices[0]] = float(value)
        else:
            # Exponent does not exist, add it
            self.exponents = np.vstack([self.exponents, exponent_row])
            self.coeffs = np.append(self.coeffs, float(value))

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
        return MultivariateTaylorFunctionBase((self.exponents.copy(), self.coeffs.copy()), self.dimension, var_name=self.var_name, implementation=self.implementation)

    def __eq__(self, other):
        """Defines equality (==) for MultivariateTaylorFunction objects."""
        if not isinstance(other, MultivariateTaylorFunctionBase):
            return False
        if self.dimension != other.dimension:
            return False
        if self.exponents.shape != other.exponents.shape or self.coeffs.shape != other.coeffs.shape:
            return False

        # A more robust comparison would be to sort both and then compare
        # but for now, we assume they are in a canonical form.
        return np.all(self.exponents == other.exponents) and np.allclose(self.coeffs, other.coeffs)

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
        return MultivariateTaylorFunctionBase.from_constant(input_val)
    elif isinstance(input_val, np.ndarray) and input_val.shape == ():
        return convert_to_mtf(input_val.item(), dimension)
    elif isinstance(input_val, np.number):
        return convert_to_mtf(float(input_val), dimension)
    elif callable(input_val) and input_val.__name__ == 'Var':
        return input_val(dimension)
    else:
        raise TypeError(f"Unsupported input type: {type(input_val)}. Cannot convert to MTF/CMTF.")

# def get_tabular_string(mtf_instance, order=None, variable_names=None):
#     print('-- ## will be removed ## --')


# def get_tabular_string(mtf_instance, order=None, variable_names=None):
#     """Returns tabular string representation of MTF or CMTF instance."""
#     coefficients = mtf_instance.coefficients
#     dimension = mtf_instance.dimension
#     if order is None:
#         if hasattr(mtf_instance, 'get_global_max_order'):
#             order = mtf_instance.get_global_max_order()
#         else:
#             order = get_global_max_order()
#     if variable_names is None:
#         variable_names = [f'x_{i+1}' for i in range(dimension)]
#     headers = ["I", "Coefficient", "Order", "Exponents"]
#     rows = []
#     term_index = 1
#     etol = get_global_etol()
#     for exponents, coeff in sorted(coefficients.items(), key=lambda item: sum(item[0])):
#         if sum(exponents) <= order and np.any(np.abs(coeff) > etol):
#             exponent_str = " ".join(map(str, exponents))
#             if np.iscomplexobj(coeff):
#                 coeff_str = f"{coeff[0].real:.8f}{coeff[0].imag:+8f}j"
#             else:
#                 coeff_str = f"{coeff[0]:+16.16e}"
#             rows.append([f"{term_index: <4}", coeff_str, str(sum(exponents)), exponent_str])
#             term_index += 1
#     if not rows:
#         return "MultivariateTaylorFunction (truncated or zero)"
#     column_widths = []
#     current_header_index = 0
#     for header in headers:
#         if header == "I":
#             column_widths.append(4 + 2)
#         else:
#             column_widths.append(max(len(header), max(len(row[current_header_index]) for row in rows)) + 2)
#         current_header_index += 1
#     header_row = "| " + "| ".join(headers[i].ljust(column_widths[i]-2) for i in range(len(headers))) + "|"
#     separator = "|" + "|".join("-" * (w-1) for w in column_widths) + "|"
#     table_str = header_row + "\n" + separator + "\n"
#     for row in rows:
#         table_str += "| " + "| ".join(row[i].ljust(column_widths[i]-2) for i in range(len(headers))) + "|\n"
#     return '\n' + table_str


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
        polynomial_part_mtf = MultivariateTaylorFunctionBase((poly_exponents, poly_coeffs), dimension)
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
    sqrt_taylor_1d_mtf = MultivariateTaylorFunctionBase(
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
    isqrt_taylor_1d_mtf = MultivariateTaylorFunctionBase(
        coefficients=isqrt_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = isqrt_taylor_1d_mtf.compose_one_dim(input_mtf)
    return composed_mtf.truncate(order)