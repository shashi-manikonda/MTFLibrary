# MTFLibrary/taylor_function.py
import numpy as np
from collections import defaultdict
import math
from MTFLibrary.elementary_coefficients import load_precomputed_coefficients
import time
import numbers

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
    def __init__(self, coefficients, dimension, var_name=None):
        """Initializes a MultivariateTaylorFunction object."""
        if not isinstance(dimension, int):
            raise ValueError("Dimension must be an integer.")
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        self.coefficients = defaultdict(_create_default_coefficient_array, coefficients)
        self.dimension = dimension
        self.var_name = var_name
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
        result = np.array([0.0]).reshape(1)
        if not self.coefficients:
            return result
        exponents_matrix = np.array(list(self.coefficients.keys()))
        coefficients_array = np.array(list(self.coefficients.values())).flatten()
        powers_matrix = evaluation_point[np.newaxis, :] ** exponents_matrix
        term_values = coefficients_array * np.prod(powers_matrix, axis=1)
        result = np.sum(term_values).reshape(1,)
        return result

    def _add_coefficient_dicts(self, dict1, dict2, subtract=False):
        """Helper function to add or subtract coefficient dictionaries."""
        combined_coeffs = defaultdict(lambda: np.array([0.0]).reshape(1))
        factor = -1.0 if subtract else 1.0
        for exponents in dict1.keys() | dict2.keys():
            coeff1 = dict1.get(exponents, np.array([0.0]).reshape(1))
            coeff2 = dict2.get(exponents, np.array([0.0]).reshape(1))
            combined_val = np.array(coeff1).flatten() + factor * np.array(coeff2).flatten()
            combined_coeffs[exponents] += combined_val
        return combined_coeffs

    def __add__(self, other):
        """Defines addition (+) for MultivariateTaylorFunction objects."""
        if isinstance(other, (MultivariateTaylorFunctionBase, complex)):
            if self.dimension != other.dimension:
                raise ValueError("MTF dimensions must match for addition.")
            sum_coeffs = self._add_coefficient_dicts(self.coefficients, other.coefficients)
            return MultivariateTaylorFunctionBase(coefficients=sum_coeffs, dimension=self.dimension)
        elif isinstance(other, (int, float)):
            return self + MultivariateTaylorFunctionBase.from_constant(other)
        else:
            return NotImplemented

    def __radd__(self, other):
        """Defines reverse addition for commutative property."""
        return self.__add__(other)

    def __sub__(self, other):
        """Defines subtraction (-) for MultivariateTaylorFunction objects."""
        if isinstance(other, (MultivariateTaylorFunctionBase, complex)):
            if self.dimension != other.dimension:
                raise ValueError("MTF dimensions must match for subtraction.")
            sub_coeffs = self._add_coefficient_dicts(self.coefficients, other.coefficients, subtract=True)
            return MultivariateTaylorFunctionBase(coefficients=sub_coeffs, dimension=self.dimension)
        elif isinstance(other, (int, float)):
            return self - MultivariateTaylorFunctionBase.from_constant(other)
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Defines reverse subtraction for non-commutative property."""
        return (MultivariateTaylorFunctionBase.from_constant(other)) - self

    def __mul__(self, other):
        """Defines multiplication (*) for MultivariateTaylorFunction objects."""
        if isinstance(other, (MultivariateTaylorFunctionBase, complex)):
            if self.dimension != other.dimension:
                raise ValueError("MTF dimensions must match for multiplication.")
            new_coeffs_dict = defaultdict(_create_default_coefficient_array)
            self_exponents_list = list(self.coefficients.keys())
            other_exponents_list = list(other.coefficients.keys())
            if not self_exponents_list or not other_exponents_list:
                return MultivariateTaylorFunctionBase(coefficients={}, dimension=self.dimension)
            self_coeffs_array = np.array(list(self.coefficients.values())).reshape(-1)
            other_coeffs_array = np.array(list(other.coefficients.values())).reshape(-1)
            self_exponents_array = np.array(self_exponents_list)
            other_exponents_array = np.array(other_exponents_list)
            new_exponents_array_intermediate = (self_exponents_array[:, np.newaxis, :] + other_exponents_array[np.newaxis, :, :])
            new_exponents_array_reshaped = new_exponents_array_intermediate.reshape(-1, self.dimension)
            new_coeffs_values = self_coeffs_array[:, np.newaxis] * other_coeffs_array[np.newaxis, :]
            new_coeffs_values_flattened = new_coeffs_values.flatten()
            for i in range(len(new_exponents_array_reshaped)):
                exponent_tuple = tuple(new_exponents_array_reshaped[i])
                new_coeffs_dict[exponent_tuple] += np.array([new_coeffs_values_flattened[i]]).reshape(1)
            return MultivariateTaylorFunctionBase(coefficients=new_coeffs_dict, dimension=self.dimension)
        elif isinstance(other, (int, float)):
            return self * MultivariateTaylorFunctionBase.from_constant(other)
        elif isinstance(other, np.ndarray) and other.shape == (1,):
            return self * MultivariateTaylorFunctionBase.from_constant(float(other[0]))
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Defines reverse multiplication for commutative property."""
        return self.__mul__(other)

    def __imul__(self, other):
        """Defines in-place multiplication (*=) with a scalar."""
        if isinstance(other, (int, float)):
            for exponent in self.coefficients:
                self.coefficients[exponent] *= other
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
        neg_coeffs = {}
        for exponents, coeff in self.coefficients.items():
            neg_coeffs[exponents] = -coeff
        return MultivariateTaylorFunctionBase(coefficients=neg_coeffs, dimension=self.dimension)

    def __truediv__(self, other):
        """Defines division (/) for MultivariateTaylorFunction objects."""
        if isinstance(other, MultivariateTaylorFunctionBase):
            inverse_other_mtf = self._inv_mtf_internal(other)
            return self * inverse_other_mtf
        elif isinstance(other, (int, float, np.number)):
            new_coeffs = {}
            for exponents, coeff_value in self.coefficients.items():
                new_coeffs[exponents] = coeff_value / other
            return MultivariateTaylorFunctionBase(coefficients=new_coeffs, dimension=self.dimension)
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
        inverse_series_1d_mtf = MultivariateTaylorFunctionBase(coefficients=dict(coeff_items), dimension=1)
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
        return MultivariateTaylorFunctionBase(coefficients=substituted_coefficients, dimension=self.dimension)

    def truncate_inplace(self, order=None):
        """Truncates the MTF *in place* to a specified order."""
        if order is None:
            order = get_global_max_order()
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")
        coefficients_to_remove = []
        for exponents in self.coefficients:
            if sum(exponents) > order:
                coefficients_to_remove.append(exponents)
        for exponents in coefficients_to_remove:
            del self.coefficients[exponents]
        inferred_max_order = 0
        for exponents in self.coefficients:
            inferred_max_order = max(inferred_max_order, sum(exponents))
        self._max_order = inferred_max_order
        return self

    def truncate(self, order=None):
        """Truncates the MultivariateTaylorFunction to a specified order."""
        etol = get_global_etol()
        if order is None:
            order = get_global_max_order()
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")
        truncated_coeffs = {}
        for exponents, coeff in self.coefficients.items():
            if sum(exponents) <= order and np.any(np.abs(coeff)>etol):
                truncated_coeffs[exponents] = coeff
        return MultivariateTaylorFunctionBase(coefficients=truncated_coeffs, dimension=self.dimension)

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
        if not mtf.coefficients:
            return True
        for coeff_array in mtf.coefficients.values():
            if not np.all(np.abs(coeff_array) < zero_tolerance):
                return False
        return True

    def compose_one_dim(self, other_mtf):
        """Performs function composition: self(other_mtf(x))."""
        if self.dimension != 1:
            raise ValueError("Composition is only supported for 1D MTF as the outer function.")
        composed_coefficients = {}
        max_order_composed = get_global_max_order()
        for self_exponents, self_coeff in self.coefficients.items():
            order_self = sum(self_exponents)
            if order_self == 0:
                if tuple([0] * other_mtf.dimension) in composed_coefficients:
                    composed_coefficients[tuple([0] * other_mtf.dimension)] += self_coeff
                else:
                    composed_coefficients[tuple([0] * other_mtf.dimension)] = self_coeff
            else:
                power_of_u = order_self
                term_mtf = other_mtf**power_of_u
                if self.is_zero_mtf(term_mtf):
                    continue
                term_mtf_scaled = term_mtf * self_coeff
                for term_exponents, term_coeff in term_mtf_scaled.coefficients.items():
                    if sum(term_exponents) <= max_order_composed:
                        if term_exponents in composed_coefficients:
                            composed_coefficients[term_exponents] += term_coeff
                        else:
                            composed_coefficients[term_exponents] = term_coeff
        return MultivariateTaylorFunctionBase(coefficients=composed_coefficients, dimension=other_mtf.dimension)

    def print_tabular(self, order=None, variable_names=None):
        """Prints a tabular representation of the MTF coefficients."""
        print(get_tabular_string(self, order, variable_names))

    def extract_coefficient(self, exponents):
        """Extracts the coefficient for a given exponent tuple."""
        coefficient = self.coefficients.get(exponents)
        if coefficient is None:
            return np.array([0.0]).reshape(1)
        return coefficient

    def set_coefficient(self, exponents, value):
        """Sets the coefficient for a given exponent tuple."""
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if len(exponents) != self.dimension:
            raise ValueError(f"Exponents tuple length must match MTF dimension ({self.dimension}).")
        if not isinstance(value, (int, float)):
            raise TypeError("Coefficient value must be a real number (int or float).")
        self.coefficients[exponents] = np.array([float(value)]).reshape(1)

    def get_max_coefficient(self):
        """Finds the maximum absolute value among all coefficients."""
        max_coeff = 0.0
        for coeff in self.coefficients.values():
            max_coeff = max(max_coeff, np.abs(coeff[0]))
        return max_coeff

    def get_min_coefficient(self, tolerance=np.nan):
        """Finds the minimum absolute value among non-negligible coefficients."""
        if tolerance==np.nan:
            tolerance = get_global_etol()
        min_coeff = float("inf")
        found_non_negligible = False
        for coeff in self.coefficients.values():
            coeff_abs = np.abs(coeff[0])
            if coeff_abs > tolerance:
                min_coeff = min(min_coeff, coeff_abs)
                found_non_negligible = True
        return (min_coeff if found_non_negligible else 0.0)

    def __str__(self):
        """Returns a string representation of the MTF (tabular format)."""
        if self.var_name:
            return f"MultivariateTaylorFunction({self.var_name})"
        return get_tabular_string(self)

    def __repr__(self):
        """Returns a detailed string representation of the MTF (for debugging)."""
        return get_tabular_string(self)
    
    def __eq__(self, other):
        """Defines equality (==) for MultivariateTaylorFunction objects."""
        if not isinstance(other, MultivariateTaylorFunctionBase):
            return False
        if self.dimension != other.dimension:
            return False
        if set(self.coefficients.keys()) != set(other.coefficients.keys()):
            return False
        for exponent, coeff in self.coefficients.items():
            if not np.array_equal(coeff, other.coefficients[exponent]):
                return False
        return True

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

def get_tabular_string(mtf_instance, order=None, variable_names=None):
    """Returns tabular string representation of MTF or CMTF instance."""
    coefficients = mtf_instance.coefficients
    dimension = mtf_instance.dimension
    if order is None:
        if hasattr(mtf_instance, 'get_global_max_order'):
            order = mtf_instance.get_global_max_order()
        else:
            order = get_global_max_order()
    if variable_names is None:
        variable_names = [f'x_{i+1}' for i in range(dimension)]
    headers = ["I", "Coefficient", "Order", "Exponents"]
    rows = []
    term_index = 1
    etol = get_global_etol()
    for exponents, coeff in sorted(coefficients.items(), key=lambda item: sum(item[0])):
        if sum(exponents) <= order and np.any(np.abs(coeff) > etol):
            exponent_str = " ".join(map(str, exponents))
            if np.iscomplexobj(coeff):
                coeff_str = f"{coeff[0].real:.8f}{coeff[0].imag:+8f}j"
            else:
                coeff_str = f"{coeff[0]:+16.16e}"
            rows.append([f"{term_index: <4}", coeff_str, str(sum(exponents)), exponent_str])
            term_index += 1
    if not rows:
        return "MultivariateTaylorFunction (truncated or zero)"
    column_widths = []
    current_header_index = 0
    for header in headers:
        if header == "I":
            column_widths.append(4 + 2)
        else:
            column_widths.append(max(len(header), max(len(row[current_header_index]) for row in rows)) + 2)
        current_header_index += 1
    header_row = "| " + "| ".join(headers[i].ljust(column_widths[i]-2) for i in range(len(headers))) + "|"
    separator = "|" + "|".join("-" * (w-1) for w in column_widths) + "|"
    table_str = header_row + "\n" + separator + "\n"
    for row in rows:
        table_str += "| " + "| ".join(row[i].ljust(column_widths[i]-2) for i in range(len(headers))) + "|\n"
    return '\n' + table_str

def _split_constant_polynomial_part(input_mtf: MultivariateTaylorFunctionBase) -> tuple[float, MultivariateTaylorFunctionBase]:
    """Helper: Splits MTF into constant and polynomial parts."""
    dimension = input_mtf.dimension
    constant_term_C_value = input_mtf.coefficients.get((0,) * dimension, np.array([0.0])).item()
    polynomial_part_coefficients = {
        exponents: coefficients
        for exponents, coefficients in input_mtf.coefficients.items()
        if exponents != (0,) * dimension
    }
    polynomial_part_mtf = MultivariateTaylorFunctionBase(
        coefficients=polynomial_part_coefficients, dimension=dimension
    )
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