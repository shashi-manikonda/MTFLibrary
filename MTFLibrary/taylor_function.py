# MTFLibrary/taylor_function.py
import numpy as np
from collections import defaultdict
import math

# from MTFLibrary.complex_taylor_function import ComplexMultivariateTaylorFunction  # Assuming this is in a separate file

_GLOBAL_MAX_ORDER = None
_GLOBAL_MAX_DIMENSION = None
_INITIALIZED = False
_DEFAULT_ETOL = 1e-9

def initialize_mtf_globals(max_order, max_dimension):
    global _GLOBAL_MAX_ORDER, _GLOBAL_MAX_DIMENSION, _INITIALIZED
    if not isinstance(max_order, int) or max_order < 0:
        raise ValueError("max_order must be a non-negative integer.")
    if not isinstance(max_dimension, int) or max_dimension <= 0:
        raise ValueError("max_dimension must be a positive integer.")
    _GLOBAL_MAX_ORDER = max_order
    _GLOBAL_MAX_DIMENSION = max_dimension
    _INITIALIZED = True
    max_num_coefficients = math.comb(max_order + max_dimension, max_dimension)
    print(f"The max number of Taylor coefficients (order={max_order},nvars={max_dimension}): {max_num_coefficients} \n")


def set_global_max_order(order):
    global _GLOBAL_MAX_ORDER, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals must be initialized before setting max order.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")
    _GLOBAL_MAX_ORDER = order

def get_global_max_order():
    global _GLOBAL_MAX_ORDER, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals are not initialized.")
    return _GLOBAL_MAX_ORDER

def get_global_max_dimension():
    global _GLOBAL_MAX_DIMENSION, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals are not initialized.")
    return _GLOBAL_MAX_DIMENSION

def set_global_etol(etol):
    global _DEFAULT_ETOL, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals must be initialized before setting error tolerance.")
    if not isinstance(etol, float) or etol <= 0:
        raise ValueError("Error tolerance (etol) must be a positive float.")
    _DEFAULT_ETOL = etol

def get_global_etol():
    global _DEFAULT_ETOL, _INITIALIZED
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals are not initialized.")
    return _DEFAULT_ETOL


def _generate_exponent(order, var_index, dimension):
    """Generates an exponent tuple of given dimension, with order at var_index and 0 elsewhere."""
    exponent = [0] * dimension
    exponent[var_index] = order
    return tuple(exponent)


def _inverse_taylor_1d(order):
    """Taylor expansion of g(w) = 1/(1+w) around 0."""
    coeffs_1d = {}
    taylor_dimension_1d = 1
    var_index_1d = 0
    for n in range(0, order + 1):
        coeff = (-1)**n
        coeffs_1d[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff],dtype=np.float64).reshape(1)
    return MultivariateTaylorFunction(coefficients=coeffs_1d, dimension=taylor_dimension_1d)


def _sqrt_taylor_1d(order):
    """Taylor expansion of g(w) = sqrt(1+w) around 0."""
    coeffs_1d = {}
    taylor_dimension_1d = 1
    var_index_1d = 0
    for n in range(0, order + 1):
        if n == 0:
            coeff = 1.0
        elif n == 1:
            coeff = 0.5
        else:
            coeff = 1.0
            for k in range(n):
                coeff *= (0.5 - k) / (k + 1) # More stable iterative calculation
        coeffs_1d[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff], dtype=np.float64).reshape(1)
    return MultivariateTaylorFunction(coefficients=coeffs_1d, dimension=taylor_dimension_1d)


def _create_default_coefficient_array():
    return np.array([0.0]).reshape(1)


class MultivariateTaylorFunction:
    """
    Represents a multivariate Taylor function.
    """

    def __init__(self, coefficients, dimension, var_name=None):
        self.coefficients = defaultdict(_create_default_coefficient_array, coefficients) # Default to 0.0, shape (1,)
        self.dimension = dimension
        self.var_name = var_name
        self.truncate_inplace()


    @classmethod
    def from_constant(cls, constant_value, dimension):
        coeffs = {(0,) * dimension: np.array([float(constant_value)]).reshape(1)} # Ensure shape (1,)
        return cls(coefficients=coeffs, dimension=dimension)

    @classmethod
    def from_variable(cls, var_index, dimension):
        if not (1 <= var_index <= dimension):
            raise ValueError(f"Variable index must be between 1 and {dimension}, inclusive.")
        exponent = [0] * dimension
        exponent[var_index - 1] = 1
        coeffs = {tuple(exponent): np.array([1.0]).reshape(1)} # Ensure shape (1,)
        return cls(coefficients=coeffs, dimension=dimension, var_name=f'x_{var_index}') # Set var_name here

    @classmethod
    def from_complex_mtf(cls, complex_mtf):
        coeffs_real = {}
        for exponent, coeff_complex in complex_mtf.coefficients.items():
            coeffs_real[exponent] = np.array([coeff_complex.real]).reshape(1) # Ensure shape (1,) and take real part
        return cls(coefficients=coeffs_real, dimension=complex_mtf.dimension)


    def __call__(self, evaluation_point):
        return self.eval(evaluation_point)

    def eval(self, evaluation_point):
        if len(evaluation_point) != self.dimension:
            raise ValueError(f"Evaluation point dimension must match MTF dimension ({self.dimension}).")
        evaluation_point = np.array(evaluation_point)
        result = np.array([0.0]).reshape(1)

        if not self.coefficients: # Handle empty coefficient dict case
            return result

        exponents_matrix = np.array(list(self.coefficients.keys())) # Shape (num_coeffs, dimension)
        coefficients_array = np.array(list(self.coefficients.values())).flatten() # Shape (num_coeffs,)

        powers_matrix = evaluation_point[np.newaxis, :] ** exponents_matrix # Broadcasting, shape (num_coeffs, dimension)
        term_values = coefficients_array * np.prod(powers_matrix, axis=1) # Shape (num_coeffs,)

        result = np.sum(term_values).reshape(1,) # Sum all terms and reshape to (1,)
        return result


    # Operator Overloading for Real Arithmetic
    def __add__(self, other):
        if isinstance(other, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)): #Allow CMTF addition
            if self.dimension != other.dimension:
                raise ValueError("MTF dimensions must match for addition.")
            sum_coeffs = self._add_coefficient_dicts(self.coefficients, other.coefficients) # Using self._ for internal helper
            return MultivariateTaylorFunction(coefficients=sum_coeffs, dimension=self.dimension)
        elif isinstance(other, (int, float)):
            return self + MultivariateTaylorFunction.from_constant(other, self.dimension)
        elif isinstance(other, (complex)):
            return self + ComplexMultivariateTaylorFunction.from_constant(other, self.dimension)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other) # Addition is commutative

    def __sub__(self, other):
        if isinstance(other, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)): #Allow CMTF subtraction
            if self.dimension != other.dimension:
                raise ValueError("MTF dimensions must match for subtraction.")
            sub_coeffs = self._add_coefficient_dicts(self.coefficients, other.coefficients, subtract=True) # Using self._ for internal helper
            return MultivariateTaylorFunction(coefficients=sub_coeffs, dimension=self.dimension)
        elif isinstance(other, (int, float)):
            return self - MultivariateTaylorFunction.from_constant(other, self.dimension)
        else:
            return NotImplemented

    def __rsub__(self, other):
        return (MultivariateTaylorFunction.from_constant(other, self.dimension)) - self



    def __mul__(self, other):
        if isinstance(other, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)):
            if self.dimension != other.dimension:
                raise ValueError("MTF dimensions must match for multiplication.")
    
            new_coeffs_dict = defaultdict(lambda: np.array([0.0]).reshape(1))
    
            self_exponents_list = list(self.coefficients.keys())
            self_coeffs_array = np.array(list(self.coefficients.values())).reshape(-1)
            other_exponents_list = list(other.coefficients.keys())
            other_coeffs_array = np.array(list(other.coefficients.values())).reshape(-1)
    
            self_exponents_array = np.array(self_exponents_list)
            other_exponents_array = np.array(other_exponents_list)
    
            # 1. Vectorized exponent addition
            new_exponents_array_intermediate = self_exponents_array[:, np.newaxis, :] + other_exponents_array[np.newaxis, :, :]
            new_exponents_array_reshaped = new_exponents_array_intermediate.reshape(-1, self.dimension)
    
            # 2. Vectorized coefficient multiplication
            new_coeffs_values = self_coeffs_array[:, np.newaxis] * other_coeffs_array[np.newaxis, :]
            new_coeffs_values_flattened = new_coeffs_values.flatten()
    
            # 3. Convert exponents to bytes for dictionary keys, and accumulate coefficients
            for i in range(len(new_exponents_array_reshaped)):
                exponent_bytes = new_exponents_array_reshaped[i].tobytes() # Use tobytes() as key
                new_coeffs_dict[exponent_bytes] += np.array([new_coeffs_values_flattened[i]]).reshape(1)
    
    
            # 4. Convert bytes keys back to tuples for the final MTF object (important for consistency!)
            final_coeffs_dict = {}
            for exponent_bytes, coeff_value in new_coeffs_dict.items():
                exponent_tuple = tuple(np.frombuffer(exponent_bytes, dtype=np.int64)) # Assuming exponents are integers (adjust dtype if needed)
                final_coeffs_dict[exponent_tuple] = coeff_value
    
    
            return MultivariateTaylorFunction(coefficients=final_coeffs_dict, dimension=self.dimension)
    
        # ... (rest of __mul__ method for scalar and other types remains the same) ...    
    # def __mul__(self, other):
    #     if isinstance(other, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)):
    #         if self.dimension != other.dimension:
    #             raise ValueError("MTF dimensions must match for multiplication.")
    
    #         new_coeffs_dict = defaultdict(lambda: np.array([0.0]).reshape(1))
    
    #         self_exponents_list = list(self.coefficients.keys())
    #         self_coeffs_array = np.array(list(self.coefficients.values())).reshape(-1)
    #         other_exponents_list = list(other.coefficients.keys())
    #         other_coeffs_array = np.array(list(other.coefficients.values())).reshape(-1)
    
    #         self_exponents_array = np.array(self_exponents_list) # Shape (num_coeffs1, dimension)
    #         other_exponents_array = np.array(other_exponents_list) # Shape (num_coeffs2, dimension)
    
    #         # 1. Vectorized exponent addition (broadcasting)
    #         new_exponents_array_intermediate = self_exponents_array[:, np.newaxis, :] + other_exponents_array[np.newaxis, :, :]
    #         # new_exponents_array_intermediate will have shape (num_coeffs1, num_coeffs2, dimension)
    
    #         # 2. Reshape to get pairs of exponents ready to become tuples
    #         new_exponents_array_reshaped = new_exponents_array_intermediate.reshape(-1, self.dimension)
    #         # Shape: (num_coeffs1 * num_coeffs2, dimension)
    
    #         # 3. Vectorized coefficient multiplication (broadcasting)
    #         new_coeffs_values = self_coeffs_array[:, np.newaxis] * other_coeffs_array[np.newaxis, :]
    #         # new_coeffs_values will have shape (num_coeffs1, num_coeffs2)
    
    #         # 4. Flatten coefficient values to match the flattened exponents array
    #         new_coeffs_values_flattened = new_coeffs_values.flatten() # Shape: (num_coeffs1 * num_coeffs2)
    
    
    #         # 5. Convert exponents array back to tuples (required for dictionary keys)
    #         new_exponents_tuples = [tuple(exponent) for exponent in new_exponents_array_reshaped]
    
    
    #         # 6. Accumulate coefficients into the defaultdict
    #         for i in range(len(new_exponents_tuples)):
    #             new_coeffs_dict[new_exponents_tuples[i]] += np.array([new_coeffs_values_flattened[i]]).reshape(1)
    
    
    #         return MultivariateTaylorFunction(coefficients=new_coeffs_dict, dimension=self.dimension)
        
        
        
        
        elif isinstance(other, (int, float)):
            return self * MultivariateTaylorFunction.from_constant(other, self.dimension)
        elif isinstance(other, np.ndarray) and other.shape == (1,):
            return self * MultivariateTaylorFunction.from_constant(float(other[0]), self.dimension)
        else:
            return NotImplemented    
    
    # def __mul__(self, other):
    #     if isinstance(other, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)): #Allow CMTF multiplication
    #         if self.dimension != other.dimension:
    #             raise ValueError("MTF dimensions must match for multiplication.")
    #         new_coeffs = defaultdict(lambda: np.array([0.0]).reshape(1)) # Initialize with shape (1,)
    #         for exponents1, coeff1 in self.coefficients.items():
    #             for exponents2, coeff2 in other.coefficients.items():
    #                 new_exponent = tuple(exponents1[i] + exponents2[i] for i in range(self.dimension))
    #                 new_coeffs[new_exponent] += np.array(coeff1).flatten() * np.array(coeff2).flatten()
    #         return MultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension)
    #     elif isinstance(other, (int, float)):
    #         return self * MultivariateTaylorFunction.from_constant(other, self.dimension)
    #     elif isinstance(other, np.ndarray) and other.shape == (1,): # Handle numpy array scalar
    #         return self * MultivariateTaylorFunction.from_constant(float(other[0]), self.dimension) # Convert numpy scalar to float
    #     else:
    #         return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other) # Multiplication is commutative

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            for exponent in self.coefficients:
                self.coefficients[exponent] *= other
            return self
        else:
            return NotImplemented

    def __pow__(self, power):
        if power == 0.5: # Square root case
            return self._sqrt_mtf_internal(self) # Call sqrt internal function
        if not isinstance(power, int) or power < 0:
            raise ValueError("Power must be a non-negative integer or 0.5 for sqrt.")
        if power == 0:
            return MultivariateTaylorFunction.from_constant(1.0, self.dimension)
        if power == 1:
            return self # Return self for power=1

        result_mtf = MultivariateTaylorFunction.from_constant(1.0, self.dimension) # Initialize to 1
        for _ in range(power): # Iterative multiplication
            result_mtf = result_mtf * self
        return result_mtf

    def _sqrt_mtf_internal(self, mtf_instance, order=None):
        """Internal method to calculate the Taylor expansion of sqrt(mtf_instance)."""
        if order is None:
            order = get_global_max_order()

        # (1) Check for constant term c0 and ensure it's non-negative (for real sqrt)
        constant_term_coeff = mtf_instance.extract_coefficient(tuple([0] * mtf_instance.dimension))
        c0 = constant_term_coeff.item()

        if c0 < -get_global_etol(): # Check if constant term is negative (real case)
            raise ValueError("Cannot take square root of MTF with negative constant term (in real MTF context). Consider CMTF for complex square roots.")
        if abs(c0) < get_global_etol():
            raise ValueError("Cannot take square root of MTF with zero constant term (or very close to zero), as division by zero will occur in rescaling. Consider adding a small constant.")


        # (2) Rescale MTF so constant term is 1
        rescaled_mtf = mtf_instance / c0

        # (3) Get Taylor expansion of g(w) = sqrt(1+w)
        sqrt_series_1d_mtf = _sqrt_taylor_1d(order)

        # (4) Compose g(f_rescaled - 1)
        composed_mtf = sqrt_series_1d_mtf.compose(rescaled_mtf - MultivariateTaylorFunction.from_constant(1.0, rescaled_mtf.dimension))

        # (5) Rescale back by sqrt(c0)
        rescale_factor = np.sqrt(c0) # Use np.sqrt for correct handling of numpy scalars
        final_mtf = composed_mtf * rescale_factor

        # (6) Truncate to requested order
        truncated_mtf = final_mtf.truncate(order)
        return truncated_mtf


    def __neg__(self):
        neg_coeffs = {}
        for exponents, coeff in self.coefficients.items():
            neg_coeffs[exponents] = -coeff # Negate coefficient
        return MultivariateTaylorFunction(coefficients=neg_coeffs, dimension=self.dimension)


    def __truediv__(self, other):
        if isinstance(other, MultivariateTaylorFunction):
            # MTF / MTF:  self / other = self * (1/other)
            inverse_other_mtf = self._inv_mtf_internal(other) # Call internal inverse function
            return self * inverse_other_mtf
        elif isinstance(other, (int, float, np.number)):
            # MTF / scalar: Element-wise division of coefficients by scalar
            new_coeffs = {}
            for exponents, coeff_value in self.coefficients.items():
                new_coeffs[exponents] = coeff_value / other
            return MultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, np.number)):
            # scalar / MTF: other / self = other * (1/self)
            inverse_self_mtf = self._inv_mtf_internal(self) # Call internal inverse function
            return inverse_self_mtf * other # Correct order: (1/self) * scalar
        else:
            return NotImplemented

    def _inv_mtf_internal(self, mtf_instance, order=None): # Internal function for inverse
        """Internal method to calculate the Taylor expansion of 1/mtf_instance."""
        if order is None:
            order = get_global_max_order()

        # (1) Check for constant term c0
        constant_term_coeff = mtf_instance.extract_coefficient(tuple([0] * mtf_instance.dimension))
        c0 = constant_term_coeff.item()

        if abs(c0) < get_global_etol():
            raise ValueError("Cannot invert MTF with zero constant term (or very close to zero).")

        # (2) Rescale MTF so constant term is 1
        rescaled_mtf = mtf_instance / c0

        # (3) Get Taylor expansion of g(w) = 1/(1+w)
        inverse_series_1d_mtf = _inverse_taylor_1d(order)

        # (4) Compose g(f_rescaled - 1)
        composed_mtf = inverse_series_1d_mtf.compose(rescaled_mtf - MultivariateTaylorFunction.from_constant(1.0, rescaled_mtf.dimension))

        # (5) Rescale back by 1/c0
        final_mtf = composed_mtf / c0

        # (6) Truncate to requested order
        truncated_mtf = final_mtf.truncate(order)
        return truncated_mtf


    def truncate_inplace(self, order=None):
        """Truncates the ComplexMultivariateTaylorFunction *in place* to a specified order."""
        if order is None:
            order = get_global_max_order() # Use global max order if order is not provided
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")

        coefficients_to_remove = [] # Keep track of exponents to remove to avoid dict modification during iteration
        for exponents in self.coefficients:
            if sum(exponents) > order:
                coefficients_to_remove.append(exponents)

        for exponents in coefficients_to_remove:
            del self.coefficients[exponents]

        # Recalculate max_order - more efficient than tracking during removal in this case
        inferred_max_order = 0
        for exponents in self.coefficients:
            inferred_max_order = max(inferred_max_order, sum(exponents))
        self._max_order = inferred_max_order # Consider adding self._max_order attribute to __init__ and other methods

        return self # Return self for method chaining, though it's modified in-place


    def truncate(self, order=None):
        """Truncates the MultivariateTaylorFunction to a specified order."""
        if order is None:
            order = get_global_max_order() # Use global max order if order is not provided
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")

        truncated_coeffs = {}
        for exponents, coeff in self.coefficients.items():
            if sum(exponents) <= order:
                truncated_coeffs[exponents] = coeff
        return MultivariateTaylorFunction(coefficients=truncated_coeffs, dimension=self.dimension)

    def compose(self, other_mtf):
        """Performs function composition: self(other_mtf(x))."""
        if self.dimension != 1:
            raise ValueError("Composition is only supported for 1D MTF as the outer function.")

        composed_coefficients = {}
        max_order_composed = get_global_max_order()**2 # Max order of composed MTF can increase

        for self_exponents, self_coeff in self.coefficients.items():
            order_self = sum(self_exponents)
            if order_self == 0: # Constant term of self
                if tuple([0] * other_mtf.dimension) in composed_coefficients:
                    composed_coefficients[tuple([0] * other_mtf.dimension)] += self_coeff
                else:
                    composed_coefficients[tuple([0] * other_mtf.dimension)] = self_coeff
            else: # Non-constant terms of self (like u^n in cos(u))
                power_of_u = order_self # 'u' is raised to this power in the 1D Taylor series
                term_mtf = other_mtf**power_of_u # Raise the inner MTF to the power
                term_mtf_scaled = term_mtf * self_coeff # Scale by the coefficient from cos(u) series

                for term_exponents, term_coeff in term_mtf_scaled.coefficients.items():
                    if sum(term_exponents) <= max_order_composed: # Honor max order for composition
                        if term_exponents in composed_coefficients:
                            composed_coefficients[term_exponents] += term_coeff
                        else:
                            composed_coefficients[term_exponents] = term_coeff

        return MultivariateTaylorFunction(coefficients=composed_coefficients, dimension=other_mtf.dimension)

    def print_tabular(self, order=None, variable_names=None):
        print(get_tabular_string(self, order, variable_names)) # Assuming get_tabular_string is defined elsewhere

    def extract_coefficient(self, exponents):
        coefficient = self.coefficients.get(exponents)
        if coefficient is None:
            return np.array([0.0]).reshape(1) # Return zero with shape (1,) if not found
        return coefficient

    def set_coefficient(self, exponents, value):
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if len(exponents) != self.dimension:
            raise ValueError(f"Exponents tuple length must match MTF dimension ({self.dimension}).")
        if not isinstance(value, (int, float)):
            raise TypeError("Coefficient value must be a real number (int or float).")
        self.coefficients[exponents] = np.array([float(value)]).reshape(1) # Ensure value is float and shape is (1,)

    def get_max_coefficient(self):
        max_coeff = 0.0
        for coeff in self.coefficients.values():
            max_coeff = max(max_coeff, np.abs(coeff[0])) # coeff[0] to get scalar value
        return max_coeff

    def get_min_coefficient(self, tolerance=_DEFAULT_ETOL):
        min_coeff = float('inf')
        found_non_negligible = False
        for coeff in self.coefficients.values():
            coeff_abs = np.abs(coeff[0]) # coeff[0] to get scalar value
            if coeff_abs > tolerance:
                min_coeff = min(min_coeff, coeff_abs)
                found_non_negligible = True
        return min_coeff if found_non_negligible else 0.0 # Return 0 if all are negligible or zero

    def to_complex_mtf(self):
        coeffs_complex = {}
        for exponent, coeff_real in self.coefficients.items():
            coeffs_complex[exponent] = np.array([complex(coeff_real[0])]).reshape(1) # Ensure shape (1,) and convert to complex
        return ComplexMultivariateTaylorFunction(coefficients=coeffs_complex, dimension=self.dimension) # Assuming ComplexMultivariateTaylorFunction is defined elsewhere

    def __str__(self):
        if self.var_name: # Use var_name if available for concise representation
            return f"MultivariateTaylorFunction({self.var_name})"
        return get_tabular_string(self) # Assuming get_tabular_string is defined elsewhere

    def __repr__(self):
        coeff_repr = ", ".join([f"{exponent}: {repr(coeff)}" for exponent, coeff in self.coefficients.items()])
        return f"MultivariateTaylorFunction(coefficients={{{coeff_repr}}}, dimension={self.dimension})"

    def _add_coefficient_dicts(self, dict1, dict2, subtract=False): # Moved _add_coefficient_dicts inside class as internal helper
        """
        Helper function to add or subtract coefficient dictionaries.
        Used internally for operator overloading (__add__, __sub__).
        """
        combined_coeffs = defaultdict(lambda: np.array([0.0]).reshape(1)) # Initialize with shape (1,)
        factor = -1.0 if subtract else 1.0

        for exponents in dict1.keys() | dict2.keys(): # Union of keys
            coeff1 = dict1.get(exponents, np.array([0.0]).reshape(1)) # Get coeffs, default to 0 if not present, ensure shape (1,)
            coeff2 = dict2.get(exponents, np.array([0.0]).reshape(1))

            combined_val = np.array(coeff1).flatten() + factor * np.array(coeff2).flatten() # Flatten both to ensure addition works correctly
            combined_coeffs[exponents] += combined_val # Use += to add directly to defaultdict

        return combined_coeffs




# # MTFLibrary/taylor_function.py
# import numpy as np
# from collections import defaultdict
# import math

# # from MTFLibrary.complex_taylor_function import ComplexMultivariateTaylorFunction

# _GLOBAL_MAX_ORDER = None
# _GLOBAL_MAX_DIMENSION = None
# _INITIALIZED = False
# _DEFAULT_ETOL = 1e-9

# def initialize_mtf_globals(max_order, max_dimension):
#     global _GLOBAL_MAX_ORDER, _GLOBAL_MAX_DIMENSION, _INITIALIZED
#     if not isinstance(max_order, int) or max_order < 0:
#         raise ValueError("max_order must be a non-negative integer.")
#     if not isinstance(max_dimension, int) or max_dimension <= 0:
#         raise ValueError("max_dimension must be a positive integer.")
#     _GLOBAL_MAX_ORDER = max_order
#     _GLOBAL_MAX_DIMENSION = max_dimension
#     _INITIALIZED = True
#     max_num_coefficients = math.comb(max_order + max_dimension, max_dimension)
#     print(f"The max number of Taylor coefficients (order={max_order},nvars={max_dimension}): {max_num_coefficients} \n")
    
    
#     # print(f"_INITIALIZED is now: {_INITIALIZED}") # Debug print
#     # initialize_mtf_globals._called = True # Flag to indicate initialization was called



# # def initialize_mtf_globals(max_order, max_dimension):
# #     print("initialize_mtf_globals is being called...") # Debug print
# #     """
# #     Initializes global settings for MTFLibrary.

# #     Args:
# #         max_order (int): Maximum order of Taylor expansion.
# #         max_dimension (int): Maximum dimension of Multivariate Taylor Function.

# #     Raises:
# #         ValueError: if max_order is not a non-negative integer or max_dimension is not a positive integer.
# #         RuntimeError: if initialize_mtf_globals is called more than once.
# #     """
# #     global _GLOBAL_MAX_ORDER, _GLOBAL_MAX_DIMENSION, _INITIALIZED
    
# #     if _INITIALIZED:
# #         raise RuntimeError("MTF Globals are already initialized. Cannot re-initialize.")
# #     if not isinstance(max_order, int) or max_order < 0:
# #         raise ValueError("max_order must be a non-negative integer.")
# #     if not isinstance(max_dimension, int) or max_dimension <= 0:
# #         raise ValueError("max_dimension must be a positive integer.")
# #     _GLOBAL_MAX_ORDER = max_order
# #     _GLOBAL_MAX_DIMENSION = max_dimension
# #     _INITIALIZED = True




# def set_global_max_order(order):
#     """
#     Sets the global maximum order for Taylor expansions after initialization.

#     Args:
#         order (int): The new maximum order.

#     Raises:
#         ValueError: If order is not a non-negative integer.
#         RuntimeError: If globals are not initialized yet.
#     """
#     global _GLOBAL_MAX_ORDER, _INITIALIZED
#     if not _INITIALIZED:
#         raise RuntimeError("MTF Globals must be initialized before setting max order.")
#     if not isinstance(order, int) or order < 0:
#         raise ValueError("Order must be a non-negative integer.")
#     _GLOBAL_MAX_ORDER = order

# def get_global_max_order():
#     """
#     Returns the current global maximum order.

#     Raises:
#         RuntimeError: If globals are not initialized yet.

#     Returns:
#         int: The current global maximum order.
#     """
#     global _GLOBAL_MAX_ORDER, _INITIALIZED
#     if not _INITIALIZED:
#         raise RuntimeError("MTF Globals are not initialized.")
#     return _GLOBAL_MAX_ORDER

# def get_global_max_dimension():
#     """
#     Returns the global maximum dimension.

#     Raises:
#         RuntimeError: If globals are not initialized yet.

#     Returns:
#         int: The global maximum dimension.
#     """
#     global _GLOBAL_MAX_DIMENSION, _INITIALIZED
#     if not _INITIALIZED:
#         raise RuntimeError("MTF Globals are not initialized.")
#     return _GLOBAL_MAX_DIMENSION

# def set_global_etol(etol):
#     """Sets the global error tolerance for coefficients."""
#     global _DEFAULT_ETOL, _INITIALIZED
#     if not _INITIALIZED:
#         raise RuntimeError("MTF Globals must be initialized before setting error tolerance.")
#     if not isinstance(etol, float) or etol <= 0:
#         raise ValueError("Error tolerance (etol) must be a positive float.")
#     _DEFAULT_ETOL = etol

# def get_global_etol():
#     """Returns the global error tolerance."""
#     global _DEFAULT_ETOL, _INITIALIZED
#     if not _INITIALIZED:
#         raise RuntimeError("MTF Globals are not initialized.")
#     return _DEFAULT_ETOL

# # def Var(var_index):
# #     """
# #     Represents an independent variable in Multivariate Taylor Functions as a MTF.

# #     Args:
# #         var_index (int): Unique positive integer identifier for the variable.

# #     Returns:
# #         MultivariateTaylorFunction: A MultivariateTaylorFunction representing the variable x_var_index.

# #     Raises:
# #         RuntimeError: if initialize_mtf_globals has not been called.
# #         ValueError: if var_index is not a positive integer or exceeds max dimension.
# #     """
# #     global _INITIALIZED, _GLOBAL_MAX_DIMENSION
# #     if not _INITIALIZED:
# #         raise RuntimeError("MTF Globals must be initialized before creating Var objects.")
# #     if not isinstance(var_index, int) or var_index <= 0 or var_index > _GLOBAL_MAX_DIMENSION:
# #         raise ValueError(f"var_index must be a positive integer between 1 and {_GLOBAL_MAX_DIMENSION}, inclusive.")

# #     dimension = _GLOBAL_MAX_DIMENSION # Use global dimension
# #     coefficients = {}
# #     # Coefficient for the first-order term of the specified variable
# #     exponent = [0] * dimension  # Create exponent tuple of correct dimension, initially all zeros
# #     exponent[var_index - 1] = 1  # Set the exponent for the x_index variable to 1
# #     coefficients[tuple(exponent)] = np.array([1.0]).reshape(1) # Coefficient of first order term is 1, shape (1,)

# #     return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension, var_name=f'x_{var_index}')


# def _generate_exponent(order, var_index, dimension):
#     """Generates an exponent tuple of given dimension, with order at var_index and 0 elsewhere."""
#     exponent = [0] * dimension
#     exponent[var_index] = order
#     return tuple(exponent)


# def _inverse_taylor_1d(order):
#     """
#     Taylor expansion of g(w) = 1/(1+w) around 0.
#     g(w) = 1 - w + w^2 - w^3 + w^4 - ...

#     Args:
#         order (int): Order of Taylor expansion.

#     Returns:
#         MultivariateTaylorFunction: Taylor expansion of 1/(1+w) as 1D MTF.
#     """

#     coeffs_1d = {}
#     taylor_dimension_1d = 1
#     var_index_1d = 0
#     for n in range(0, order + 1):
#         coeff = (-1)**n
#         coeffs_1d[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff],dtype=np.float64).reshape(1)
#     return MultivariateTaylorFunction(coefficients=coeffs_1d, dimension=taylor_dimension_1d)


# def _sqrt_taylor_1d(order):
#     """
#     Taylor expansion of g(w) = sqrt(1+w) around 0.
#     g(w) = 1 + (1/2)w - (1/8)w^2 + (1/16)w^3 - (5/128)w^4 + ...

#     Coefficients follow pattern for (1+w)^(1/2) binomial expansion.
#     Coefficient of w^n is (1/2 choose n) = (1/2) * (1/2 - 1) * ... * (1/2 - n + 1) / n!
#     """
#     coeffs_1d = {}
#     taylor_dimension_1d = 1
#     var_index_1d = 0
#     for n in range(0, order + 1):
#         if n == 0:
#             coeff = 1.0
#         elif n == 1:
#             coeff = 0.5
#         else:
#             coeff = 1.0
#             for k in range(n):
#                 coeff *= (0.5 - k) / (k + 1) # More stable iterative calculation
#         coeffs_1d[_generate_exponent(n, var_index_1d, taylor_dimension_1d)] = np.array([coeff], dtype=np.float64).reshape(1)
#     return MultivariateTaylorFunction(coefficients=coeffs_1d, dimension=taylor_dimension_1d)




# # --- Define this function at the top level, outside the class ---
# def _create_default_coefficient_array():
#     return np.array([0.0]).reshape(1)
# # ---------------------------------------------------------------


# class MultivariateTaylorFunction:
#     """
#     Represents a multivariate Taylor function.
#     """

#     def __init__(self, coefficients, dimension, var_name=None): # Added var_name for better string representation
#         """
#         Initializes a MultivariateTaylorFunction.

#         Args:
#             coefficients (dict): Dictionary of coefficients, keys are exponent tuples,
#                                             values are NumPy arrays of shape (1,) (real coefficients).
#             dimension (int): Dimension of the multivariate function.
#             var_name (str, optional): Name of the variable if it represents a single variable (for Var function). Defaults to None.

#         Raises:
#             RuntimeError: if global settings are not initialized.
#             ValueError: if dimension is not a positive integer.
#         """
#         global _INITIALIZED, _GLOBAL_MAX_ORDER
#         if not _INITIALIZED:
#             raise RuntimeError("Global MTF settings must be initialized before creating MTF objects.")
#         if not isinstance(dimension, int) or dimension <= 0:
#             raise ValueError("Dimension must be a positive integer.")
#         # Use the globally defined function _create_default_coefficient_array
#         self.coefficients = defaultdict(_create_default_coefficient_array, coefficients) # Default to 0.0, shape (1,)
#         self.dimension = dimension
#         self.var_name = var_name # Store variable name if provided
#         self.truncate_inplace()

#     # ... (rest of the MultivariateTaylorFunction class methods) ...


#     # def __init__(self, coefficients, dimension, var_name=None): # Added var_name for better string representation
#     #     """
#     #     Initializes a MultivariateTaylorFunction.

#     #     Args:
#     #         coefficients (dict): Dictionary of coefficients, keys are exponent tuples,
#     #                               values are NumPy arrays of shape (1,) (real coefficients).
#     #         dimension (int): Dimension of the multivariate function.
#     #         var_name (str, optional): Name of the variable if it represents a single variable (for Var function). Defaults to None.

#     #     Raises:
#     #         RuntimeError: if global settings are not initialized.
#     #         ValueError: if dimension is not a positive integer.
#     #     """
#     #     global _INITIALIZED, _GLOBAL_MAX_ORDER
#     #     if not _INITIALIZED:
#     #         raise RuntimeError("Global MTF settings must be initialized before creating MTF objects.")
#     #     if not isinstance(dimension, int) or dimension <= 0:
#     #         raise ValueError("Dimension must be a positive integer.")
#     #     self.coefficients = defaultdict(lambda: np.array([0.0]).reshape(1), coefficients) # Default to 0.0, shape (1,)
#     #     self.dimension = dimension
#     #     self.var_name = var_name # Store variable name if provided
#     #     self.truncate_inplace()

#     @classmethod
#     def from_constant(cls, constant_value, dimension):
#         """
#         Creates a MultivariateTaylorFunction representing a constant.

#         Args:
#             constant_value (float): The constant value.
#             dimension (int): The dimension of the MTF.

#         Returns:
#             MultivariateTaylorFunction: MTF representing the constant.
#         """
#         coeffs = {(0,) * dimension: np.array([float(constant_value)]).reshape(1)} # Ensure shape (1,)
#         return cls(coefficients=coeffs, dimension=dimension)

#     @classmethod
#     def from_variable(cls, var_index, dimension): # Kept for internal Var implementation, not directly used externally anymore
#         """
#         Creates a MultivariateTaylorFunction representing a single variable.

#         Args:
#             var_index (int): Index of the variable (1-indexed).
#             dimension (int): The dimension of the MTF.

#         Returns:
#             MultivariateTaylorFunction: MTF representing the variable x_var_index.
#         """
#         if not (1 <= var_index <= dimension):
#             raise ValueError(f"Variable index must be between 1 and {dimension}, inclusive.")
#         exponent = [0] * dimension
#         exponent[var_index - 1] = 1
#         coeffs = {tuple(exponent): np.array([1.0]).reshape(1)} # Ensure shape (1,)
#         return cls(coefficients=coeffs, dimension=dimension, var_name=f'x_{var_index}') # Set var_name here

#     @classmethod
#     def from_complex_mtf(cls, complex_mtf):
#         """Creates a MultivariateTaylorFunction from a ComplexMultivariateTaylorFunction (real part)."""
#         coeffs_real = {}
#         for exponent, coeff_complex in complex_mtf.coefficients.items():
#             coeffs_real[exponent] = np.array([coeff_complex.real]).reshape(1) # Ensure shape (1,) and take real part
#         return cls(coefficients=coeffs_real, dimension=complex_mtf.dimension)


#     def __call__(self, evaluation_point):
#         """Evaluates the MTF at a given point."""
#         return self.eval(evaluation_point)

#     def eval(self, evaluation_point):
#         """Evaluates the MTF at a given point."""
#         if len(evaluation_point) != self.dimension:
#             raise ValueError(f"Evaluation point dimension must match MTF dimension ({self.dimension}).")
#         evaluation_point = np.array(evaluation_point) # Convert to numpy array for efficiency
#         result = np.array([0.0]).reshape(1) # Initialize result as shape (1,)
#         for exponents, coefficient in self.coefficients.items():
#             term_value = np.array([coefficient]) # Start with coefficient, shape (1, 1) if coefficient was also (1,1)
#             for i, exp in enumerate(exponents):
#                 term_value = term_value * (evaluation_point[i]**exp) #scalar multiplication
#             result += term_value.flatten() # Flatten to (1,) before adding
#         return result


#     # Operator Overloading for Real Arithmetic
#     def __add__(self, other):
#         """Adds two MTFs or an MTF and a constant."""
#         if isinstance(other, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)): #Allow CMTF addition
#             if self.dimension != other.dimension:
#                 raise ValueError("MTF dimensions must match for addition.")
#             sum_coeffs = _add_coefficient_dicts(self.coefficients, other.coefficients)
#             return MultivariateTaylorFunction(coefficients=sum_coeffs, dimension=self.dimension)
#         elif isinstance(other, (int, float)):
#             return self + MultivariateTaylorFunction.from_constant(other, self.dimension)
#         elif isinstance(other, (complex)):
#             return self + ComplexMultivariateTaylorFunction.from_constant(other, self.dimension)
#         else:
#             return NotImplemented

#     def __radd__(self, other):
#         """Right addition."""
#         return self.__add__(other) # Addition is commutative

#     def __sub__(self, other):
#         """Subtracts two MTFs or an MTF and a constant."""
#         if isinstance(other, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)): #Allow CMTF subtraction
#             if self.dimension != other.dimension:
#                 raise ValueError("MTF dimensions must match for subtraction.")
#             sub_coeffs = _add_coefficient_dicts(self.coefficients, other.coefficients, subtract=True) # Use _add_coefficient_dicts with subtract=True
#             return MultivariateTaylorFunction(coefficients=sub_coeffs, dimension=self.dimension)
#         elif isinstance(other, (int, float)):
#             return self - MultivariateTaylorFunction.from_constant(other, self.dimension)
#         else:
#             return NotImplemented

#     def __rsub__(self, other):
#         """Right subtraction (constant - MTF)."""
#         return (MultivariateTaylorFunction.from_constant(other, self.dimension)) - self

#     def __mul__(self, other):
#         """Multiplies two MTFs or an MTF and a constant."""
#         if isinstance(other, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)): #Allow CMTF multiplication
#             if self.dimension != other.dimension:
#                 raise ValueError("MTF dimensions must match for multiplication.")
#             new_coeffs = defaultdict(lambda: np.array([0.0]).reshape(1)) # Initialize with shape (1,)
#             for exponents1, coeff1 in self.coefficients.items():
#                 for exponents2, coeff2 in other.coefficients.items():
#                     new_exponent = tuple(exponents1[i] + exponents2[i] for i in range(self.dimension))
#                     new_coeffs[new_exponent] += np.array(coeff1).flatten() * np.array(coeff2).flatten() # Flatten before element-wise multiplication
#             return MultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension)
#         elif isinstance(other, (int, float)):
#             return self * MultivariateTaylorFunction.from_constant(other, self.dimension)
#         elif isinstance(other, np.ndarray) and other.shape == (1,): # Handle numpy array scalar
#             return self * MultivariateTaylorFunction.from_constant(float(other[0]), self.dimension) # Convert numpy scalar to float
#         else:
#             return NotImplemented

#     def __rmul__(self, other):
#         """Right multiplication."""
#         return self.__mul__(other) # Multiplication is commutative

#     def __imul__(self, other):
#         """In-place multiplication."""
#         if isinstance(other, (int, float)):
#             for exponent in self.coefficients:
#                 self.coefficients[exponent] *= other
#             return self
#         else:
#             return NotImplemented

#     def __pow__(self, power):
#         if power == 0.5: # Square root case
#             return self._sqrt_mtf_internal(self) # Call sqrt internal function
#         if not isinstance(power, int) or power < 0:
#             raise ValueError("Power must be a non-negative integer or 0.5 for sqrt.")
#         if power == 0:
#             return MultivariateTaylorFunction.from_constant(1.0, self.dimension)
#         if power == 1:
#             return self # Return self for power=1

#         result_mtf = MultivariateTaylorFunction.from_constant(1.0, self.dimension) # Initialize to 1
#         for _ in range(power): # Iterative multiplication
#             result_mtf = result_mtf * self
#         return result_mtf

#     def _sqrt_mtf_internal(self, mtf_instance, order=None):
#         """
#         Internal method to calculate the Taylor expansion of sqrt(mtf_instance).
#         Used by __pow__(0.5).
#         """
#         if order is None:
#             order = get_global_max_order()

#         # (1) Check for constant term c0 and ensure it's non-negative (for real sqrt)
#         constant_term_coeff = mtf_instance.extract_coefficient(tuple([0] * mtf_instance.dimension))
#         c0 = constant_term_coeff.item()

#         if c0 < -get_global_etol(): # Check if constant term is negative (real case)
#             raise ValueError("Cannot take square root of MTF with negative constant term (in real MTF context). Consider CMTF for complex square roots.")
#         if abs(c0) < get_global_etol():
#             raise ValueError("Cannot take square root of MTF with zero constant term (or very close to zero), as division by zero will occur in rescaling. Consider adding a small constant.")


#         # (2) Rescale MTF so constant term is 1
#         rescaled_mtf = mtf_instance / c0

#         # (3) Get Taylor expansion of g(w) = sqrt(1+w)
#         sqrt_series_1d_mtf = _sqrt_taylor_1d(order)

#         # (4) Compose g(f_rescaled - 1)
#         composed_mtf = sqrt_series_1d_mtf.compose(rescaled_mtf - MultivariateTaylorFunction.from_constant(1.0, rescaled_mtf.dimension))

#         # (5) Rescale back by sqrt(c0)
#         rescale_factor = np.sqrt(c0) # Use np.sqrt for correct handling of numpy scalars
#         final_mtf = composed_mtf * rescale_factor

#         # (6) Truncate to requested order
#         truncated_mtf = final_mtf.truncate(order)
#         return truncated_mtf


#     def __neg__(self):
#         """Negation of an MTF."""
#         neg_coeffs = {}
#         for exponents, coeff in self.coefficients.items():
#             neg_coeffs[exponents] = -coeff # Negate coefficient
#         return MultivariateTaylorFunction(coefficients=neg_coeffs, dimension=self.dimension)



#     def __truediv__(self, other):
#         """
#         Division operation for MultivariateTaylorFunction.
#         Handles cases: MTF / MTF, MTF / scalar.
#         For scalar / MTF, __rtruediv__ is used.
#         """
#         if isinstance(other, MultivariateTaylorFunction):
#             # MTF / MTF:  self / other = self * (1/other)
#             inverse_other_mtf = self._inv_mtf_internal(other) # Call internal inverse function
#             return self * inverse_other_mtf
#         elif isinstance(other, (int, float, np.number)):
#             # MTF / scalar: Element-wise division of coefficients by scalar
#             new_coeffs = {}
#             for exponents, coeff_value in self.coefficients.items():
#                 new_coeffs[exponents] = coeff_value / other
#             return MultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension)
#         else:
#             return NotImplemented

#     def __rtruediv__(self, other):
#         """
#         Right division operation for MultivariateTaylorFunction.
#         Handles cases: scalar / MTF.
#         """
#         if isinstance(other, (int, float, np.number)):
#             # scalar / MTF: other / self = other * (1/self)
#             inverse_self_mtf = self._inv_mtf_internal(self) # Call internal inverse function
#             return inverse_self_mtf * other # Correct order: (1/self) * scalar
#         else:
#             return NotImplemented

#     def _inv_mtf_internal(self, mtf_instance, order=None): # Internal function for inverse, to avoid recursion in __truediv__
#         """
#         Internal method to calculate the Taylor expansion of 1/mtf_instance.
#         Used by __truediv__ and __rtruediv__.
#         """
#         if order is None:
#             order = get_global_max_order()

#         # (1) Check for constant term c0
#         constant_term_coeff = mtf_instance.extract_coefficient(tuple([0] * mtf_instance.dimension))
#         c0 = constant_term_coeff.item()

#         if abs(c0) < get_global_etol():
#             raise ValueError("Cannot invert MTF with zero constant term (or very close to zero).")

#         # (2) Rescale MTF so constant term is 1
#         rescaled_mtf = mtf_instance / c0

#         # (3) Get Taylor expansion of g(w) = 1/(1+w)
#         inverse_series_1d_mtf = _inverse_taylor_1d(order)

#         # (4) Compose g(f_rescaled - 1)
#         composed_mtf = inverse_series_1d_mtf.compose(rescaled_mtf - MultivariateTaylorFunction.from_constant(1.0, rescaled_mtf.dimension))

#         # (5) Rescale back by 1/c0
#         final_mtf = composed_mtf / c0

#         # (6) Truncate to requested order
#         truncated_mtf = final_mtf.truncate(order)
#         return truncated_mtf


#     def truncate_inplace(self, order=None):
#         """
#         Truncates the ComplexMultivariateTaylorFunction *in place* to a specified order.
#         If no order is provided, it truncates to the global maximum order.
#         Modifies the MTF object directly.

#         Args:
#             order (int, optional): The order to truncate to. If None, global max order is used. Defaults to None.

#         Returns:
#             ComplexMultivariateTaylorFunction: Returns self for chaining.

#         Raises:
#             ValueError: if provided order is not a non-negative integer.
#         """
#         if order is None:
#             order = get_global_max_order() # Use global max order if order is not provided
#         elif not isinstance(order, int) or order < 0:
#             raise ValueError("Order must be a non-negative integer.")

#         coefficients_to_remove = [] # Keep track of exponents to remove to avoid dict modification during iteration
#         for exponents in self.coefficients:
#             if sum(exponents) > order:
#                 coefficients_to_remove.append(exponents)

#         for exponents in coefficients_to_remove:
#             del self.coefficients[exponents]

#         # Recalculate max_order - more efficient than tracking during removal in this case
#         inferred_max_order = 0
#         for exponents in self.coefficients:
#             inferred_max_order = max(inferred_max_order, sum(exponents))
#         self._max_order = inferred_max_order

#         return self # Return self for method chaining, though it's modified in-place
    

#     def truncate(self, order=None):
#         """
#         Truncates the MultivariateTaylorFunction to a specified order.
#         If no order is provided, it truncates to the global maximum order.

#         Args:
#             order (int, optional): The order to truncate to. If None, global max order is used. Defaults to None.

#         Returns:
#             MultivariateTaylorFunction: A new MTF truncated to the given order.

#         Raises:
#             ValueError: if provided order is not a non-negative integer.
#         """
#         if order is None:
#             order = get_global_max_order() # Use global max order if order is not provided
#         elif not isinstance(order, int) or order < 0:
#             raise ValueError("Order must be a non-negative integer.")

#         truncated_coeffs = {}
#         for exponents, coeff in self.coefficients.items():
#             if sum(exponents) <= order:
#                 truncated_coeffs[exponents] = coeff
#         return MultivariateTaylorFunction(coefficients=truncated_coeffs, dimension=self.dimension)

#     def compose(self, other_mtf):
#         """
#         Performs function composition: self(other_mtf(x)).

#         Args:
#             other_mtf (MultivariateTaylorFunction): The MTF to be substituted into self.

#         Returns:
#             MultivariateTaylorFunction: The composed MTF.
#         """
#         if self.dimension != 1:
#             raise ValueError("Composition is only supported for 1D MTF as the outer function.")

#         composed_coefficients = {}
#         max_order_composed = get_global_max_order()**2 # Max order of composed MTF can increase

#         for self_exponents, self_coeff in self.coefficients.items():
#             order_self = sum(self_exponents)
#             if order_self == 0: # Constant term of self
#                 if tuple([0] * other_mtf.dimension) in composed_coefficients:
#                     composed_coefficients[tuple([0] * other_mtf.dimension)] += self_coeff
#                 else:
#                     composed_coefficients[tuple([0] * other_mtf.dimension)] = self_coeff
#             else: # Non-constant terms of self (like u^n in cos(u))
#                 power_of_u = order_self # 'u' is raised to this power in the 1D Taylor series
#                 term_mtf = other_mtf**power_of_u # Raise the inner MTF to the power
#                 term_mtf_scaled = term_mtf * self_coeff # Scale by the coefficient from cos(u) series

#                 for term_exponents, term_coeff in term_mtf_scaled.coefficients.items():
#                     if sum(term_exponents) <= max_order_composed: # Honor max order for composition
#                         if term_exponents in composed_coefficients:
#                             composed_coefficients[term_exponents] += term_coeff
#                         else:
#                             composed_coefficients[term_exponents] = term_coeff

#         return MultivariateTaylorFunction(coefficients=composed_coefficients, dimension=other_mtf.dimension)
    
#     def print_tabular(self, order=None, variable_names=None):
#         """Prints the tabular string representation of the MTF."""
#         print(get_tabular_string(self, order, variable_names))


#     def extract_coefficient(self, exponents):
#         """Extracts the coefficient for a given exponent tuple."""
#         coefficient = self.coefficients.get(exponents)
#         if coefficient is None:
#             return np.array([0.0]).reshape(1) # Return zero with shape (1,) if not found
#         return coefficient

#     def set_coefficient(self, exponents, value):
#         """Sets the coefficient for a given exponent tuple."""
#         if not isinstance(exponents, tuple):
#             raise TypeError("Exponents must be a tuple.")
#         if len(exponents) != self.dimension:
#             raise ValueError(f"Exponents tuple length must match MTF dimension ({self.dimension}).")
#         if not isinstance(value, (int, float)):
#             raise TypeError("Coefficient value must be a real number (int or float).")
#         self.coefficients[exponents] = np.array([float(value)]).reshape(1) # Ensure value is float and shape is (1,)

#     def get_max_coefficient(self):
#         """Returns the maximum absolute value of any coefficient."""
#         max_coeff = 0.0
#         for coeff in self.coefficients.values():
#             max_coeff = max(max_coeff, np.abs(coeff[0])) # coeff[0] to get scalar value
#         return max_coeff

#     def get_min_coefficient(self, tolerance=_DEFAULT_ETOL):
#         """Returns the minimum absolute value of any non-negligible coefficient."""
#         min_coeff = float('inf')
#         found_non_negligible = False
#         for coeff in self.coefficients.values():
#             coeff_abs = np.abs(coeff[0]) # coeff[0] to get scalar value
#             if coeff_abs > tolerance:
#                 min_coeff = min(min_coeff, coeff_abs)
#                 found_non_negligible = True
#         return min_coeff if found_non_negligible else 0.0 # Return 0 if all are negligible or zero

#     def to_complex_mtf(self):
#         """Converts a Real MTF to a ComplexMTF."""
#         coeffs_complex = {}
#         for exponent, coeff_real in self.coefficients.items():
#             coeffs_complex[exponent] = np.array([complex(coeff_real[0])]).reshape(1) # Ensure shape (1,) and convert to complex
#         return ComplexMultivariateTaylorFunction(coefficients=coeffs_complex, dimension=self.dimension)


#     def __str__(self):
#         if self.var_name: # Use var_name if available for concise representation
#             return f"MultivariateTaylorFunction({self.var_name})"
#         return get_tabular_string(self)

#     def __repr__(self):
#         coeff_repr = ", ".join([f"{exponent}: {repr(coeff)}" for exponent, coeff in self.coefficients.items()])
#         return f"MultivariateTaylorFunction(coefficients={{{coeff_repr}}}, dimension={self.dimension})"



class ComplexMultivariateTaylorFunction:
    """
    Represents a multivariate Taylor function with complex coefficients.

    This class extends the concept of MultivariateTaylorFunction to handle
    Taylor expansions where coefficients are complex numbers. It supports
    similar operations as MultivariateTaylorFunction, adapted for complex arithmetic.
    """

    def __init__(self, coefficients, dimension=None):
        """
        Initializes a ComplexMultivariateTaylorFunction object.

        Args:
            coefficients (dict): A dictionary where keys are tuples representing
                exponents and values are complex numpy arrays representing coefficients.
            dimension (int, optional): The dimension of the multivariate Taylor function.
                If None, it will be inferred from the exponents in coefficients.
                Defaults to None.

        Raises:
            TypeError: if coefficients is not a dictionary.
            ValueError: if dimension is not a positive integer, or if coefficients are not complex.
        """
        global _DEFAULT_ETOL
        if not isinstance(coefficients, dict):
            raise TypeError("Coefficients must be a dictionary.")

        self.coefficients = {}  # Use a new dictionary to ensure immutability and proper storage
        self._dimension = dimension

        inferred_dimension = 0
        inferred_max_order = 0

        for exponents, coeff_value in coefficients.items():
            if not isinstance(exponents, tuple):
                raise TypeError("Exponents must be tuples.")
            if not isinstance(coeff_value, np.ndarray) or coeff_value.dtype != np.complex128:
                raise ValueError("Coefficients must be complex numpy arrays.")

            if dimension is not None and len(exponents) != dimension:
                raise ValueError(f"Exponent tuple length ({len(exponents)}) does not match dimension ({dimension}).")

            if any(not isinstance(exp, int) or exp < 0 for exp in exponents):
                raise ValueError("Exponents must be non-negative integers.")

            # Store coefficient only if its magnitude is above global tolerance ETOL
            if np.max(np.abs(coeff_value)) > _DEFAULT_ETOL: # Use global ETOL here
                self.coefficients[exponents] = coeff_value

        if dimension is None and inferred_dimension > 0:
            self._dimension = inferred_dimension
        elif dimension is None:
            self._dimension = 1 # Default dimension if no coefficients provided

        if self._dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")


    @classmethod
    def from_constant(cls, constant_value, dimension):
        """
        Creates a ComplexMultivariateTaylorFunction representing a constant value.

        Args:
            constant_value (complex or number): The constant complex value.
            dimension (int): The dimension of the MTF.

        Returns:
            ComplexMultivariateTaylorFunction: An MTF representing the constant.
        """
        if not isinstance(constant_value, (complex, int, float)):
            raise TypeError("Constant value must be a complex or real number.")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")

        constant_value_complex = np.array([complex(constant_value)], dtype=np.complex128).reshape(1) # Ensure complex numpy array
        coeffs = {tuple([0] * dimension): constant_value_complex}
        return cls(coefficients=coeffs, dimension=dimension)


    @classmethod
    def from_variable(cls, var_index, dimension):
        """
        Creates a ComplexMultivariateTaylorFunction representing a single variable.

        Args:
            var_index (int): The index of the variable (1-indexed).
            dimension (int): The dimension of the MTF.

        Returns:
            ComplexMultivariateTaylorFunction: An MTF representing the variable.

        Raises:
            ValueError: if var_index is out of range [1, dimension].
        """
        if not isinstance(var_index, int) or var_index <= 0 or var_index > dimension:
            raise ValueError(f"Variable index must be in range [1, dimension], got {var_index} for dimension {dimension}.")

        coeffs = {tuple([(1 if i == var_index - 1 else 0) for i in range(dimension)]): np.array([1.0 + 0.0j], dtype=np.complex128).reshape(1)} # Complex 1.0
        return cls(coefficients=coeffs, dimension=dimension)


    @property
    def dimension(self):
        """Returns the dimension of the multivariate Taylor function."""
        return self._dimension

    @property
    def max_order(self):
        """Returns the maximum order of the Taylor function."""
        global _GLOBAL_MAX_ORDER
        return _GLOBAL_MAX_ORDER

    def truncate(self, order=None):
        """
        Truncates the ComplexMultivariateTaylorFunction to a specified order.
        If no order is provided, it truncates to the global maximum order.

        Args:
            order (int, optional): The order to truncate to. If None, global max order is used. Defaults to None.

        Returns:
            ComplexMultivariateTaylorFunction: A new MTF truncated to the given order.

        Raises:
            ValueError: if provided order is not a non-negative integer.
        """
        if order is None:
            order = get_global_max_order() # Use global max order if order is not provided
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")

        truncated_coeffs = {}
        for exponents, coeff_value in self.coefficients.items():
            if sum(exponents) <= order:
                truncated_coeffs[exponents] = coeff_value
        return ComplexMultivariateTaylorFunction(coefficients=truncated_coeffs, dimension=self.dimension) # etol removed

    def extract_coefficient(self, exponents):
        """
        Extracts the coefficient for a given tuple of exponents.

        If the coefficient for the given exponents is not explicitly stored,
        it returns a complex zero numpy array of shape (1,).

        Args:
            exponents (tuple): A tuple of non-negative integers representing the exponents.

        Returns:
            numpy.ndarray: The complex coefficient as a numpy array of shape (1,).
                           Returns a complex zero array if the coefficient is not found.
        """
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if len(exponents) != self.dimension:
            raise ValueError(f"Exponent tuple length ({len(exponents)}) does not match MTF dimension ({self.dimension}).")
        if exponents in self.coefficients:
            return self.coefficients[exponents]
        else:
            return np.array([0.0j], dtype=np.complex128).reshape(1) # Complex zero


    def set_coefficient(self, exponents, value):
        """
        Sets the coefficient for a given tuple of exponents.

        Args:
            exponents (tuple): A tuple of non-negative integers representing the exponents.
            value (complex or number): The complex value to set as the coefficient.

        Raises:
            TypeError: if exponents is not a tuple or value is not numeric.
            ValueError: if exponent tuple length does not match dimension, or exponents are invalid.
        """
        global _DEFAULT_ETOL
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if not isinstance(value, (complex, int, float, np.number)): # Allow numpy numbers as well
            raise TypeError("Coefficient value must be a complex or real number.")
        if len(exponents) != self.dimension:
            raise ValueError(f"Exponent tuple length must be {self.dimension}.")
        if any(not isinstance(exp, int) or exp < 0 for exp in exponents):
            raise ValueError("Exponents must be non-negative integers.")

        coeff_value_complex = np.array([complex(value)], dtype=np.complex128).reshape(1) # Ensure complex numpy array

        if np.max(np.abs(coeff_value_complex)) > _DEFAULT_ETOL: # Use global ETOL here
            self.coefficients[exponents] = coeff_value_complex
        elif exponents in self.coefficients: # If setting to (effectively) zero, remove from dict to save space
            del self.coefficients[exponents]


    def eval(self, point):
        """
        Evaluates the ComplexMultivariateTaylorFunction at a given point.

        Args:
            point (list or numpy.ndarray): A list or numpy array representing the point
                in the function's domain. Must be of the same dimension as the MTF.

        Returns:
            numpy.ndarray: The complex value of the MTF evaluated at the given point,
                           as a numpy array of shape (1,).

        Raises:
            ValueError: if the dimension of the point does not match the MTF's dimension.
            TypeError: if point is not list or numpy array.
        """
        if not isinstance(point, (list, tuple, np.ndarray)):
            raise TypeError("Evaluation point must be a list, tuple, or numpy array.")
        point = np.array(point) # Ensure numpy array for operations
        if point.shape != (self.dimension,): # Expecting 1D array of dimension size
            raise ValueError(f"Evaluation point must be of dimension {self.dimension}.")

        result = np.array([0.0j], dtype=np.complex128).reshape(1) # Initialize as complex zero
        for exponents, coeff_value in self.coefficients.items():
            term_value = coeff_value
            for i in range(self.dimension):
                term_value = term_value * (point[i] ** exponents[i]) # Power operation
            result = result + term_value
        return result


    def get_max_coefficient(self):
        """
        Returns the maximum magnitude of all coefficients in the MTF.

        Returns:
            float: The maximum magnitude. Returns 0.0 if there are no coefficients above tolerance.
        """
        max_coeff_magnitude = 0.0
        for coeff_value in self.coefficients.values():
            max_coeff_magnitude = max(max_coeff_magnitude, np.max(np.abs(coeff_value))) # Magnitude for complex

        return max_coeff_magnitude


    def get_min_coefficient(self, tolerance=None):
        """
        Returns the minimum magnitude of non-negligible coefficients in the MTF.

        Coefficients with magnitude below the error tolerance `ETOL` are considered negligible
        by default (or `tolerance` if provided).

        Args:
            tolerance (float, optional): Tolerance value below which coefficients are considered negligible.
                                         If None, the global ETOL is used. Defaults to None.

        Returns:
            float: The minimum magnitude of non-negligible coefficients. Returns 0.0 if all coefficients are negligible.
        """
        global _DEFAULT_ETOL
        min_magnitude = float('inf')
        found_non_negligible = False
        current_tolerance = tolerance if tolerance is not None else _DEFAULT_ETOL # Use global ETOL here

        for coeff_value in self.coefficients.values():
            magnitude = np.max(np.abs(coeff_value)) # Magnitude for complex
            if magnitude > current_tolerance:
                min_magnitude = min(min_magnitude, magnitude)
                found_non_negligible = True

        return min_magnitude if found_non_negligible else 0.0


    def conjugate(self):
        """
        Returns the complex conjugate of the ComplexMultivariateTaylorFunction.

        Returns:
            ComplexMultivariateTaylorFunction: The conjugated MTF.
        """
        conjugated_coeffs = {}
        for exponents, coeff_value in self.coefficients.items():
            conjugated_coeffs[exponents] = np.conjugate(coeff_value) # Conjugate of complex coefficients
        return ComplexMultivariateTaylorFunction(coefficients=conjugated_coeffs, dimension=self.dimension) # etol removed


    def real_part(self):
        """
        Returns the real part of the ComplexMultivariateTaylorFunction as a MultivariateTaylorFunction.

        Returns:
            MultivariateTaylorFunction: The real part MTF.
        """
        real_coeffs = {}
        for exponents, coeff_value in self.coefficients.items():
            real_coeffs[exponents] = np.real(coeff_value) # Real part of complex coefficients
        return MultivariateTaylorFunction(coefficients=real_coeffs, dimension=self.dimension) # etol removed


    def imag_part(self):
        """
        Returns the imaginary part of the ComplexMultivariateTaylorFunction as a MultivariateTaylorFunction.

        Returns:
            MultivariateTaylorFunction: The imaginary part MTF.
        """
        imag_coeffs = {}
        for exponents, coeff_value in self.coefficients.items():
            imag_coeffs[exponents] = np.imag(coeff_value) # Imaginary part of complex coefficients
        return MultivariateTaylorFunction(coefficients=imag_coeffs, dimension=self.dimension) # etol removed


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


    # --- Arithmetic Operations ---

    def __add__(self, other):
        """Defines addition operation for ComplexMultivariateTaylorFunction."""
        if isinstance(other, ComplexMultivariateTaylorFunction):
            if self.dimension != other.dimension:
                raise ValueError("Dimensions of MTFs must match for addition.")
            sum_coeffs = self.coefficients.copy() # Start with coefficients of self
            for exponents, coeff_value in other.coefficients.items():
                if exponents in sum_coeffs:
                    sum_coeffs[exponents] = sum_coeffs[exponents] + coeff_value # Add coefficients if exponents exist
                else:
                    sum_coeffs[exponents] = coeff_value # Otherwise, just add the new coefficient
            return ComplexMultivariateTaylorFunction(coefficients=sum_coeffs, dimension=self.dimension) # etol removed

        elif isinstance(other, (complex, int, float)):
            constant_mtf = ComplexMultivariateTaylorFunction.from_constant(other, self.dimension)
            return self + constant_mtf # Delegate to MTF + MTF addition

        else:
            return NotImplemented

    def __radd__(self, other):
        """Defines right addition operation."""
        return self + other # Addition is commutative


    def __sub__(self, other):
        """Defines subtraction operation."""
        if isinstance(other, ComplexMultivariateTaylorFunction):
            if self.dimension != other.dimension:
                raise ValueError("Dimensions of MTFs must match for subtraction.")
            diff_coeffs = self.coefficients.copy() # Start with coefficients of self
            for exponents, coeff_value in other.coefficients.items():
                if exponents in diff_coeffs:
                    diff_coeffs[exponents] = diff_coeffs[exponents] - coeff_value # Subtract coefficients
                else:
                    diff_coeffs[exponents] = -coeff_value # Subtract coefficient of other from zero
            return ComplexMultivariateTaylorFunction(coefficients=diff_coeffs, dimension=self.dimension) # etol removed

        elif isinstance(other, (complex, int, float)):
            constant_mtf = ComplexMultivariateTaylorFunction.from_constant(other, self.dimension)
            return self - constant_mtf # Delegate to MTF - MTF subtraction

        else:
            return NotImplemented


    def __rsub__(self, other):
        """Defines right subtraction operation (e.g., constant - MTF)."""
        constant_mtf = ComplexMultivariateTaylorFunction.from_constant(other, self.dimension)
        return constant_mtf - self # Use commutative subtraction, constant MTF - self


    def __mul__(self, other):
        """Defines multiplication operation."""
        if isinstance(other, ComplexMultivariateTaylorFunction):
            if self.dimension != other.dimension:
                raise ValueError("Dimensions of MTFs must match for multiplication.")
            prod_coeffs = {}
            new_max_order = self.max_order + other.max_order # Still calculating new max order for internal use

            for exp1, coeff1 in self.coefficients.items():
                for exp2, coeff2 in other.coefficients.items():
                    result_exponent = tuple(exp1[i] + exp2[i] for i in range(self.dimension))
                    if sum(result_exponent) <= new_max_order: # Honor max_order - using inferred max order for logic
                        term_coefficient = coeff1 * coeff2
                        if result_exponent in prod_coeffs:
                            prod_coeffs[result_exponent] = prod_coeffs[result_exponent] + term_coefficient
                        else:
                            prod_coeffs[result_exponent] = term_coefficient

            return ComplexMultivariateTaylorFunction(coefficients=prod_coeffs, dimension=self.dimension) # etol removed

        elif isinstance(other, (complex, int, float)):
            scaled_coeffs = {}
            for exponents, coeff_value in self.coefficients.items():
                scaled_coeffs[exponents] = coeff_value * other # Scale each coefficient
            return ComplexMultivariateTaylorFunction(coefficients=scaled_coeffs, dimension=self.dimension) # etol removed

        else:
            return NotImplemented


    def __rmul__(self, other):
        """Defines right multiplication operation (commutative)."""
        return self * other # Multiplication is commutative


    def __pow__(self, power):
        if power == 0.5: # Square root case
            return self._sqrt_cmtf_internal(self) # Call sqrt internal function
        if not isinstance(power, int) or power < 0:
            raise ValueError("Power must be a non-negative integer or 0.5 for sqrt.")
        if power == 0:
            return ComplexMultivariateTaylorFunction.from_constant(1.0, self.dimension)
        if power == 1:
            return self # Return self for power=1

        result_cmtf = ComplexMultivariateTaylorFunction.from_constant(1.0, self.dimension) # Initialize to 1
        for _ in range(power): # Iterative multiplication
            result_cmtf = result_cmtf * self
        return result_cmtf

    def _sqrt_cmtf_internal(self, cmtf_instance, order=None):
        """
        Internal method to calculate the Taylor expansion of sqrt(cmtf_instance) for CMTF.
        Used by __pow__(0.5).
        """
        if order is None:
            order = get_global_max_order()

        # (1) Check for constant term c0
        constant_term_coeff = cmtf_instance.extract_coefficient(tuple([0] * cmtf_instance.dimension))
        c0 = constant_term_coeff.item()

        if abs(c0) < get_global_etol():
            raise ValueError("Cannot take square root of CMTF with zero constant term (or very close to zero), as division by zero will occur in rescaling. Consider adding a small constant.")


        # (2) Rescale CMTF so constant term is 1
        rescaled_cmtf = cmtf_instance / c0

        # (3) Get Taylor expansion of g(w) = sqrt(1+w) - reuse real series
        sqrt_series_1d_mtf = _sqrt_taylor_1d(order) # We can reuse the real series for sqrt(1+w)

        # (4) Compose g(f_rescaled - 1) - using rescaled CMTF
        composed_cmtf = sqrt_series_1d_mtf.compose(rescaled_cmtf - ComplexMultivariateTaylorFunction.from_constant(1.0, rescaled_cmtf.dimension))

        # (5) Rescale back by sqrt(c0) - now using cmath.sqrt for complex sqrt
        rescale_factor = np.sqrt(c0) # Use np.sqrt which handles complex numbers correctly
        final_cmtf = composed_cmtf * rescale_factor

        # (6) Truncate to requested order
        truncated_cmtf = final_cmtf.truncate(order)
        return truncated_cmtf

    def __neg__(self):
        """Defines negation operation (-MTF)."""
        negated_coeffs = {}
        for exponents, coeff_value in self.coefficients.items():
            negated_coeffs[exponents] = -coeff_value # Negate each coefficient
        return ComplexMultivariateTaylorFunction(coefficients=negated_coeffs, dimension=self.dimension) # etol removed


    def __truediv__(self, other):
        """
        Division operation for ComplexMultivariateTaylorFunction.
        Handles cases: CMTF / CMTF, CMTF / MTF, CMTF / scalar.
        For scalar / CMTF, mtf / CMTF, __rtruediv__ is used.
        """
        if isinstance(other, ComplexMultivariateTaylorFunction):
            # CMTF / CMTF: self / other = self * (1/other)
            inverse_other_cmtf = self._inv_cmtf_internal(other) # Call internal inverse for CMTF
            return self * inverse_other_cmtf
        elif isinstance(other, MultivariateTaylorFunction):
            # CMTF / MTF: self / other = self * (1/other)
            inverse_other_mtf = other._inv_mtf_internal(other) # Use MTF's inverse function
            return self * convert_to_cmtf(inverse_other_mtf) # Multiply CMTF by converted inverse MTF
        elif isinstance(other, (int, float, complex, np.number)):
            # CMTF / scalar: Element-wise division by scalar
            new_coeffs = {}
            for exponents, coeff_value in self.coefficients.items():
                new_coeffs[exponents] = coeff_value / other
            return ComplexMultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """
        Right division operation for ComplexMultivariateTaylorFunction.
        Handles cases: scalar / CMTF, MTF / CMTF.
        """
        if isinstance(other, (int, float, complex, np.number)):
            # scalar / CMTF: other / self = other * (1/self)
            inverse_self_cmtf = self._inv_cmtf_internal(self) # Call internal inverse for CMTF
            return inverse_self_cmtf * other # Correct order: (1/self) * scalar
        elif isinstance(other, MultivariateTaylorFunction):
            # MTF / CMTF: other / self = other * (1/self)
            inverse_self_cmtf = self._inv_cmtf_internal(self) # Inverse of CMTF
            return convert_to_cmtf(other) * inverse_self_cmtf # Multiply CMTF inverse by converted MTF
        else:
            return NotImplemented


    def _inv_cmtf_internal(self, cmtf_instance, order=None): # Internal function for inverse for CMTF, to avoid recursion in __truediv__
        """
        Internal method to calculate the Taylor expansion of 1/cmtf_instance for CMTF.
        Used by __truediv__ and __rtruediv__.
        """
        if order is None:
            order = get_global_max_order()

        # (1) Check for constant term c0
        constant_term_coeff = cmtf_instance.extract_coefficient(tuple([0] * cmtf_instance.dimension))
        c0 = constant_term_coeff.item()

        if abs(c0) < get_global_etol():
            raise ValueError("Cannot invert CMTF with zero constant term (or very close to zero).")

        # (2) Rescale CMTF so constant term is 1
        rescaled_cmtf = cmtf_instance / c0

        # (3) Get Taylor expansion of g(w) = 1/(1+w) - reuse real series
        inverse_series_1d_mtf = _inverse_taylor_1d(order)

        # (4) Compose g(f_rescaled - 1) - using rescaled CMTF
        composed_cmtf = inverse_series_1d_mtf.compose(rescaled_cmtf - ComplexMultivariateTaylorFunction.from_constant(1.0, rescaled_cmtf.dimension))

        # (5) Rescale back by 1/c0
        final_cmtf = composed_cmtf / c0

        # (6) Truncate to requested order
        truncated_cmtf = final_cmtf.truncate(order)
        return truncated_cmtf

    # def __str__(self):
    #     """Returns a string representation of the ComplexMultivariateTaylorFunction using tabular format."""
    #     if not self.coefficients:
    #         return "0 (Complex MTF)"

    #     # Assuming get_tabular_string is a function available in scope
    #     # that takes coefficients, dimension, and possibly max_order as arguments
    #     try:
    #         from MTFLibrary.utils import get_tabular_string  # Or wherever it's defined
    #         return get_tabular_string(self.coefficients, self.dimension, is_complex=True) # Pass is_complex=True, removed max_order
    #     except ImportError:
    #         # Fallback to basic string representation if get_tabular_string is not found
    #         terms = []
    #         for exponents, coeff_value in self.coefficients.items():
    #             term_str = str(coeff_value.item()) # Get scalar complex value as string
    #             var_part = "".join([f"x{i+1}^{exp}" if exp > 0 else "" for i, exp in enumerate(exponents)]) # Variable part like x1^2x2
    #             if var_part:
    #                 term_str += var_part
    #             terms.append(term_str)
    #         return " + ".join(terms) + " (Complex MTF - Tabular String Failed)" # Indicate tabular failed


    def __repr__(self):
        """Returns a more detailed string representation, useful for debugging."""
        coeff_repr = ", ".join([f"{exponent}: {repr(coeff)}" for exponent, coeff in self.coefficients.items()])
        return f"ComplexMultivariateTaylorFunction(coefficients={{{coeff_repr}}}, dimension={self.dimension})"

    def __str__(self):
        if self.var_name: # Use var_name if available for concise representation
            return f"MultivariateTaylorFunction({self.var_name})"
        return get_tabular_string(self)

    def print_tabular(self, order=None, variable_names=None):
        """Prints the tabular string representation of the MTF."""
        print(get_tabular_string(self, order, variable_names))




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


def convert_to_mtf(input_val, dimension=None):
    """
    Converts various input types to MultivariateTaylorFunction or ComplexMultivariateTaylorFunction.

    Args:
        input_val: Input value to convert (scalar, Var (now function), MTF, CMTF, np.number, 0D np.ndarray).
        dimension (int, optional): Dimension to use if creating a new MTF/CMTF from scalar.
                                     Defaults to 1 if input_val is a scalar (int, float, complex) and dimension is not provided.
                                     Must be explicitly provided for other scalar types if no MTF/CMTF context is available.

    Returns:
        MultivariateTaylorFunction or ComplexMultivariateTaylorFunction: The converted MTF or CMTF.

    Raises:
        TypeError: if input_val type is not supported.
        ValueError: if dimension is not a positive integer.
        RuntimeError: if globals are not initialized when needed.
    """
    if isinstance(input_val, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)):
        return input_val
    elif isinstance(input_val, (int, float)):
        if dimension is None:
            dimension = 1  # Default to dimension 1 for int/float if not provided
        return MultivariateTaylorFunction.from_constant(input_val, dimension)
    elif isinstance(input_val, complex):
        if dimension is None:
            dimension = 1  # Default to dimension 1 for complex if not provided
        return ComplexMultivariateTaylorFunction.from_constant(input_val, dimension)
    elif isinstance(input_val, np.ndarray) and input_val.shape == (): # 0D numpy array (numpy scalar)
        return convert_to_mtf(input_val.item(), dimension) # Recursively convert numpy scalar to python scalar then to MTF
    elif isinstance(input_val, np.number): #numpy numbers like np.int64, np.float64, np.complex128
        return convert_to_mtf(float(input_val), dimension) # Convert numpy number to python scalar then to MTF
    elif callable(input_val) and input_val.__name__ == 'Var': # Check if it's the Var function
        return input_val(dimension) # call the Var function with dimension (assuming dimension needed)
    else:
        raise TypeError(f"Unsupported input type: {type(input_val)}. Cannot convert to MTF/CMTF.")


 

def get_tabular_string(mtf_instance, order=None, variable_names=None):
    """
    Returns a string representation of a Multivariate Taylor Function (MTF) or
    Complex Multivariate Taylor Function (CMTF) instance in tabular format.

    Args:
        mtf_instance: An instance of either MultivariateTaylorFunction or ComplexMultivariateTaylorFunction.
                       It is assumed that the instance has attributes:
                           - coefficients (dict):  Taylor coefficients dictionary.
                           - dimension (int): Dimension of the MTF.
                       and access to functions:
                           - get_global_max_order() (or a way to get max order, if order is None)
                           - get_global_etol() (or a way to get error tolerance)
        order (int, optional): The maximum order of terms to include in the table.
                                 Defaults to mtf_instance.get_global_max_order() if available, or None.
        variable_names (list of str, optional): List of variable names to use as headers.
                                                 Defaults to ['x_1', 'x_2', ... , 'x_dimension'].

    Returns:
        str: A string representing the MTF in tabular format.
             Returns "MultivariateTaylorFunction (truncated or zero)" if no terms to display.
    """
    coefficients = mtf_instance.coefficients
    dimension = mtf_instance.dimension

    if order is None:
        if hasattr(mtf_instance, 'get_global_max_order'): # Check if instance has get_global_max_order method
            order = mtf_instance.get_global_max_order()
        else:
            order = get_global_max_order() # Fallback to global function if instance method not available, assume get_global_max_order() is defined elsewhere

    if variable_names is None:
        variable_names = [f'x_{i+1}' for i in range(dimension)]
    headers = ["I", "Coefficient", "Order", "Exponents"] # Keep "Exponents", Removed variable names
    rows = []
    term_index = 1
    etol = get_global_etol() # Assume get_global_etol() is defined elsewhere, get error tolerance here once

    for exponents, coeff in sorted(coefficients.items(), key=lambda item: sum(item[0])): #Sort by order
        if sum(exponents) <= order and np.any(np.abs(coeff) > etol): # Respect order and error tolerance
            exponent_str = " ".join(map(str, exponents)) # Keep exponent_str

            # Format coefficient to string, handling both real and complex
            if np.iscomplexobj(coeff):
                coeff_str = f"{coeff[0].real:.8f}{coeff[0].imag:+8f}j" # Format complex: a+bj, a-bj
            else:
                coeff_str = f"{coeff[0]:+16.16e}" # Format real as before

            rows.append([f"{term_index: <4}", coeff_str, str(sum(exponents)), exponent_str]) # Added formatted coeff_str
            term_index += 1

    if not rows: # Handle case with no terms to display
        return "MultivariateTaylorFunction (truncated or zero)"

    # Calculate column widths
    column_widths = []
    current_header_index = 0
    for header in headers:
        if header == "I": # Fixed width for "I" column
            column_widths.append(4 + 2) # Fixed width of 4 + 2 padding
        else:
            column_widths.append(max(len(header), max(len(row[current_header_index]) for row in rows)) + 2) # +2 for padding
        current_header_index += 1

    # Format header
    header_row = "| " + "| ".join(headers[i].ljust(column_widths[i]-2) for i in range(len(headers))) + "|"
    separator = "|" + "|".join("-" * (w-1) for w in column_widths) + "|"
    table_str = header_row + "\n" + separator + "\n"

    # Format rows
    for row in rows:
        table_str += "| " + "| ".join(row[i].ljust(column_widths[i]-2) for i in range(len(headers))) + "|\n"
    return table_str


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
    elif isinstance(variable, MultivariateTaylorFunction): # Var function returns MTF now
        # Convert MTF to CMTF: just need to ensure coefficients are complex type
        cmtf_coeffs = {}
        for exponents, coeff_value in variable.coefficients.items():
            cmtf_coeffs[exponents] = coeff_value.astype(np.complex128) # Convert to complex coefficients
        return ComplexMultivariateTaylorFunction(coefficients=cmtf_coeffs, dimension=variable.dimension)
    elif isinstance(variable, (int, float, complex, np.number)):
        # Create constant CMTF from scalar
        return ComplexMultivariateTaylorFunction.from_constant(variable, dimension=1 if isinstance(variable, MultivariateTaylorFunction) else (variable.dimension if isinstance(variable, MultivariateTaylorFunction) else 1) ) # Dimension doesn't strictly matter for constant
    else:
        raise TypeError("Unsupported type for conversion to ComplexMultivariateTaylorFunction.")
