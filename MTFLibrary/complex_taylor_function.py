# MTFLibrary/complex_taylor_function.py
import numpy as np
from collections import defaultdict
from MTFLibrary.taylor_function import (initialize_mtf_globals, get_global_max_order, 
        get_global_max_dimension, set_global_max_order, set_global_etol, 
        get_global_etol, convert_to_mtf, get_mtf_initialized_status, 
        get_tabular_string, MultivariateTaylorFunctionBase)
# from MTFLibrary.elementary_functions import (cos_taylor, sin_taylor, 
#         exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor, 
#         sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, 
#         arctanh_taylor)
# from MTFLibrary.MTFExtended import Var, MultivariateTaylorFunction


class ComplexMultivariateTaylorFunction(MultivariateTaylorFunctionBase):
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
        # _DEFAULT_ETOL = get_global_etol()

        # global _DEFAULT_ETOL
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
            if np.max(np.abs(coeff_value)) > get_global_etol(): # Use global ETOL here
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
        # global _GLOBAL_MAX_ORDER
        return get_global_max_order()

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
        # global get_global_etol()
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if not isinstance(value, (complex, int, float, np.number)): # Allow numpy numbers as well
            raise TypeError("Coefficient value must be a complex or real number.")
        if len(exponents) != self.dimension:
            raise ValueError(f"Exponent tuple length must be {self.dimension}.")
        if any(not isinstance(exp, int) or exp < 0 for exp in exponents):
            raise ValueError("Exponents must be non-negative integers.")

        coeff_value_complex = np.array([complex(value)], dtype=np.complex128).reshape(1) # Ensure complex numpy array

        if np.max(np.abs(coeff_value_complex)) > get_global_etol(): # Use global ETOL here
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
        # global _DEFAULT_ETOL
        min_magnitude = float('inf')
        found_non_negligible = False
        current_tolerance = tolerance if tolerance is not None else get_global_etol() # Use global ETOL here

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
        return MultivariateTaylorFunctionBase(coefficients=real_coeffs, dimension=self.dimension) # etol removed


    def imag_part(self):
        """
        Returns the imaginary part of the ComplexMultivariateTaylorFunction as a MultivariateTaylorFunction.

        Returns:
            MultivariateTaylorFunction: The imaginary part MTF.
        """
        imag_coeffs = {}
        for exponents, coeff_value in self.coefficients.items():
            imag_coeffs[exponents] = np.imag(coeff_value) # Imaginary part of complex coefficients
        return MultivariateTaylorFunctionBase(coefficients=imag_coeffs, dimension=self.dimension) # etol removed


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
        sqrt_series_1d_mtf = MultivariateTaylorFunctionBase._sqrt_taylor_1d(order) # We can reuse the real series for sqrt(1+w)

        # (4) Compose g(f_rescaled - 1) - using rescaled CMTF
        composed_cmtf = sqrt_series_1d_mtf.compose_one_dim(rescaled_cmtf - ComplexMultivariateTaylorFunction.from_constant(1.0, rescaled_cmtf.dimension))

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
        elif isinstance(other, MultivariateTaylorFunctionBase):
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
        elif isinstance(other, MultivariateTaylorFunctionBase):
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
        inverse_series_1d_mtf = MultivariateTaylorFunctionBase._inverse_taylor_1d(order)

        # (4) Compose g(f_rescaled - 1) - using rescaled CMTF
        composed_cmtf = inverse_series_1d_mtf.compose_one_dim(rescaled_cmtf - ComplexMultivariateTaylorFunction.from_constant(1.0, rescaled_cmtf.dimension))

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


# def convert_to_mtf(input_val, dimension=None):
#     """
#     Converts various input types to MultivariateTaylorFunction or ComplexMultivariateTaylorFunction.

#     Args:
#         input_val: Input value to convert (scalar, Var (now function), MTF, CMTF, np.number, 0D np.ndarray).
#         dimension (int, optional): Dimension to use if creating a new MTF/CMTF from scalar.
#                                      Defaults to 1 if input_val is a scalar (int, float, complex) and dimension is not provided.
#                                      Must be explicitly provided for other scalar types if no MTF/CMTF context is available.

#     Returns:
#         MultivariateTaylorFunction or ComplexMultivariateTaylorFunction: The converted MTF or CMTF.

#     Raises:
#         TypeError: if input_val type is not supported.
#         ValueError: if dimension is not a positive integer.
#         RuntimeError: if globals are not initialized when needed.
#     """
#     if isinstance(input_val, (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction)):
#         return input_val
#     elif isinstance(input_val, (int, float)):
#         if dimension is None:
#             dimension = 1  # Default to dimension 1 for int/float if not provided
#         return MultivariateTaylorFunction.from_constant(input_val, dimension)
#     elif isinstance(input_val, complex):
#         if dimension is None:
#             dimension = 1  # Default to dimension 1 for complex if not provided
#         return ComplexMultivariateTaylorFunction.from_constant(input_val, dimension)
#     elif isinstance(input_val, np.ndarray) and input_val.shape == (): # 0D numpy array (numpy scalar)
#         return convert_to_mtf(input_val.item(), dimension) # Recursively convert numpy scalar to python scalar then to MTF
#     elif isinstance(input_val, np.number): #numpy numbers like np.int64, np.float64, np.complex128
#         return convert_to_mtf(float(input_val), dimension) # Convert numpy number to python scalar then to MTF
#     elif callable(input_val) and input_val.__name__ == 'Var': # Check if it's the Var function
#         return input_val(dimension) # call the Var function with dimension (assuming dimension needed)
#     else:
#         raise TypeError(f"Unsupported input type: {type(input_val)}. Cannot convert to MTF/CMTF.")


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
        cmtf_coeffs = {}
        for exponents, coeff_value in variable.coefficients.items():
            cmtf_coeffs[exponents] = coeff_value.astype(np.complex128) # Convert to complex coefficients
        return ComplexMultivariateTaylorFunction(coefficients=cmtf_coeffs, dimension=variable.dimension)
    elif isinstance(variable, (int, float, complex, np.number)):
        # Create constant CMTF from scalar
        return ComplexMultivariateTaylorFunction.from_constant(variable, dimension=1 if isinstance(variable, MultivariateTaylorFunction) else (variable.dimension if isinstance(variable, MultivariateTaylorFunction) else 1) ) # Dimension doesn't strictly matter for constant
    else:
        raise TypeError("Unsupported type for conversion to ComplexMultivariateTaylorFunction.")
    