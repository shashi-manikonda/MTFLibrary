# MTFLibrary/MTFExtended.py
import numpy as np
from collections import defaultdict

from MTFLibrary.taylor_function import (initialize_mtf_globals, get_global_max_order,
    get_global_max_dimension, set_global_max_order, set_global_etol,
    get_global_etol, MultivariateTaylorFunctionBase, convert_to_mtf,
    get_mtf_initialized_status) 
from MTFLibrary.elementary_functions import (cos_taylor, sin_taylor, tan_taylor,
    exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor,
    sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor,
    arctanh_taylor)

class MultivariateTaylorFunction(MultivariateTaylorFunctionBase, np.ndarray):
    """
    Extended MultivariateTaylorFunction class with NumPy ufunc support.
    Inherits from MultivariateTaylorFunction and implements __array_ufunc__.
    """
    def __new__(cls, coefficients, dimension, var_name=None):
        # 1. Create the ndarray instance first.
        #    We need to decide on a default shape and dtype for the ndarray part.
        #    Since MTF coefficients are dictionaries, perhaps an empty array initially is okay?
        #    Shape (0,) and dtype float64 were used before in __init__.
        obj = np.ndarray.__new__(cls, shape=(0,), dtype=np.float64)
        return obj

    def __init__(self, coefficients, dimension, var_name=None):
        # 2. Initialize the MultivariateTaylorFunctionBase part using super().__init__.
        super().__init__(coefficients, dimension, var_name=var_name)
        # 3. (Potentially) Initialize or setup anything else specific to MultivariateTaylorFunction
        #    related to the ndarray part, if needed.  For now, let's see if just __new__ and super().__init__ work.
        self.coefficients = coefficients # Explicitly set coefficients here in __init__
        self.dimension = dimension
        self.var_name = var_name


    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Implements NumPy ufunc protocol, directly calling taylor functions from elementary_functions.py.
        Handles scalar inputs by converting them to MultivariateTaylorFunction constants.
        """
        if method == '__call__':  # Element-wise application of ufunc
            input_mtf = inputs[0] # Assume first input is MTF, for unary ufuncs
    
            if not isinstance(input_mtf, MultivariateTaylorFunctionBase):
                input_mtf = convert_to_mtf(input_mtf, dimension=self.dimension)
    
            if ufunc is np.sin:
                return sin_taylor(input_mtf)
            elif ufunc is np.cos:
                return cos_taylor(input_mtf)
            elif ufunc is np.tan:
                return tan_taylor(input_mtf)
            elif ufunc is np.exp:
                return exp_taylor(input_mtf)
            elif ufunc is np.sqrt:
                return sqrt_taylor(input_mtf)
            elif ufunc is np.log:
                return log_taylor(input_mtf)
            elif ufunc is np.arctan:
                return arctan_taylor(input_mtf)
            elif ufunc is np.sinh:
                return sinh_taylor(input_mtf)
            elif ufunc is np.cosh:
                return cosh_taylor(input_mtf)
            elif ufunc is np.tanh:
                return tanh_taylor(input_mtf)
            elif ufunc is np.arcsin:
                return arcsin_taylor(input_mtf)
            elif ufunc is np.arccos:
                return arccos_taylor(input_mtf)
            elif ufunc is np.arctanh:
                return arctanh_taylor(input_mtf)
            elif ufunc is np.reciprocal:
                return self._inv_mtf_internal(input_mtf)
            elif ufunc is np.add:
                return self + input_mtf
            elif ufunc is np.subtract:
                return self - input_mtf
            elif ufunc is np.multiply:
                return self * input_mtf
            elif ufunc is np.divide or ufunc is np.true_divide:
                return self / input_mtf
            elif ufunc is np.negative:
                return -self
            elif ufunc is np.positive:
                return +self
            elif ufunc is np.power or ufunc is np.float_power:
                if isinstance(inputs[1], (int, float, np.number)):
                    return self ** inputs[1]
            elif ufunc is np.square: # <----- ADDED np.square
                return input_mtf * input_mtf
    
            return NotImplemented  # Let NumPy handle other ufuncs or methods
    
        return NotImplemented # For other methods like reduce, accumulate, etc. which are not handled.
    
def Var(var_index):
    """
    Represents an independent variable as an MultivariateTaylorFunction object.

    Args:
        var_index (int): Unique positive integer identifier for the variable.

    Returns:
        MultivariateTaylorFunction: A MultivariateTaylorFunction object representing the variable x_var_index.
    """
    dimension = get_global_max_dimension()

    if not get_mtf_initialized_status():
        raise RuntimeError("MTF Globals must be initialized before creating Var objects.")
    if not isinstance(var_index, int) or var_index <= 0 or var_index > dimension:
        raise ValueError(f"var_index must be a positive integer between 1 and {dimension}, inclusive.")

    coefficients = {}
    exponent = [0] * dimension
    exponent[var_index - 1] = 1
    coefficients[tuple(exponent)] = np.array([1.0]).reshape(1)

    return MultivariateTaylorFunction(coefficients=coefficients, dimension=dimension)


def compose(mtf_instance: MultivariateTaylorFunctionBase, other_function_dict: dict[int, MultivariateTaylorFunctionBase]) -> MultivariateTaylorFunctionBase:
    """
    Composes a Taylor function with other Taylor functions.

    Substitutes variables in the given MultivariateTaylorFunctionBase object
    with other MultivariateTaylorFunctionBase objects.

    Args:
        mtf_instance (MultivariateTaylorFunctionBase): The MultivariateTaylorFunctionBase to compose.
        other_function_dict (Dict[int, MultivariateTaylorFunctionBase]): A dictionary where keys are
            1-based integer indices of the variables to substitute, and values are
            MultivariateTaylorFunctionBase objects (substituting functions).

    Returns:
        MultivariateTaylorFunctionBase: A new MTF representing the composed function.

    Raises:
        TypeError: if mtf_instance is not a MultivariateTaylorFunctionBase,
                   or if other_function_dict is not a dictionary,
                   or if keys of other_function_dict are not integers,
                   or if values of other_function_dict are not MultivariateTaylorFunctionBase.
    """
    if not isinstance(mtf_instance, MultivariateTaylorFunctionBase):
        raise TypeError("mtf_instance must be a MultivariateTaylorFunctionBase object.")
    if not isinstance(other_function_dict, dict):
        raise TypeError("other_function_dict must be a dictionary.")

    for var_index, substitution_function in other_function_dict.items():
        if not isinstance(var_index, int):
            raise TypeError("Keys of other_function_dict must be integers (variable indices).")
        if not isinstance(substitution_function, MultivariateTaylorFunctionBase):
            raise TypeError("Values of other_function_dict must be MultivariateTaylorFunctionBase objects.")
        if not (1 <= var_index <= mtf_instance.dimension):
            raise ValueError(f"Variable index {var_index} is out of bounds for dimension {mtf_instance.dimension}.")

    composed_coefficients = defaultdict(lambda: np.array([0.0]).reshape(1))

    if not mtf_instance.coefficients:
        return MultivariateTaylorFunctionBase({}, mtf_instance.dimension)

    for original_multi_index, original_coefficient in mtf_instance.coefficients.items():
        term_result = MultivariateTaylorFunctionBase.from_constant(original_coefficient, mtf_instance.dimension)

        for i in range(mtf_instance.dimension):
            order = original_multi_index[i]
            if order > 0:
                var_to_substitute = i + 1
                if var_to_substitute in other_function_dict:
                    substitution_function = other_function_dict[var_to_substitute]
                    term_result = term_result * (substitution_function ** order)
                else:
                    # If no substitution, treat the variable as itself
                    variable_function = MultivariateTaylorFunctionBase.from_variable(var_to_substitute, mtf_instance.dimension)
                    term_result = term_result * (variable_function ** order)

        for exp, coeff in term_result.coefficients.items():
            composed_coefficients[exp] += coeff

    return MultivariateTaylorFunctionBase(composed_coefficients, mtf_instance.dimension)
