# MTFLibrary/MTFExtended.py
from .elementary_functions import cos_taylor, sin_taylor, exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor, sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, arctanh_taylor
from .taylor_function import MultivariateTaylorFunction, initialize_mtf_globals, convert_to_mtf # Import base class and core functions
from .taylor_function import get_global_max_order, get_global_max_dimension, set_global_etol
import numpy as np

class MTFExtended(MultivariateTaylorFunction):
    """
    Extended MultivariateTaylorFunction class with NumPy ufunc support.
    Inherits from MultivariateTaylorFunction and implements __array_ufunc__.
    """
def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
    """
    Implements NumPy ufunc protocol, directly calling taylor functions from elementary_functions.py.
    Handles scalar inputs by converting them to MTFExtended constants.
    """
    if method == '__call__':  # Element-wise application of ufunc
        # --- Scalar Input Handling ---
        input_mtf = inputs[0]
        print(f"Inside __array_ufunc__, ufunc: {ufunc}, type(inputs[0]): {type(inputs[0])}, self.dimension: {self.dimension}") # DEBUG PRINT
        if not isinstance(input_mtf, MTFExtended): # Check if input is NOT an MTFExtended (i.e., scalar)
            print("Converting scalar input to MTFConstant...") # DEBUG PRINT
            input_mtf = convert_to_mtf(input_mtf, dimension=self.dimension) # Convert scalar to MTFConstant
        else:
            print("Input is already MTFExtended.") # DEBUG PRINT


        if ufunc is np.sin:
            return sin_taylor(input_mtf)
        elif ufunc is np.cos:
            return cos_taylor(input_mtf)
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
        elif ufunc is np.reciprocal: # NumPy's name for 1/x ufunc
            return self._inv_mtf_internal(input_mtf) # Use input_mtf here (already possibly converted scalar)
        # gaussian_taylor is not directly mapped to a single numpy ufunc, so skipping for now.
        # Add more ufuncs here as needed, directly calling corresponding taylor functions

    return NotImplemented  # Let NumPy handle other ufuncs or methods


# Modify Var function to return MTFExtended instances
def Var(var_index):
    """
    Represents an independent variable as an MTFExtended object.

    Args:
        var_index (int): Unique positive integer identifier for the variable.

    Returns:
        MTFExtended: A MTFExtended object representing the variable x_var_index.
    """
    from MTFLibrary.taylor_function import _INITIALIZED, _GLOBAL_MAX_DIMENSION, get_global_max_dimension # IMPORT GLOBALS HERE
    
    if not _INITIALIZED:
        raise RuntimeError("MTF Globals must be initialized before creating Var objects.")
    if not isinstance(var_index, int) or var_index <= 0 or var_index > _GLOBAL_MAX_DIMENSION:
        raise ValueError(f"var_index must be a positive integer between 1 and {_GLOBAL_MAX_DIMENSION}, inclusive.")

    dimension = get_global_max_dimension() #_GLOBAL_MAX_DIMENSION # Use global dimension
    coefficients = {}
    exponent = [0] * dimension
    exponent[var_index - 1] = 1
    coefficients[tuple(exponent)] = np.array([1.0]).reshape(1)

    return MTFExtended(coefficients=coefficients, dimension=dimension, var_name=f'x_{var_index}') # Return MTFExtended


# # Example Usage (in taylor_function.py):
# if __name__ == '__main__':
#     initialize_mtf_globals(max_order=10, max_dimension=2)
#     x1 = Var(1) # Var now creates MTFExtended instances
#     x2 = Var(2)
#     mtf_array = np.array([[x1 + 1, x2 * 2], [x1 * x2, x2 - 0.5]])
#     scalar_mtf = x1 + 2

#     sin_np_array_result = np.sin(mtf_array)
#     print("Result of np.sin(mtf_array):")
#     print(sin_np_array_result)
#     print("\nEvaluating np.sin(mtf_array[0,0]) at [0.1, 0.2]:")
#     print(sin_np_array_result[0, 0].eval([0.1, 0.2]))

#     cos_np_scalar_result = np.cos(scalar_mtf)
#     print("\nResult of np.cos(scalar_mtf):")
#     print(cos_np_scalar_result)
#     print("\nEvaluating np.cos(scalar_mtf) at [0.1, 0.2]:")
#     print(cos_np_scalar_result.eval([0.1, 0.2]))

#     exp_np_array_result = np.exp(mtf_array)
#     print("\nResult of np.exp(mtf_array):")
#     print(exp_np_array_result)

#     sqrt_np_scalar_result = np.sqrt(scalar_mtf)
#     print("\nResult of np.sqrt(scalar_mtf):")
#     print(sqrt_np_scalar_result)

#     log_np_array_result = np.log(mtf_array)
#     print("\nResult of np.log(mtf_array):")
#     print(log_np_array_result)

#     arctan_np_scalar_result = np.arctan(scalar_mtf)
#     print("\nResult of np.arctan(scalar_mtf):")
#     print(arctan_np_scalar_result)

#     sinh_np_array_result = np.sinh(mtf_array)
#     print("\nResult of np.sinh(mtf_array):")
#     print(sinh_np_array_result)

#     cosh_np_scalar_result = np.cosh(scalar_mtf)
#     print("\nResult of np.cosh(scalar_mtf):")
#     print(cosh_np_scalar_result)

#     tanh_np_array_result = np.tanh(mtf_array)
#     print("\nResult of np.tanh(mtf_array):")
#     print(tanh_np_array_result)

#     arcsin_np_scalar_result = np.arcsin(Var(1)/2) # valid range for arcsin
#     print("\nResult of np.arcsin(Var(1)/2):")
#     print(arcsin_np_scalar_result)

#     arccos_np_scalar_result = np.arccos(Var(1)/2) # valid range for arccos
#     print("\nResult of np.arccos(Var(1)/2):")
#     print(arccos_np_scalar_result)

#     arctanh_np_array_result = np.arctanh(Var(2)/2) # valid range for arctanh
#     print("\nResult of np.arctanh(Var(2)/2):")
#     print(arctanh_np_array_result)