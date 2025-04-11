# MTFLibrary/MTFExtended.py
import numpy as np
from collections import defaultdict
import pandas as pd
from functools import reduce

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


    def __reduce__(self):
        ndarray_state = super().__getstate__()
        custom_state = {'coefficients': self.coefficients,
                        'dimension': self.dimension,
                        'var_name': self.var_name}
        return (self.__class__,
                (self.coefficients, self.dimension, self.var_name),
                (ndarray_state, custom_state))

    def __setstate__(self, state):
        ndarray_state, custom_state = state
        if isinstance(ndarray_state, tuple) and len(ndarray_state) >= 4:
            super().__setstate__(ndarray_state)
        elif isinstance(ndarray_state, dict):
            # This might happen in some cases, try updating __dict__
            self.__dict__.update(ndarray_state)
        else:
            raise TypeError(f"Unexpected type for ndarray_state: {type(ndarray_state)}, value: {ndarray_state}")
        self.__dict__.update(custom_state)


'''
Create a Alias for MultivariateTaylorFunction class for ease of calling and use
'''
MTF = MultivariateTaylorFunction


def Var(var_index):
    """
    Represents an independent variable as an MultivariateTaylorFunction object.

    Args:
        var_index (int): Unique positive integer identifier for the variable.

    Returns:
        MultivariateTaylorFunction: A MultivariateTaylorFunction object representing the variable x_var_index.
    
    Note:
    Var(var_id_int) Function: The Var(var_id_int) function is used to initialize 
    a symbolic variable for Taylor expansion. It creates a first-order Taylor 
    polynomial representing a single variable. The var_id_int argument specifies 
    the dimension of the variable, represented as an integer from 1 to 
    max_dimension (inclusive). The order of this integer also corresponds to the 
    order of the variable in the coefficient exponent tuple 
    (e.g., for max_dimension=3, the first element of the exponent tuple is the 
     power of the variable with ID 1, the second element is the power of the 
    variable with ID 2, and so on).
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
        term_result = MultivariateTaylorFunctionBase.from_constant(original_coefficient)

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


def mtfarray(mtfs, column_names=None):
    """
    Merges a list of MultivariateTaylorFunction objects into a single pandas DataFrame
    based on their Taylor series coefficients' order and exponents.

    Args:
        mtfs (list): A list of MultivariateTaylorFunction objects.
        column_names (list, optional): An optional list of strings to rename the coefficient
                                       columns in the output DataFrame. The length of this list
                                       must match the number of MTFs in the `mtfs` argument.
                                       Defaults to None, in which case the coefficient columns
                                       will be named based on the MTF names or indices.

    Returns:
        pandas.DataFrame: A DataFrame where each row represents a unique combination
                          of 'Order' and 'Exponents' found across all input MTFs.
                          The DataFrame will have columns for the coefficients of each
                          input MTF, along with 'Order' and 'Exponents'. Missing
                          coefficients are filled with 0.

    Raises:
        TypeError: If `mtfs` is not a list or if any element in `mtfs` is not a
                   MultivariateTaylorFunction, or if `column_names` is provided but is not a list.
        ValueError: If the MTF objects in the `mtfs` list have different dimensions
                    or different variable lists (if the variable consistency check is enabled),
                    or if `column_names` is provided but its length does not match
                    the number of input MTFs.
    """

    
    if isinstance(mtfs, np.ndarray):
        mtfs = list(mtfs)
    
    if not isinstance(mtfs, list):
        raise TypeError("Input 'mtfs' must be a list.")

    if not mtfs:
        return pd.DataFrame(columns=['Order', 'Exponents'])  # Return an empty DataFrame for an empty list

    valid_mtf_types = (MultivariateTaylorFunction,
                        MultivariateTaylorFunctionBase)
    for mtf in mtfs:
        if not isinstance(mtf, valid_mtf_types):
            raise TypeError(f"All elements in 'mtfs' must be instances of {MultivariateTaylorFunction.__name__}, but found {type(mtf).__name__}.")
            

    first_dim = mtfs[0].dimension
    for i, mtf in enumerate(mtfs[1:]):
        if mtf.dimension != first_dim:
            raise ValueError(f"MTF at index {i+1} has dimension {mtf.dimension}, but the first MTF has dimension {first_dim}. All MTFs must have the same dimension.")

    dfs = []
    for i, mtf in enumerate(mtfs):
        df = mtf.get_tabular_dataframe()
        if column_names and len(column_names) == len(mtfs):
            if 'Coefficient' in df.columns:
                df = df.rename(columns={'Coefficient': f'Coeff_{column_names[i]}'})
        else:
            mtf_name = getattr(mtf, 'name', str(i + 1))
            if 'Coefficient' in df.columns:
                df = df.rename(columns={'Coefficient': f'Coefficient_{mtf_name}'})
        dfs.append(df)

    tmap = reduce(lambda left, right: pd.merge(left, right, on=['Order', 'Exponents'], how='outer'), dfs)

    coef_cols_initial = [col for col in tmap.columns if col.startswith('Coeff')]
    cols = coef_cols_initial +['Order', 'Exponents'] 
    tmap = tmap[cols]
    tmap[coef_cols_initial] = tmap[coef_cols_initial].fillna(0)

    tmap = tmap.sort_values(by=['Order', 'Exponents'], ascending=[True, False]).reset_index(drop=True)
    return tmap