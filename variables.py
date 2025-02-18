# variables.py
from taylor_function import MultivariateTaylorFunction
import numpy as np

class Var:
    _var_id_counter = 0  # Counter to assign unique variable IDs

    def __init__(self, var_id=None, dimension=1):
        """
        Initialize a variable for Multivariate Taylor Function.

        Parameters:
            var_id (int, optional): Unique identifier for the variable.
                                     If None, an auto-generated ID will be assigned.
            dimension (int): Dimension of the space where the variable is defined.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if var_id is None:
            Var._var_id_counter += 1
            self.var_id = Var._var_id_counter
        else:
            if not isinstance(var_id, int) or var_id <= 0:
                raise ValueError("Variable ID must be a positive integer.")
            self.var_id = var_id
        self.dimension = dimension

    def _create_taylor_function_from_var(self):
        """
        Create an identity MultivariateTaylorFunction for this variable.
        """
        coefficients = {}
        index = [0] * self.dimension
        index[self.var_id-1] = 1
        coefficients[tuple(index)] = np.array([1.0])
        return MultivariateTaylorFunction(coefficients=coefficients, dimension=self.dimension)

    def __add__(self, other):
        """Override addition to return MTF."""
        mtf_var = self._create_taylor_function_from_var()
        if isinstance(other, (MultivariateTaylorFunction, Var)):
            mtf_other = other if isinstance(other, MultivariateTaylorFunction) else other._create_taylor_function_from_var()
            return mtf_var + mtf_other
        elif isinstance(other, (int, float)):
            return mtf_var + other
        else:
            return NotImplemented

    def __radd__(self, other):
        """Override reverse addition."""
        return self.__add__(other)

    def __sub__(self, other):
        """Override subtraction to return MTF."""
        mtf_var = self._create_taylor_function_from_var()
        if isinstance(other, (MultivariateTaylorFunction, Var)):
            mtf_other = other if isinstance(other, MultivariateTaylorFunction) else other._create_taylor_function_from_var()
            return mtf_var - mtf_other
        elif isinstance(other, (int, float)):
            return mtf_var - other
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Override reverse subtraction."""
        mtf_var = self._create_taylor_function_from_var()
        return other - mtf_var

    def __mul__(self, other):
        """Override multiplication to return MTF."""
        mtf_var = self._create_taylor_function_from_var()
        if isinstance(other, (MultivariateTaylorFunction, Var)):
            mtf_other = other if isinstance(other, MultivariateTaylorFunction) else other._create_taylor_function_from_var()
            return mtf_var * mtf_other
        elif isinstance(other, (int, float)):
            return mtf_var * other
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Override reverse multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Override division to return MTF."""
        mtf_var = self._create_taylor_function_from_var()
        if isinstance(other, (int, float)):
            return mtf_var / other
        elif isinstance(other, MultivariateTaylorFunction):
            return mtf_var / other
        else:
            return NotImplemented


    def __pow__(self, exponent):
        """Override power to return MTF."""
        mtf_var = self._create_taylor_function_from_var()
        if isinstance(exponent, int):
            return mtf_var ** exponent
        else:
            raise ValueError("Exponent must be an integer for Var power operation.")

    def __str__(self):
        return f"var_{self.var_id}"

    def __repr__(self):
        return f"Var({self.var_id}, dim={self.dimension})"