# variables.py
"""
Defines the Var class for representing variables used in Multivariate Taylor Functions (MTFs).

The Var class is designed to create symbolic variables that can be used with
MultivariateTaylorFunction objects. Each Var instance represents a unique variable
in a multivariate function and can be used to construct identity MTFs, perform
arithmetic operations that result in MTFs, and manage variable dimensions and IDs.

Classes:
    Var: Represents a variable for use in Multivariate Taylor Function expansions.
"""
from taylor_function import MultivariateTaylorFunction
import numpy as np

class Var:
    """
    Represents a variable for use in Multivariate Taylor Function expansions.

    Each instance of Var represents a unique variable in a multivariate function.
    Variables can be created with a specified dimension and are assigned a unique ID,
    either automatically or explicitly provided. Var objects are designed to seamlessly
    interact with MultivariateTaylorFunction objects, allowing for the construction of
    Taylor expansions and symbolic computations.

    Attributes:
        var_id (int): A unique identifier for the variable. Assigned automatically if not provided.
        dimension (int): The dimension of the space in which this variable is defined.
                         Must be a positive integer.

    Class Attributes:
        _var_id_counter (int): A class-level counter used to automatically generate unique variable IDs.
                                 Initialized to 0 and incremented each time a Var is created without
                                 a specified var_id.
    """
    _var_id_counter = 0  # Counter to assign unique variable IDs

    def __init__(self, var_id=None, dimension=1):
        """
        Initialize a Var object.

        Parameters:
            var_id (int, optional): Unique identifier for the variable.
                                     If None, a unique ID is automatically generated using the class counter.
                                     Must be a positive integer if provided.
                                     Defaults to None.
            dimension (int): Dimension of the space where the variable is defined.
                             Must be a positive integer. Defaults to 1.

        Raises:
            ValueError: if dimension is not a positive integer, or if var_id is provided but is not a positive integer.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if var_id is None:
            Var._var_id_counter += 1  # Increment the counter to generate a new unique ID
            self.var_id = Var._var_id_counter # Assign the current counter value as the variable ID
        else:
            if not isinstance(var_id, int) or var_id <= 0:
                raise ValueError("Variable ID must be a positive integer.")
            self.var_id = var_id # Use the provided var_id
        self.dimension = dimension # Store the dimension of the variable's space
        self.is_Var = True # Flag to identify Var objects - for duck typing in convert_to_mtf


    def _create_taylor_function_from_var(self):
        """
        Create an identity MultivariateTaylorFunction for this variable.

        This private method generates a MultivariateTaylorFunction that represents the
        identity function for this variable. It's used internally to convert a Var object
        into its MTF representation for operations. For a variable 'x_i' of dimension 'd',
        the identity MTF is simply the Taylor expansion of 'x_i' itself.

        Returns:
            MultivariateTaylorFunction: An MTF representing the identity function of this variable.
                                         The MTF will have a coefficient of 1.0 for the first-order term
                                         corresponding to this variable's index and zero for all other terms.
        """
        coefficients = {}
        index = [0] * self.dimension # Initialize a multi-index with zeros, length of dimension
        index[self.var_id-1] = 1 # Set the exponent for the current variable's dimension to 1 (1-indexed to 0-indexed)
        coefficients[tuple(index)] = np.array([1.0]) # Set coefficient for the first-order term to 1.0
        return MultivariateTaylorFunction(coefficients=coefficients, dimension=self.dimension, expansion_point=np.zeros(self.dimension), order=1) # Create and return the MTF

    def __add__(self, other):
        """
        Override addition (+) operator for Var objects. Returns a MultivariateTaylorFunction.

        This method allows addition of a Var object with another Var, a MultivariateTaylorFunction,
        or a scalar (int or float). It converts the Var object and the 'other' object (if necessary)
        to MultivariateTaylorFunction objects and then performs the addition operation on MTFs,
        returning the resulting MTF.

        Parameters:
            other (MultivariateTaylorFunction | Var | int | float): The object to add to this Var.

        Returns:
            MultivariateTaylorFunction: The MTF result of the addition operation.
                                         Returns NotImplemented if 'other' is not of a supported type.
        """
        mtf_var = self._create_taylor_function_from_var() # Convert self (Var) to MTF
        if isinstance(other, (MultivariateTaylorFunction, Var)):
            mtf_other = other if isinstance(other, MultivariateTaylorFunction) else other._create_taylor_function_from_var() # Convert 'other' to MTF if it's a Var
            return mtf_var + mtf_other # Perform MTF addition
        elif isinstance(other, (int, float)):
            return mtf_var + other # Perform MTF and scalar addition
        else:
            return NotImplemented # Indicate addition is not supported with this type


    def __radd__(self, other):
        """
        Override reverse addition for Var objects. Returns a MultivariateTaylorFunction.

        This method is called for reverse addition operations (e.g., scalar + Var).
        It simply calls the __add__ method as addition is commutative.

        Parameters:
            other (MultivariateTaylorFunction | Var | int | float): The object to add to this Var from the left.

        Returns:
            MultivariateTaylorFunction: The MTF result of the addition operation.
                                         Returns NotImplemented if 'other' is not of a supported type.
        """
        return self.__add__(other) # Re-use the __add__ method for reverse addition

    def __sub__(self, other):
        """
        Override subtraction (-) operator for Var objects. Returns a MultivariateTaylorFunction.

        This method allows subtraction of a Var object with another Var, a MultivariateTaylorFunction,
        or a scalar (int or float). It converts the Var object and the 'other' object (if necessary)
        to MultivariateTaylorFunction objects and then performs the subtraction operation on MTFs,
        returning the resulting MTF.

        Parameters:
            other (MultivariateTaylorFunction | Var | int | float): The object to subtract from this Var.

        Returns:
            MultivariateTaylorFunction: The MTF result of the subtraction operation.
                                         Returns NotImplemented if 'other' is not of a supported type.
        """
        mtf_var = self._create_taylor_function_from_var() # Convert self (Var) to MTF
        if isinstance(other, (MultivariateTaylorFunction, Var)):
            mtf_other = other if isinstance(other, MultivariateTaylorFunction) else other._create_taylor_function_from_var() # Convert 'other' to MTF if it's a Var
            return mtf_var - mtf_other # Perform MTF subtraction
        elif isinstance(other, (int, float)):
            return mtf_var - other # Perform MTF and scalar subtraction
        else:
            return NotImplemented # Indicate subtraction is not supported with this type

    def __rsub__(self, other):
        """
        Override reverse subtraction for Var objects (e.g., scalar - Var). Returns a MultivariateTaylorFunction.

        This method handles operations like scalar - Var. It converts the Var object to its MTF form
        and then performs the reverse subtraction of the MTF from the 'other' object (which is expected to be a scalar).

        Parameters:
            other (int | float): The scalar value from which to subtract this Var.

        Returns:
            MultivariateTaylorFunction: The MTF result of the reverse subtraction operation.
                                         Returns NotImplemented if 'other' is not of a supported type.
        """
        mtf_var = self._create_taylor_function_from_var() # Convert self (Var) to MTF
        return other - mtf_var # Perform scalar and MTF reverse subtraction

    def __mul__(self, other):
        """
        Override multiplication (*) operator for Var objects. Returns a MultivariateTaylorFunction.

        This method allows multiplication of a Var object with another Var, a MultivariateTaylorFunction,
        or a scalar (int or float). It converts the Var object and 'other' object (if needed) to
        MultivariateTaylorFunction objects and then performs the multiplication operation on MTFs,
        returning the resulting MTF.

        Parameters:
            other (MultivariateTaylorFunction | Var | int | float): The object to multiply with this Var.

        Returns:
            MultivariateTaylorFunction: The MTF result of the multiplication operation.
                                         Returns NotImplemented if 'other' is not of a supported type.
        """
        mtf_var = self._create_taylor_function_from_var() # Convert self (Var) to MTF
        if isinstance(other, (MultivariateTaylorFunction, Var)):
            mtf_other = other if isinstance(other, MultivariateTaylorFunction) else other._create_taylor_function_from_var() # Convert 'other' to MTF if it's a Var
            return mtf_var * mtf_other # Perform MTF multiplication
        elif isinstance(other, (int, float)):
            return mtf_var * other # Perform MTF and scalar multiplication
        else:
            return NotImplemented # Indicate multiplication is not supported with this type

    def __rmul__(self, other):
        """
        Override reverse multiplication for Var objects. Returns a MultivariateTaylorFunction.

        This method is called for reverse multiplication operations (e.g., scalar * Var).
        It simply calls the __mul__ method as multiplication is commutative.

        Parameters:
            other (MultivariateTaylorFunction | Var | int | float): The object to multiply with this Var from the left.

        Returns:
            MultivariateTaylorFunction: The MTF result of the multiplication operation.
                                         Returns NotImplemented if 'other' is not of a supported type.
        """
        return self.__mul__(other) # Re-use the __mul__ method for reverse multiplication

    def __truediv__(self, other):
        """
        Override true division (/) operator for Var objects. Returns a MultivariateTaylorFunction.

        This method allows division of a Var object by a scalar (int or float) or a MultivariateTaylorFunction.
        Division by another Var object directly is not explicitly handled here and would likely result in
        division by its MTF form. It converts the Var object to its MTF form and then performs the division.
        Note: MTF division by another MTF is limited (e.g., division by zero constant term MTF is an error,
        and general MTF/MTF division may require inverse and multiplication).

        Parameters:
            other (int | float | MultivariateTaylorFunction): The divisor for this Var.

        Returns:
            MultivariateTaylorFunction: The MTF result of the division operation.
                                         Returns NotImplemented if 'other' is not of a supported type.
        """
        mtf_var = self._create_taylor_function_from_var() # Convert self (Var) to MTF
        if isinstance(other, (int, float)):
            return mtf_var / other # Perform MTF and scalar division
        elif isinstance(other, MultivariateTaylorFunction):
            return mtf_var / other # Perform MTF division (scalar division in MTF truediv)
        else:
            return NotImplemented # Indicate division is not supported with this type


    def __pow__(self, exponent):
        """
        Override power (**) operator for Var objects. Returns a MultivariateTaylorFunction.

        This method allows raising a Var object to an integer power. It converts the Var object to its
        MultivariateTaylorFunction form and then performs the power operation on the MTF.

        Parameters:
            exponent (int): The integer exponent to which the Var is raised. Must be a non-negative integer.

        Returns:
            MultivariateTaylorFunction: The MTF result of raising the Var to the given power.

        Raises:
            ValueError: If the exponent is not an integer.
        """
        mtf_var = self._create_taylor_function_from_var() # Convert self (Var) to MTF
        if isinstance(exponent, int):
            return mtf_var ** exponent # Perform MTF power operation
        else:
            raise ValueError("Exponent must be an integer for Var power operation.") # Raise error for non-integer exponent

    def __str__(self):
        """
        Return a user-friendly string representation of the Var object.

        This method is called when str() is used on a Var object, or when it is printed.
        It provides a simple string representation in the format "var_ID", where ID is the variable's unique ID.

        Returns:
            str: A string representing the Var object, e.g., "var_1", "var_2", etc.
        """
        return f"var_{self.var_id}" # Return formatted string "var_ID"


    def __repr__(self):
        """
        Return a detailed string representation of the Var object for debugging and inspection.

        This method is called when repr() is used on a Var object or when it's displayed in an interactive session.
        It provides a string that includes the variable ID and its dimension, useful for distinguishing
        Var objects and understanding their properties.

        Returns:
            str: A string representation that includes variable ID and dimension, e.g., "Var(1, dim=1)".
        """
        return f"Var({self.var_id}, dim={self.dimension})" # Return formatted string with ID and dimension