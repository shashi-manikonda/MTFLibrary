# taylor_function.py
"""
This module defines the core class MultivariateTaylorFunction (MTF) for representing and
manipulating multivariate Taylor expansions.

It also includes functions for managing the global maximum order of Taylor expansions
and for converting various input types to MTF objects.

Classes:
    MultivariateTaylorFunction: Represents a multivariate Taylor function and supports
                                arithmetic operations, differentiation, integration, composition,
                                inversion, and evaluation.

Functions:
    set_global_max_order(order): Sets the global maximum order for Taylor expansions.
    get_global_max_order(): Gets the current global maximum order.
    convert_to_mtf(input_value): Converts a scalar, Var, or MTF to a MultivariateTaylorFunction.

Global Variables:
    _GLOBAL_MAX_ORDER:  Private variable storing the global maximum order for Taylor series truncation.
"""
import numpy as np

# Global variable to store the maximum order for Taylor expansions.
# This is used to truncate results of operations to control the size of expansions.
_GLOBAL_MAX_ORDER = 10  # Default global max order


def set_global_max_order(order):
    """
    Sets the global maximum order for Taylor expansions.

    This function allows users to change the global setting for the maximum order of
    Taylor expansions.  Operations on MultivariateTaylorFunction objects, such as
    multiplication and composition, will truncate the resulting Taylor series to this order
    to manage the complexity and size of the expansions.

    Parameters:
        order (int): The desired maximum order for Taylor expansions.
                     Must be a non-negative integer.
    Raises:
        ValueError: If the input order is not a non-negative integer.
    """
    global _GLOBAL_MAX_ORDER  # Declare that we are using the global variable
    if not isinstance(order, int) or order < 0:
        raise ValueError("Global max order must be a non-negative integer.")
    _GLOBAL_MAX_ORDER = order


def get_global_max_order():
    """
    Gets the current global maximum order for Taylor expansions.

    Returns:
        int: The current global maximum order.
    """
    return _GLOBAL_MAX_ORDER


def convert_to_mtf(input_value):
    """
    Converts a scalar, Var, or MultivariateTaylorFunction to a MultivariateTaylorFunction.

    This function ensures that input values are consistently represented as
    MultivariateTaylorFunction objects. If the input is already an MTF, it's returned as is.
    If it's a Var object, an MTF representing that variable is created. If it's a scalar,
    an MTF representing a constant function is created.

    Parameters:
        input_value (scalar | Var | MultivariateTaylorFunction): The value to convert.
            Can be a scalar (int or float), a Var object (representing a variable),
            or an already existing MultivariateTaylorFunction object.

    Returns:
        MultivariateTaylorFunction: The MultivariateTaylorFunction representation
            of the input value.

    Raises:
        TypeError: If the input value is not of a supported type (scalar, Var, or MTF).
    """
    if isinstance(input_value, MultivariateTaylorFunction):
        return input_value  # Already an MTF, return as is
    elif hasattr(input_value, 'is_Var') and input_value.is_Var: # Check if it's a Var object using duck typing
        # Create MTF for Var object. Assume Var objects have 'dimension' attribute.
        coefficients = {
            # For a variable, the coefficient for the first order term in the direction of the variable is 1.0
            tuple([0] * (input_value.dimension - 1) + [1] if input_value.dimension > 0 else [1]): np.array([1.0])
        }
        return MultivariateTaylorFunction(
            coefficients=coefficients,
            dimension=input_value.dimension,
            expansion_point=np.zeros(input_value.dimension),
            order=1  # Taylor expansion of a single variable 'x' up to order 1 is just 'x' itself.
        )
    elif isinstance(input_value, (int, float)):
        # Create MTF for scalar constant. Default dimension is 1, can be adjusted if needed based on context.
        dimension = 1
        coefficients = {tuple([0] * dimension): np.array([float(input_value)])} # Constant term coefficient
        return MultivariateTaylorFunction(
            coefficients=coefficients,
            dimension=dimension,
            expansion_point=np.zeros(dimension),
            order=0  # Taylor expansion of a constant is just the constant itself (order 0).
        )
    else:
        raise TypeError(f"Input value of type {type(input_value)} cannot be converted to MultivariateTaylorFunction.")


class MultivariateTaylorFunction:
    """
    Represents a multivariate Taylor function.

    This class allows for the creation and manipulation of Taylor series expansions
    for functions of multiple variables. It supports arithmetic operations (+, -, *, /, **, -),
    differentiation, integration, composition, and evaluation. Taylor series are represented
    by their coefficients, expansion point, dimension of the input space, and truncation order.

    Attributes:
        coefficients (dict): A dictionary storing the coefficients of the Taylor series.
            Keys are multi-indices (tuples of non-negative integers) representing the order
            of differentiation with respect to each variable. Values are numpy arrays of coefficients.
        dimension (int): The dimension of the input space of the function. Must be a positive integer.
        expansion_point (np.array): The point in the input space around which the Taylor expansion is centered.
            Defaults to a zero vector of the given dimension if not provided.
        order (int, optional): The maximum order to which the Taylor expansion is considered accurate.
            Used for truncation after operations to manage the size of the expansion. If None, no truncation
            is automatically performed, but truncation can be explicitly called.
    """

    def __init__(self, coefficients=None, dimension=1, expansion_point=None, order=None):
        """
        Initialize a MultivariateTaylorFunction.

        Parameters:
            coefficients (dict, optional): Dictionary of coefficients for the Taylor expansion.
                Keys are multi-indices (tuples) and values are numpy arrays of coefficients.
                If None, initializes with an empty coefficient dictionary.
            dimension (int): The dimension of the function's input space. Must be a positive integer.
            expansion_point (np.array, optional): The point around which the Taylor expansion is defined.
                Defaults to a zero vector of size 'dimension' if None.
            order (int, optional): Maximum order of Taylor expansion to keep.
                If None, no truncation is initially set. Truncation is applied during operations.

        Raises:
            ValueError: If dimension is not a positive integer.
        """
        if coefficients is None:
            self.coefficients = {}  # Initialize with empty coefficients if none provided
        else:
            self.coefficients = coefficients.copy()  # Use a copy to avoid unintended modifications of input dict
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        self.dimension = dimension
        self.expansion_point = expansion_point if expansion_point is not None else np.zeros(dimension) # Default expansion point at origin
        self.order = order  # Store the order of the Taylor expansion, used for truncation

    @classmethod
    def from_constant(cls, constant_value, dimension):
        """
        Class method to create a MultivariateTaylorFunction representing a constant value.

        This is a factory method that constructs an MTF for a function that is constant
        across its input space. The Taylor expansion of a constant function is simply the constant itself.

        Parameters:
            constant_value (float): The constant value that the MTF should represent.
            dimension (int): The dimension of the function's input space.

        Returns:
            MultivariateTaylorFunction: A MTF object representing the constant function.
        """
        # For a constant function, only the zero-order term (constant term) has a non-zero coefficient.
        coefficients = {(0,) * dimension: np.array([float(constant_value)])}  # Coefficient for the constant term
        return cls(coefficients=coefficients, dimension=dimension)

    @classmethod
    def identity_function(cls, dimension, var_ids):
        """
        Class method to create a MultivariateTaylorFunction representing identity functions for specified variables.

        Creates an MTF that represents the identity function f(x) = x_i for each variable x_i
        specified in var_ids. This is useful for variable substitution in composition operations.

        Parameters:
            dimension (int): The dimension of the function's input space.
            var_ids (list of int): A list of variable IDs (1-indexed) for which to create identity functions.
                For each var_id in this list, an identity function MTF will be created where the output
                is equal to the input variable corresponding to var_id.

        Returns:
            MultivariateTaylorFunction: A MTF representing the identity function(s). For each variable ID
                                        in var_ids, the MTF will have a first-order term with coefficient 1
                                        corresponding to that variable, and zero for all other terms.
        """
        coefficients = {}
        for var_id in var_ids:
            index = [0] * dimension
            index[var_id - 1] = 1  # Set the exponent for the specified variable to 1 (1-indexed to 0-indexed)
            coefficients[tuple(index)] = np.array([1.0]) # Coefficient of 1 for the first-order term of the identity variable
        return cls(coefficients=coefficients, dimension=dimension)

    def evaluate(self, point):
        """
        Evaluate the MultivariateTaylorFunction at a given point in the input space.

        This method computes the numerical value of the Taylor expansion at a specified point
        by summing up the contributions of all terms in the expansion.

        Parameters:
            point (list or np.array): The point at which to evaluate the MTF.
                Must be a list or numpy array of the same dimension as the MTF.

        Returns:
            float: The evaluated value of the MultivariateTaylorFunction at the given point.

        Raises:
            ValueError: If the dimension of the input point does not match the dimension of the MTF.
        """
        if len(point) != self.dimension:
            raise ValueError(f"Input point dimension {len(point)} does not match MTF dimension {self.dimension}.")
        point_np = np.array(point) - self.expansion_point # Shift point by expansion point to evaluate around expansion_point correctly.
        value = np.array([0.0])  # Initialize the evaluated value as a numpy array to handle array operations
        for index, coeff in self.coefficients.items():
            term_value = coeff # Start with the coefficient value for the current term
            for i, exp in enumerate(index):
                term_value = term_value * (point_np[i] ** exp) # Multiply by (x_i - expansion_point_i)^exp for each variable
            value = value + term_value # Accumulate the value of each term
        return float(value[0]) # Convert the final numpy array value to a standard Python float


    def truncate(self, order=None):
        """
        Truncate the Taylor expansion to a specified order.

        This method creates a new MultivariateTaylorFunction object that contains only the terms
        of the Taylor expansion up to the given 'order'. Terms with a total order (sum of indices)
        greater than 'order' are discarded. If no order is specified, the global maximum order
        set by `set_global_max_order` is used for truncation.

        Parameters:
            order (int, optional): The maximum order to truncate the Taylor expansion to.
                If None, the global maximum order (`get_global_max_order()`) will be used.

        Returns:
            MultivariateTaylorFunction: A new MultivariateTaylorFunction object that is a truncated version
                of the original MTF, containing only terms up to the specified order.
        """
        if order is None:
            order = get_global_max_order()  # Use global max order if order is not provided

        truncated_coeffs = {} # Dictionary to hold coefficients of the truncated MTF
        for index, coeff in self.coefficients.items():
            if sum(index) <= order: # Check if the total order of the term is within the truncation limit
                truncated_coeffs[index] = coeff # Include coefficient if term order is within limit

        # Determine the effective order of the truncated MTF.
        # It's the minimum of the requested 'order' and the original MTF's order (if set).
        calculated_order = min(order, self.order) if self.order is not None else order

        return MultivariateTaylorFunction(
            coefficients=truncated_coeffs,
            dimension=self.dimension,
            expansion_point=self.expansion_point,
            order=calculated_order # Set the order of the truncated MTF
        )

    def __add__(self, other):
        """
        Override the addition operator (+) for MultivariateTaylorFunction objects and scalar addition.

        Supports addition between two MTF objects and addition of a scalar (int or float) to an MTF.
        For MTF + MTF, it adds corresponding coefficients of terms with the same multi-indices.
        For MTF + scalar, it adds the scalar to the constant term (zero-order term) of the MTF.

        Parameters:
            other (MultivariateTaylorFunction | int | float): The object to add to this MTF.

        Returns:
            MultivariateTaylorFunction: A new MTF object representing the sum of the two operands.

        Raises:
            ValueError: If adding two MTFs of different dimensions.
            TypeError: If 'other' is not a MultivariateTaylorFunction, int, or float.
        """
        if isinstance(other, MultivariateTaylorFunction):
            if self.dimension != other.dimension:
                raise ValueError("Dimensions of MTFs must match for addition.")
            new_coefficients = self.coefficients.copy() # Start with coefficients of the first MTF
            for index, coeff in other.coefficients.items():
                # Add coefficients for common indices, initialize to 0 if index not in self.coefficients
                new_coefficients[index] = new_coefficients.get(index, np.array([0.0])) + coeff
            # Truncate the result to the maximum of the orders of the operands, or global max order if order is None.
            result_order = max(self.order, other.order) if self.order is not None and other.order is not None else get_global_max_order()
            return MultivariateTaylorFunction(
                coefficients=new_coefficients,
                expansion_point=self.expansion_point, # Expansion point is unchanged by addition
                dimension=self.dimension,
                order=result_order
            ).truncate()
        elif isinstance(other, (int, float)):
            new_coefficients = self.coefficients.copy()
            constant_index = (0,) * self.dimension # Index for the constant term
            # Add scalar to the constant term, initialize to 0 if constant term not present.
            new_coefficients[constant_index] = new_coefficients.get(constant_index, np.array([0.0])) + np.array([float(other)])
            return MultivariateTaylorFunction(
                coefficients=new_coefficients,
                expansion_point=self.expansion_point,
                dimension=self.dimension,
                order=self.order # Order remains the same as only constant term is affected by scalar addition
            ).truncate()
        else:
            return NotImplemented # Indicate that addition with this type is not supported

    def __radd__(self, other):
        """
        Override reverse addition to handle cases like scalar + MTF.
        For addition, a + b = b + a, so reverse addition is the same as regular addition.
        """
        return self.__add__(other) # Re-use the __add__ method as addition is commutative

    def __sub__(self, other):
        """
        Override the subtraction operator (-) for MTF subtraction and scalar subtraction.

        Supports subtraction between two MTF objects and subtraction of a scalar (int or float) from an MTF.
        For MTF - MTF, it subtracts corresponding coefficients. For MTF - scalar, it subtracts the scalar
        from the constant term of the MTF.

        Parameters:
            other (MultivariateTaylorFunction | int | float): The object to subtract from this MTF.

        Returns:
            MultivariateTaylorFunction: A new MTF object representing the difference.

        Raises:
            ValueError: If subtracting two MTFs of different dimensions.
            TypeError: If 'other' is not a MultivariateTaylorFunction, int, or float.
        """
        if isinstance(other, MultivariateTaylorFunction):
            if self.dimension != other.dimension:
                raise ValueError("Dimensions of MTFs must match for subtraction.")
            new_coefficients = self.coefficients.copy()
            for index, coeff in other.coefficients.items():
                # Subtract coefficients of 'other' from 'self' for common indices
                new_coefficients[index] = new_coefficients.get(index, np.array([0.0])) - coeff
            # Truncate result to the maximum of the orders, or global max order if order is None.
            result_order = max(self.order, other.order) if self.order is not None and other.order is not None else get_global_max_order()
            return MultivariateTaylorFunction(
                coefficients=new_coefficients,
                expansion_point=self.expansion_point,
                dimension=self.dimension,
                order=result_order
            ).truncate()
        elif isinstance(other, (int, float)):
            new_coefficients = self.coefficients.copy()
            constant_index = (0,) * self.dimension
            # Subtract scalar from the constant term
            new_coefficients[constant_index] = new_coefficients.get(constant_index, np.array([0.0])) - np.array([float(other)])
            return MultivariateTaylorFunction(
                coefficients=new_coefficients,
                expansion_point=self.expansion_point,
                dimension=self.dimension,
                order=self.order # Order remains the same as only constant term is affected by scalar subtraction
            ).truncate()
        else:
            return NotImplemented # Indicate subtraction is not implemented for this type

    def __rsub__(self, other):
        """
        Override reverse subtraction for scalar - MTF.
        Computes scalar - MTF by negating the MTF and adding the scalar.

        Parameters:
            other (int | float): The scalar value to subtract the MTF from.

        Returns:
            MultivariateTaylorFunction: A new MTF representing the result of scalar - MTF.

        Raises:
            TypeError: If 'other' is not an int or float.
        """
        if isinstance(other, (int, float)):
            neg_self = -self # Negate the current MTF using the __neg__ operator
            return other + neg_self # Use addition (__add__) with the negated MTF and scalar
        else:
            return NotImplemented # Reverse subtraction not implemented for this type

    def _multiply_mtf(self, other):
        """
        Internal method to handle multiplication between two MTF objects (MTF * MTF).

        This method calculates the coefficients of the product of two MTFs. The multiplication
        is done term-by-term, and the indices of the resulting terms are the sum of the indices
        of the terms being multiplied.

        Parameters:
            other (MultivariateTaylorFunction): The MTF to multiply with.

        Returns:
            dict: A dictionary of coefficients for the product MTF.
        """
        new_coefficients = {} # Initialize dictionary for coefficients of the product
        for index1, coeff1 in self.coefficients.items(): # Iterate through terms of the first MTF
            for index2, coeff2 in other.coefficients.items(): # Iterate through terms of the second MTF
                result_index = tuple(i + j for i, j in zip(index1, index2)) # Sum of indices to get index of product term
                # Multiply coefficients and accumulate in new_coefficients. Initialize to 0 if index not already present.
                new_coefficients[result_index] = new_coefficients.get(result_index, np.array([0.0])) + coeff1 * coeff2
        return new_coefficients

    def _multiply_scalar(self, scalar):
        """
        Internal method to handle multiplication of an MTF by a scalar (MTF * scalar).

        This method scales each coefficient of the MTF by the given scalar.

        Parameters:
            scalar (int | float): The scalar value to multiply with.

        Returns:
            dict: A dictionary of coefficients for the scaled MTF.
        """
        new_coefficients = {} # Initialize dictionary for scaled coefficients
        for index, coeff in self.coefficients.items():
            new_coefficients[index] = coeff * np.array([float(scalar)]) # Scale each coefficient by the scalar
        return new_coefficients

    def __mul__(self, other):
        """
        Override the multiplication operator (*) for MTF multiplication and scalar multiplication.

        Supports multiplication between two MTF objects (MTF * MTF) and multiplication of an MTF by a scalar (MTF * scalar).
        For MTF * MTF, it performs polynomial multiplication and truncates the result.
        For MTF * scalar, it scales all coefficients of the MTF by the scalar.

        Parameters:
            other (MultivariateTaylorFunction | int | float): The object to multiply with this MTF.

        Returns:
            MultivariateTaylorFunction: A new MTF object representing the product.

        Raises:
            ValueError: If multiplying two MTFs of different dimensions.
            TypeError: If 'other' is not a MultivariateTaylorFunction, int, or float.
        """
        if isinstance(other, MultivariateTaylorFunction):
            if self.dimension != other.dimension:
                raise ValueError("Dimensions of MTFs must match for multiplication.")
            new_coefficients = self._multiply_mtf(other) # Use internal method for MTF * MTF multiplication
            # Truncate the result to the minimum of the orders, or global max order if order is None.
            result_order = min(self.order, other.order) if self.order is not None and other.order is not None else get_global_max_order()
            return MultivariateTaylorFunction(
                coefficients=new_coefficients,
                expansion_point=self.expansion_point,
                dimension=self.dimension,
                order=result_order
            ).truncate() # Truncate after multiplication
        elif isinstance(other, (int, float)):
            return MultivariateTaylorFunction(
                coefficients=self._multiply_scalar(other), # Use internal method for MTF * scalar multiplication
                expansion_point=self.expansion_point,
                dimension=self.dimension,
                order=self.order # Order remains the same as only coefficients are scaled by scalar multiplication
            ).truncate()
        else:
            return NotImplemented # Indicate multiplication is not implemented for this type

    def __rmul__(self, other):
        """
        Override reverse multiplication to handle cases like scalar * MTF.
        Multiplication is commutative, so reverse multiplication is the same as regular multiplication.
        """
        return self.__mul__(other) # Re-use the __mul__ method as multiplication is commutative

    def __truediv__(self, other):
        """
        Override division operator (/) for MTF / scalar. MTF / MTF division is not implemented.

        Supports division of an MTF by a scalar (int or float). Division by another MTF is not directly
        implemented and will raise a NotImplementedError. To divide by an MTF, use the `inverse()` method
        to find the inverse MTF and then multiply.

        Parameters:
            other (int | float): The scalar value to divide the MTF by.

        Returns:
            MultivariateTaylorFunction: A new MTF object representing the quotient (MTF / scalar).

        Raises:
            ZeroDivisionError: If attempting to divide by zero.
            NotImplementedError: If attempting to divide by another MultivariateTaylorFunction.
            TypeError: If 'other' is not an int, float, or MultivariateTaylorFunction (for MTF/MTF case).
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide MTF by zero.")
            scalar_inverse = 1.0 / other # Calculate the inverse of the scalar
            return self * scalar_inverse # Use multiplication with the scalar inverse to perform division
        elif isinstance(other, MultivariateTaylorFunction):
            raise NotImplementedError("Division of MTF by another MTF is not implemented. Use inverse() and multiplication instead.")
        else:
            return NotImplemented # Indicate division is not implemented for this type

    def __rtruediv__(self, other):
        """
        Override reverse division for scalar / MTF. Not implemented as it's less common and more complex.

        Raises:
            NotImplementedError: Always, as right division of scalar by MTF is not implemented.
        """
        raise NotImplementedError("Right division of scalar by MTF is not implemented.")

    def __pow__(self, exponent):
        """
        Override power operator (**) for MTF to an integer power.

        Supports raising an MTF to a non-negative integer power. Negative powers are not supported directly;
        for negative powers, first find the inverse using `inverse()` and then raise to the positive power.

        Parameters:
            exponent (int): The integer exponent to raise the MTF to. Must be a non-negative integer.

        Returns:
            MultivariateTaylorFunction: A new MTF object representing MTF raised to the power of 'exponent'.

        Raises:
            ValueError: If the exponent is not an integer or is a negative integer.
        """
        if not isinstance(exponent, int):
            raise ValueError("MTF power exponent must be an integer.")
        if exponent < 0:
            raise ValueError("Negative power exponents are not supported for MTF power. Use inverse() for negative powers.")
        if exponent == 0:
            return MultivariateTaylorFunction.from_constant(1.0, dimension=self.dimension) # MTF to the power of 0 is a constant 1 MTF.
        if exponent == 1:
            return self # MTF to the power of 1 is itself.

        result = self # Start with the base MTF
        for _ in range(exponent - 1): # Multiply 'exponent-1' times to get to the desired power.
            result = result * self # Repeatedly multiply by itself using MTF multiplication
        return result.truncate() # Truncate the final result after exponentiation

    def __neg__(self):
        """
        Override negation operator (-) for MTF negation (-MTF).

        Negates every coefficient in the MTF, effectively reversing the sign of the entire Taylor expansion.

        Returns:
            MultivariateTaylorFunction: A new MTF object representing the negation of the original MTF.
        """
        new_coefficients = {} # Initialize dictionary for negated coefficients
        for index, coeff in self.coefficients.items():
            new_coefficients[index] = -coeff # Negate each coefficient
        return MultivariateTaylorFunction(
            coefficients=new_coefficients,
            expansion_point=self.expansion_point,
            dimension=self.dimension,
            order=self.order # Order remains the same as only coefficients are negated
        ).truncate()

    def derivative(self, wrt_variable_id):
        """
        Compute the partial derivative of the MTF with respect to a specified variable.

        Calculates the derivative of the MultivariateTaylorFunction with respect to the variable
        identified by 'wrt_variable_id'.  The variable ID is 1-indexed, meaning 1 refers to the
        first variable, 2 to the second, and so on, up to the dimension of the MTF.

        Parameters:
            wrt_variable_id (int): The ID of the variable with respect to which to differentiate.
                Must be an integer between 1 and the dimension of the MTF (inclusive).

        Returns:
            MultivariateTaylorFunction: A new MTF object representing the partial derivative.
                The order of the derivative MTF is reduced by 1 compared to the original MTF.

        Raises:
            ValueError: If 'wrt_variable_id' is not a valid variable ID for this MTF's dimension.
        """
        if not isinstance(wrt_variable_id, int) or not 1 <= wrt_variable_id <= self.dimension:
            raise ValueError(f"Invalid variable ID: {wrt_variable_id}. Must be an integer between 1 and {self.dimension}.")

        new_coefficients = {} # Dictionary to store coefficients of the derivative MTF
        for index, coeff in self.coefficients.items():
            exponent = index[wrt_variable_id - 1] # Get exponent of the variable to differentiate with respect to
            if exponent > 0: # Only differentiate terms where the exponent is greater than 0
                deriv_coefficient = coeff * np.array([float(exponent)]) # Derivative coefficient is original coeff * exponent
                deriv_index_list = list(index) # Convert index tuple to list for modification
                deriv_index_list[wrt_variable_id - 1] -= 1 # Decrease exponent of differentiated variable by 1
                deriv_index = tuple(deriv_index_list) # Convert back to tuple
                new_coefficients[deriv_index] = deriv_coefficient # Store coefficient for the derivative term

        return MultivariateTaylorFunction(
            coefficients=new_coefficients,
            dimension=self.dimension,
            expansion_point=self.expansion_point,
            order=self.order # Order of derivative is same as original (though highest order term reduces order by 1, truncate will handle)
        ).truncate() # Truncate the derivative MTF

    def integrate(self, wrt_variable_id, integration_constant=0.0):
        """
        Compute the indefinite integral of the MTF with respect to a specified variable.

        Calculates the indefinite integral of the MultivariateTaylorFunction with respect to the
        variable identified by 'wrt_variable_id'.  Adds a constant of integration, which defaults to 0.

        Parameters:
            wrt_variable_id (int): The ID of the variable with respect to which to integrate.
                Must be an integer between 1 and the dimension of the MTF (inclusive).
            integration_constant (float, optional): The constant of integration to add to the result. Defaults to 0.0.

        Returns:
            MultivariateTaylorFunction: A new MTF object representing the indefinite integral.
                The order of the integrated MTF is increased by 1 compared to the original MTF.

        Raises:
            ValueError: If 'wrt_variable_id' is not a valid variable ID for this MTF's dimension.
        """
        if not isinstance(wrt_variable_id, int) or not 1 <= wrt_variable_id <= self.dimension:
            raise ValueError(f"Invalid variable ID: {wrt_variable_id}. Must be an integer between 1 and {self.dimension}.")

        new_coefficients = {} # Dictionary to store coefficients of the integral MTF
        for index, coeff in self.coefficients.items():
            integ_index_list = list(index) # Convert index tuple to list for modification
            integ_index_list[wrt_variable_id - 1] += 1 # Increase exponent of integrated variable by 1
            integ_index = tuple(integ_index_list) # Convert back to tuple
            exponent = integ_index[wrt_variable_id - 1] # Get new exponent after integration
            integ_coefficient = coeff / np.array([float(exponent)]) # Integral coefficient is original coeff / new exponent
            new_coefficients[integ_index] = integ_coefficient # Store coefficient for the integral term

        constant_index = (0,) * self.dimension # Index for the constant term
        # Add the integration constant as the constant term of the integrated MTF
        new_coefficients[constant_index] = new_coefficients.get(constant_index, np.array([0.0])) + np.array([float(integration_constant)])

        return MultivariateTaylorFunction(
            coefficients=new_coefficients,
            dimension=self.dimension,
            expansion_point=self.expansion_point,
            order=self.order # Order of integral is same as original (though constant term is added, truncate will handle)
        ).truncate() # Truncate the integral MTF


    def compose(self, substitution_dict):
        """
        Compose this MTF with other MTFs or constants.

        Performs function composition by substituting variables in this MTF with other MTFs or constants.
        For each variable x_i in this MTF, a substitution can be provided in the form of another MTF
        or a constant value. The result is a new MTF representing the composition.

        Parameters:
            substitution_dict (dict): A dictionary defining the substitutions.
                Keys should be Var objects (representing variables of this MTF), and values
                should be either MultivariateTaylorFunction objects (MTFs to substitute with) or
                scalar values (int or float constants to substitute with).

        Returns:
            MultivariateTaylorFunction: A new MTF representing the composed function.

        Raises:
            ValueError: If no substitution is provided for a variable that is expected in the composition.
            TypeError: If a substitution value is not an MTF or a scalar.
        """
        composed_coefficients = {(0,) * self.dimension: np.array([0.0])}  # Initialize coefficients for composed MTF with a zero constant term

        def expand_term(term_coeff, term_index):
            """
            Recursive helper function to expand a single term in the polynomial composition.

            For a term of the form c * x_1^e_1 * x_2^e_2 * ... * x_d^e_d, this function substitutes each x_i^e_i
            with the Taylor expansion of the substitution function for x_i raised to the power e_i, and then multiplies
            these substituted expansions together.

            Parameters:
                term_coeff (np.array): The coefficient of the current term.
                term_index (tuple): The multi-index of the current term.

            Returns:
                dict: Coefficients resulting from expanding this term after substitution.
            """
            current_term_result_coeffs = {(0,) * self.dimension: term_coeff} # Start with just the coefficient

            for var_pos, var_exponent in enumerate(term_index): # Iterate through each variable in the term's index
                if var_exponent > 0: # If the exponent for the current variable is positive, substitution is needed
                    var_id = var_pos + 1 # Variable ID is 1-indexed, var_pos is 0-indexed
                    # Find the substitution value for the current variable ID from the substitution dictionary
                    substitution_value = substitution_dict.get(next((v for v in substitution_dict if v.var_id == var_id and v.dimension == self.dimension), None))

                    if substitution_value is None:
                        raise ValueError(f"No substitution provided for variable with ID {var_id} and dimension {self.dimension}.")

                    if isinstance(substitution_value, MultivariateTaylorFunction):
                        # Case 1: Substitution is an MTF
                        term_product_mtf = substitution_value**var_exponent # Raise the substitution MTF to the power of the exponent
                        term_product_coeffs = term_product_mtf.coefficients # Get the coefficients of the powered MTF

                    elif isinstance(substitution_value, (int, float)):
                        # Case 2: Substitution is a constant
                        constant_mtf = MultivariateTaylorFunction.from_constant(substitution_value, dimension=self.dimension) # Convert constant to MTF
                        term_product_mtf = constant_mtf**var_exponent # Raise the constant MTF to the power of the exponent
                        term_product_coeffs = term_product_mtf.coefficients # Get the coefficients (should be a constant MTF)

                    else:
                        raise TypeError("Substitution value must be MTF or scalar.")


                    # Multiply the current term's result coefficients with the coefficients from the substitution
                    next_term_result_coeffs = {} # Initialize coefficients for the next term result
                    for current_index, current_coeff in current_term_result_coeffs.items(): # Iterate through current term coefficients
                        for substitution_index, substitution_coeff in term_product_coeffs.items(): # Iterate through substitution coefficients
                            new_index = tuple(sum(pair) for pair in zip(current_index, substitution_index)) # Add indices
                            next_term_result_coeffs[new_index] = next_term_result_coeffs.get(new_index, np.array([0.0])) + current_coeff * substitution_coeff # Accumulate product of coefficients
                    current_term_result_coeffs = next_term_result_coeffs # Update current term result for next variable in index

            return current_term_result_coeffs # Return coefficients after processing all variables in the term index

        # Iterate through each term in the original MTF
        for index, coeff in self.coefficients.items():
            term_contribution_coeffs = expand_term(coeff, index) # Expand this term by substitution
            # Accumulate the contributions from expanding this term into the overall composed coefficients
            for term_index, term_coeff in term_contribution_coeffs.items():
                composed_coefficients[term_index] = composed_coefficients.get(term_index, np.array([0.0])) + term_coeff


        return MultivariateTaylorFunction(
            coefficients=composed_coefficients,
            dimension=self.dimension,
            expansion_point=self.expansion_point,
            order=self.order # Order remains same, composition within order, truncate at end if needed
        ).truncate() # Truncate the composed MTF to manage order

    def inverse(self, order):
        """
        Calculate the inverse of the MultivariateTaylorFunction using polynomial inversion.

        This method computes the Taylor expansion of 1/MTF up to the specified 'order'.
        It is based on polynomial inversion and is currently implemented only for dimension=1.
        The inverse is calculated around the same expansion point as the original MTF.

        Parameters:
            order (int): The order of the Taylor expansion of the inverse function.

        Returns:
            MultivariateTaylorFunction: The MTF representing the inverse function (1/MTF) up to the given order.

        Raises:
            NotImplementedError: If the dimension of the MTF is not 1 (inversion only for 1D MTFs).
            ValueError: If the constant term of the MTF is zero or very close to zero, as inversion is not possible.
        """
        if self.dimension != 1:
            raise NotImplementedError("Inverse function is only implemented for dimension=1.")
        # Check if the constant term is non-zero, which is necessary for inversion to exist.
        if (0,) * self.dimension not in self.coefficients or np.isclose(self.coefficients.get((0,) * self.dimension), 0):
            raise ValueError("Constant term of MTF must be non-zero to have an inverse.")

        inverse_coefficients = {(0,) * self.dimension: np.array([1.0 / self.coefficients.get((0,) * self.dimension)])} # Initialize inverse constant term
        for order_val in range(1, order + 1): # Iterate for each order from 1 up to the requested order
            current_coeff_index = (order_val,) * self.dimension # Index for the coefficient of current order
            current_coeff_sum = np.array([0.0]) # Initialize sum for calculating current coefficient

            for term_order in range(1, order_val + 1): # Summation for polynomial inversion formula
                index_pair = ((term_order,) * self.dimension, (order_val - term_order,) * self.dimension) # Indices for multiplication in formula
                coeff1 = self.coefficients.get(index_pair[0], np.array([0.0])) # Get coefficient from original MTF
                coeff2 = inverse_coefficients.get(index_pair[1], np.array([0.0])) # Get coefficient from inverse MTF calculated so far
                current_coeff_sum += coeff1.item() * coeff2.item() # Accumulate product

            # Calculate the coefficient for the current order using the inversion formula
            current_coeff = -inverse_coefficients.get((0,) * self.dimension) * current_coeff_sum
            inverse_coefficients[current_coeff_index] = current_coeff # Store the calculated coefficient for the current order

        return MultivariateTaylorFunction(
            coefficients=inverse_coefficients,
            dimension=self.dimension,
            expansion_point=self.expansion_point,
            order=order # Set order of inverse MTF to requested order
        )

    def __str__(self):
        """
        Return a user-friendly string representation of the MTF (tabular format).

        This method is implicitly called when you use `str(mtf_object)` or `print(mtf_object)`.
        It uses `get_tabular_string` to generate a formatted, tabular string representation
        of the Taylor expansion, making it easy to read and understand.

        Returns:
            str: Tabular string representation of the Taylor expansion.
        """
        return self.get_tabular_string() # Delegate to get_tabular_string for tabular output

    def __repr__(self):
        """
        Return a detailed string representation of the MTF for debugging and inspection.

        This method is called when `repr(mtf_object)` is used or when an MTF object is displayed in
        an interactive Python session without explicitly calling `print`. It provides a comprehensive
        string that includes all the defining attributes of the MTF: coefficients, dimension,
        expansion point, and order.

        Returns:
            str: A string representation that includes coefficients, dimension, expansion point, and order.
        """
        coeff_repr = ", ".join([f"{index}: {coeff}" for index, coeff in self.coefficients.items()]) # Format coefficients for representation
        return (f"MultivariateTaylorFunction(coefficients={{{coeff_repr}}}, dimension={self.dimension}, "
                f"expansion_point={list(self.expansion_point)}, order={self.order})") # Construct detailed representation string

    def get_tabular_string(self, order=None, variable_names=None):
        """
        Returns a string representing the Taylor expansion in a tabular format.

        Provides a formatted table that lists each term of the Taylor expansion, including its index,
        coefficient, total order, exponents for each variable, and variable names (optional).
        The table is sorted by the total order of the terms for readability.

        Parameters:
            order (int, optional): Maximum order of terms to include in the table.
                Defaults to the global maximum order if None, showing terms up to the global max order.
            variable_names (list of str, optional): List of names for each variable.
                Used as column headers in the table. If None, default names 'x_1', 'x_2', ... are used.

        Returns:
            str: Tabular string representation of the Taylor expansion.
        """
        truncated_function = self.truncate(order) # Truncate the MTF to the display order for clarity
        if variable_names is None:
            variable_names = [f"x_{i+1}" for i in range(self.dimension)] # Default variable names if none provided

        # Construct header row of the table
        header_row = "| Term Index | Coefficient | Order | Exponents "
        if variable_names:
            header_row += "| " + " | ".join(variable_names) + " |" # Add variable name columns if provided
        else:
            header_row += "|"
        header_row += "\n"

        # Construct separator row for table formatting
        separator_row = "|------------|-------------|-------|-----------"
        if variable_names:
            separator_row += "|" + "-----|" * self.dimension + "-" # Add separators for variable name columns
        separator_row += "\n"

        table_str = header_row + separator_row # Start table string with header and separator
        term_count = 0  # Initialize counter for term index

        # Sort terms by total order for ordered display in table
        sorted_items = sorted(truncated_function.coefficients.items(), key=lambda item: sum(item[0]))

        for multi_index, coefficient in sorted_items: # Iterate through terms in sorted order
            term_count += 1 # Increment term index for each term
            term_order = sum(multi_index) # Calculate total order of the term
            index_str = f"| {term_count:<10} " # Format term index
            coefficient_str = f"| {coefficient[0]:11.8f} " # Format coefficient value
            order_str = f"| {term_order:<5} " # Format term order
            exponent_str = f"| { ' '.join(map(str, multi_index)) :<9} " # Format exponents as string

            row = index_str + coefficient_str + order_str + exponent_str # Start row with index, coefficient, order, exponents

            if variable_names:
                for exponent in multi_index:
                    row += f"| {exponent:5} " # Add column for each variable's exponent with name if variable names provided
                row += "|"
            else:
                row += "|"

            table_str += row + "\n" # Append row to table string

        return table_str # Return the complete tabular string


    def print_tabular(self, order=None, variable_names=None, coeff_format="{:15.3e}"):
        """
        Prints the tabular representation of the Taylor expansion to the console.

        This method is a convenience wrapper around `get_tabular_string`. It generates the tabular
        string representation of the MTF and prints it to the standard output.

        Parameters:
            order (int, optional): Maximum order of terms to include in the printed table.
                Defaults to the global maximum order if None.
            variable_names (list of str, optional): Names of variables for table column headers.
                Defaults to ['x_1', 'x_2', ...] if None.
            coeff_format (str, optional): Format string for displaying coefficients in the table.
                Defaults to "{:15.3e}" (scientific notation with 3 decimal places, width 15).
        """
        print(self.get_tabular_string(order=order, variable_names=variable_names)) # Generate tabular string and print it to console