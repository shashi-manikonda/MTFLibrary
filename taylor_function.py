# taylor_function.py
from taylor_operations import get_global_max_order # ADD THIS LINE
import numpy as np


# taylor_operations.py (or taylor_function.py if you put them there)

_GLOBAL_MAX_ORDER = 10 # Default global max order

def set_global_max_order(order):
    """Sets the global maximum order for Taylor expansions."""
    global _GLOBAL_MAX_ORDER # <---- Ensure 'global' keyword is here
    _GLOBAL_MAX_ORDER = order

def get_global_max_order():
    """Gets the current global maximum order."""
    return _GLOBAL_MAX_ORDER


# _GLOBAL_MAX_ORDER = 10 # Default global max order

# def set_global_max_order(order):
#     """
#     Set the global maximum order for Taylor expansions.

#     This affects operations like multiplication, division, power, and composition,
#     where the order of the result might need to be truncated.

#     Parameters:
#         order (int): The desired maximum order. Must be a non-negative integer.
#     """
#     global _GLOBAL_MAX_ORDER
#     if not isinstance(order, int) or order < 0:
#         raise ValueError("Global max order must be a non-negative integer.")
#     _GLOBAL_MAX_ORDER = order

# def get_global_max_order():
#     """
#     Get the current global maximum order for Taylor expansions.

#     Returns:
#         int: The current global maximum order.
#     """
#     return _GLOBAL_MAX_ORDER


class MultivariateTaylorFunction:
    def __init__(self, coefficients=None, dimension=1, expansion_point=None, order=None):
        """
        Initialize a MultivariateTaylorFunction.

        Parameters:
            coefficients (dict): Dictionary of coefficients where keys are multi-indices (tuples)
                                 and values are numpy arrays representing coefficients.
            dimension (int): The dimension of the function's input space.
            expansion_point (np.array): The point around which the Taylor expansion is defined.
            order (int, optional): Maximum order of Taylor expansion to keep. If None, no truncation is performed unless explicitly called.
        """
        if coefficients is None:
            self.coefficients = {}
        else:
            self.coefficients = {} # Create a new dictionary
            self.coefficients.update(coefficients) # Copy items from input to the new dictionary
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        self.dimension = dimension
        self.expansion_point = expansion_point if expansion_point is not None else np.zeros(dimension)
        self.order = order

    @classmethod
    def from_constant(cls, constant_value, dimension):
        """
        Create a MultivariateTaylorFunction representing a constant value.

        Parameters:
            constant_value (float): The constant value.
            dimension (int): The dimension of the function's input space.

        Returns:
            MultivariateTaylorFunction: A MTF representing the constant function.
        """
        coefficients = {(0,) * dimension: np.array([float(constant_value)])} # Now creating 1D array
        return cls(coefficients=coefficients, dimension=dimension)

    @classmethod
    def identity_function(cls, dimension, var_ids):
        """
        Create a MultivariateTaylorFunction representing an identity function for specified variables.

        Parameters:
            dimension (int): The dimension of the function's input space.
            var_ids (list of int): List of variable IDs for which to create identity functions.

        Returns:
            MultivariateTaylorFunction: A MTF representing the identity function(s).
        """
        coefficients = {}
        for var_id in var_ids:
            index = [0] * dimension
            index[var_id-1] = 1 # var_id is 1-indexed, index is 0-indexed
            coefficients[tuple(index)] = np.array([1.0])
        return cls(coefficients=coefficients, dimension=dimension)


    def evaluate(self, point):
        """
        Evaluate the MultivariateTaylorFunction at a given point.

        Parameters:
            point (list or np.array): The point at which to evaluate the MTF.

        Returns:
            float: The evaluated value of the MTF at the given point.
        """
        if len(point) != self.dimension:
            raise ValueError(f"Input point dimension {len(point)} does not match MTF dimension {self.dimension}.")
        point_np = np.array(point) - self.expansion_point
        value = np.array([0.0]) # Initialize value as numpy array
        for index, coeff in self.coefficients.items():
            term_value = coeff
            for i, exp in enumerate(index):
                term_value = term_value * (point_np[i]**exp)
            value = value + term_value # Use numpy array addition
        return float(value[0]) # Convert to float after extracting the scalar from array


    # def truncate(self, order=None):
    #     """
    #     Truncate the Taylor series to a specified order or the global maximum order.
    
    #     Parameters:
    #         order (int, optional): The order to truncate to. If None, global max order is used.
    
    #     Returns:
    #         MultivariateTaylorFunction: A new MTF truncated to the specified order.
    #     """
    #     if order is None:
    #         if self.order is None:
    #             trunc_order = get_global_max_order() # Use global max order if no order is given and MTF has no order
    #         else:
    #             trunc_order = self.order # If MTF has order, use that
    #     else:
    #         trunc_order = order
    
    #     truncated_coefficients = {}
    #     for index in self.coefficients:
    #         if sum(index) <= trunc_order:
    #             truncated_coefficients[index] = self.coefficients[index]
    
    #     return MultivariateTaylorFunction(coefficients=truncated_coefficients,
    #                                        dimension=self.dimension,
    #                                        expansion_point=self.expansion_point,
    #                                        order=trunc_order) # Keep/set the order



    def truncate(self, order=None):
        """
        Truncate the Taylor expansion to a specified order.
        If no order is specified, the global maximum order is used.
        """
        if order is None:
            order = get_global_max_order() # Get global max order when order is None
            print(f"Order was None, set order to global max order: {order}") # Debug print - keep debug print temporarily
        truncated_coeffs = {}
        for index, coeff in self.coefficients.items():
            if sum(index) <= order:
                truncated_coeffs[index] = coeff
        calculated_order = min(order, self.order) if self.order is not None else order # Correctly handle None self.order
        print(f"Truncated coefficients: {truncated_coeffs}") # Debug print - keep debug print temporarily
        print(f"Calculated order for MTF: {calculated_order}") # Debug print - keep debug print temporarily
        return MultivariateTaylorFunction(coefficients=truncated_coeffs, dimension=self.dimension, expansion_point=self.expansion_point, order=calculated_order)

    def __add__(self, other):
        """
        Override the addition operator for MTF addition and scalar addition.
        """
        if isinstance(other, MultivariateTaylorFunction):
            if self.dimension != other.dimension:
                raise ValueError("Dimensions of MTFs must match for addition.")
            new_coefficients = self.coefficients.copy()
            for index, coeff in other.coefficients.items():
                new_coefficients[index] = new_coefficients.get(index, np.array([0.0])) + coeff
            result = MultivariateTaylorFunction(coefficients=new_coefficients, expansion_point=self.expansion_point, dimension=self.dimension, order=max(self.order, other.order) if self.order is not None and other.order is not None else self.order).truncate()
            return result
        elif isinstance(other, (int, float)):
            new_coefficients = self.coefficients.copy()
            constant_index = (0,) * self.dimension
            new_coefficients[constant_index] = new_coefficients.get(constant_index, np.array([0.0])) + np.array([float(other)])
            result = MultivariateTaylorFunction(coefficients=new_coefficients, expansion_point=self.expansion_point, dimension=self.dimension, order=self.order).truncate()
            return result
        else:
            return NotImplemented

    def __radd__(self, other):
        """
        Override reverse addition.
        """
        return self.__add__(other)


    def __sub__(self, other):
        """
        Override the subtraction operator for MTF subtraction and scalar subtraction.
        """
        if isinstance(other, MultivariateTaylorFunction):
            if self.dimension != other.dimension:
                raise ValueError("Dimensions of MTFs must match for subtraction.")
            new_coefficients = self.coefficients.copy()
            for index, coeff in other.coefficients.items():
                new_coefficients[index] = new_coefficients.get(index, np.array([0.0])) - coeff
            result = MultivariateTaylorFunction(coefficients=new_coefficients, expansion_point=self.expansion_point, dimension=self.dimension, order=max(self.order, other.order) if self.order is not None and other.order is not None else self.order).truncate()
            return result
        elif isinstance(other, (int, float)):
            new_coefficients = self.coefficients.copy()
            constant_index = (0,) * self.dimension
            new_coefficients[constant_index] = new_coefficients.get(constant_index, np.array([0.0])) - np.array([float(other)])
            result = MultivariateTaylorFunction(coefficients=new_coefficients, expansion_point=self.expansion_point, dimension=self.dimension, order=self.order).truncate()
            return result
        else:
            return NotImplemented

    def __rsub__(self, other):
        """
        Override reverse subtraction for scalar - MTF.
        """
        if isinstance(other, (int, float)):
            neg_self = -self
            return other + neg_self # Use addition and negation
        else:
            return NotImplemented

    def _multiply_mtf(self, other):
        """
        Multiplication logic for MTF * MTF.
        """
        new_coefficients = {}
        for index1, coeff1 in self.coefficients.items():
            for index2, coeff2 in other.coefficients.items():
                result_index = tuple(i + j for i, j in zip(index1, index2))
                new_coefficients[result_index] = new_coefficients.get(result_index, np.array([0.0])) + coeff1 * coeff2
        return new_coefficients


    def _multiply_scalar(self, scalar):
        """
        Multiplication logic for MTF * scalar.
        """
        new_coefficients = {}
        for index, coeff in self.coefficients.items():
            new_coefficients[index] = coeff * np.array([float(scalar)]) # Make scalar a numpy array
        return new_coefficients

    def __mul__(self, other):
        """
        Override the multiplication operator for MTF multiplication and scalar multiplication.
        """
        if isinstance(other, MultivariateTaylorFunction):
            if self.dimension != other.dimension:
                raise ValueError("Dimensions of MTFs must match for multiplication.")
            new_coefficients = self._multiply_mtf(other)
            result = MultivariateTaylorFunction(coefficients=new_coefficients, expansion_point=self.expansion_point, dimension=self.dimension, order=min(self.order, other.order) if self.order is not None and other.order is not None else self.order).truncate() # Final truncate
            return result
        elif isinstance(other, (int, float)):
            return MultivariateTaylorFunction(coefficients=self._multiply_scalar(other), expansion_point=self.expansion_point, dimension=self.dimension, order=self.order).truncate()
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        Override reverse multiplication.
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Override division operator for MTF / scalar. MTF / MTF is not implemented.
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide MTF by zero.")
            scalar_inverse = 1.0 / other
            return self * scalar_inverse # Use multiplication by scalar inverse
        elif isinstance(other, MultivariateTaylorFunction):
            raise NotImplementedError("Division of MTF by another MTF is not implemented. Use inverse() and multiplication instead.")
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """
        Override reverse division for scalar / MTF.
        """
        raise NotImplementedError("Right division of scalar by MTF is not implemented.")


    def __pow__(self, exponent):
        """
        Override power operator for MTF ** integer exponent.
        """
        if not isinstance(exponent, int):
            raise ValueError("MTF power exponent must be an integer.")
        if exponent < 0:
            raise ValueError("Negative power exponents are not supported for MTF power. Use inverse() for negative powers.")
        if exponent == 0:
            return MultivariateTaylorFunction.from_constant(1.0, dimension=self.dimension) # MTF to the power 0 is constant 1
        if exponent == 1:
            return self # MTF to the power 1 is self

        result = self
        for _ in range(exponent - 1): # Multiply exponent-1 times, as result is already self^1
            result = result * self # Use MTF multiplication
        return result.truncate()


    def __neg__(self):
        """
        Override negation operator for -MTF.
        """
        new_coefficients = {}
        for index, coeff in self.coefficients.items():
            new_coefficients[index] = -coeff
        return MultivariateTaylorFunction(coefficients=new_coefficients, expansion_point=self.expansion_point, dimension=self.dimension, order=self.order).truncate()


    def derivative(self, wrt_variable_id):
        """
        Compute the derivative of the MTF with respect to a specified variable.

        Parameters:
            wrt_variable_id (int): The ID of the variable with respect to which to differentiate (1-indexed).

        Returns:
            MultivariateTaylorFunction: The derivative of the MTF.
        """
        if not isinstance(wrt_variable_id, int) or not 1 <= wrt_variable_id <= self.dimension:
            raise ValueError(f"Invalid variable ID: {wrt_variable_id}. Must be an integer between 1 and {self.dimension}.")

        new_coefficients = {}
        for index, coeff in self.coefficients.items():
            if index[wrt_variable_id-1] > 0:
                deriv_coefficient = coeff * np.array([float(index[wrt_variable_id-1])]) # Scalar multiplication
                deriv_index_list = list(index) # Convert tuple to list for modification
                deriv_index_list[wrt_variable_id-1] -= 1 # Decrement the exponent for differentiation variable
                deriv_index = tuple(deriv_index_list) # Convert back to tuple
                new_coefficients[deriv_index] = deriv_coefficient

        return MultivariateTaylorFunction(coefficients=new_coefficients,
                                           dimension=self.dimension,
                                           expansion_point=self.expansion_point,
                                           order=self.order).truncate() # Truncate after derivative


    def integrate(self, wrt_variable_id, integration_constant=0.0):
        """
        Compute the integral of the MTF with respect to a specified variable.

        Parameters:
            wrt_variable_id (int): The ID of the variable with respect to which to integrate (1-indexed).
            integration_constant (float): The constant of integration.

        Returns:
            MultivariateTaylorFunction: The integral of the MTF.
        """
        if not isinstance(wrt_variable_id, int) or not 1 <= wrt_variable_id <= self.dimension:
            raise ValueError(f"Invalid variable ID: {wrt_variable_id}. Must be an integer between 1 and {self.dimension}.")

        new_coefficients = {}
        # Integrate each term
        for index, coeff in self.coefficients.items():
            integ_index_list = list(index) # Convert tuple to list for modification
            integ_index_list[wrt_variable_id-1] += 1 # Increment exponent for integration variable
            integ_index = tuple(integ_index_list) # Convert back to tuple
            integ_coefficient = coeff / np.array([float(integ_index[wrt_variable_id-1])]) # Division by new exponent
            new_coefficients[integ_index] = integ_coefficient

        # Add integration constant term
        constant_index = (0,) * self.dimension
        new_coefficients[constant_index] = new_coefficients.get(constant_index, np.array([0.0])) + np.array([float(integration_constant)])

        return MultivariateTaylorFunction(coefficients=new_coefficients,
                                           dimension=self.dimension,
                                           expansion_point=self.expansion_point,
                                           order=self.order).truncate() # Truncate after integration


    def compose(self, substitution_dict):
        """
        Compose this MTF with other MTFs or constants.

        Parameters:
            substitution_dict (dict): A dictionary where keys are Var objects and values are MTFs or constants.

        Returns:
            MultivariateTaylorFunction: The composed MTF.
        """
        composed_coefficients = {(0,) * self.dimension: np.array([0.0])} # Initialize with zero constant term

        def expand_term(term_coeff, term_index):
            """Recursive helper function to expand a term in the polynomial composition."""
            current_term_result_coeffs = {(0,) * self.dimension: term_coeff} # Start with the coefficient itself
            for var_pos, var_exponent in enumerate(term_index):
                if var_exponent > 0:
                    var_id = var_pos + 1
                    substitution_value = substitution_dict.get(next((v for v in substitution_dict if v.var_id == var_id and v.dimension == self.dimension), None)) # Find Var by id and dimension

                    if substitution_value is None:
                        raise ValueError(f"No substitution provided for variable with ID {var_id} and dimension {self.dimension}.")

                    if isinstance(substitution_value, MultivariateTaylorFunction):
                        # Substitute with MTF
                        expanded_substitution_coeffs = {}
                        identity_mtf_x_var = MultivariateTaylorFunction.identity_function(dimension=self.dimension, var_ids=[var_id]) # MTF for identity function of x_var
                        power_mtf = identity_mtf_x_var**var_exponent # x_var^exponent
                        substitution_mtf_power = substitution_value**var_exponent # g(x_var)^exponent or h(y_var)^exponent

                        term_product_coeffs = {}
                        term_product_mtf = MultivariateTaylorFunction(coefficients={(0,) * self.dimension: np.array([1.0])}, dimension=self.dimension) # Initialize product MTF to 1
                        for _ in range(var_exponent):
                            term_product_mtf = term_product_mtf * substitution_value # Multiply substitution_value var_exponent times
                        term_product_coeffs = term_product_mtf.coefficients


                        # Multiply current_term_result_coeffs with substitution MTF
                        next_term_result_coeffs = {}
                        for current_index, current_coeff in current_term_result_coeffs.items():
                            for substitution_index, substitution_coeff in term_product_coeffs.items():
                                new_index = tuple(sum(pair) for pair in zip(current_index, substitution_index))
                                next_term_result_coeffs[new_index] = next_term_result_coeffs.get(new_index, np.array([0.0])) + current_coeff * substitution_coeff
                        current_term_result_coeffs = next_term_result_coeffs


                    elif isinstance(substitution_value, (int, float)):
                        # Substitute with constant
                        constant_mtf = MultivariateTaylorFunction.from_constant(substitution_value, dimension=self.dimension)
                        identity_mtf_x_var = MultivariateTaylorFunction.identity_function(dimension=self.dimension, var_ids=[var_id]) # MTF for identity function of x_var
                        power_mtf = identity_mtf_x_var**var_exponent # x_var^exponent
                        substitution_mtf_power = constant_mtf**var_exponent # constant^exponent - which is still a constant MTF


                        term_product_coeffs = substitution_mtf_power.coefficients # Coefficients for constant substitution

                        # Multiply current_term_result_coeffs with substitution MTF (constant)
                        next_term_result_coeffs = {}
                        for current_index, current_coeff in current_term_result_coeffs.items():
                            for substitution_index, substitution_coeff in term_product_coeffs.items(): # Should only be one constant term
                                new_index = tuple(sum(pair) for pair in zip(current_index, substitution_index))
                                next_term_result_coeffs[new_index] = next_term_result_coeffs.get(new_index, np.array([0.0])) + current_coeff * substitution_coeff
                        current_term_result_coeffs = next_term_result_coeffs
                    else:
                        raise TypeError("Substitution value must be MTF or scalar.")
            return current_term_result_coeffs


        for index, coeff in self.coefficients.items():
            term_contribution_coeffs = expand_term(coeff, index) # Expand each term
            # Add contribution of this term to composed_coefficients
            for term_index, term_coeff in term_contribution_coeffs.items():
                composed_coefficients[term_index] = composed_coefficients.get(term_index, np.array([0.0])) + term_coeff


        return MultivariateTaylorFunction(coefficients=composed_coefficients,
                                           dimension=self.dimension,
                                           expansion_point=self.expansion_point,
                                           order=self.order).truncate() # Truncate after composition


    def inverse(self, order):
        """
        Calculate the inverse of the MultivariateTaylorFunction using polynomial inversion.
    
        Only implemented for dimension=1.
    
        Parameters:
            order (int): The order of the Taylor expansion of the inverse function.
    
        Returns:
            MultivariateTaylorFunction: The inverse MTF.
        """
        if self.dimension != 1:
            raise NotImplementedError("Inverse function is only implemented for dimension=1.")
        if (0,) * self.dimension not in self.coefficients or np.isclose(self.coefficients.get((0,) * self.dimension), 0):
            raise ValueError("Constant term of MTF must be non-zero to have an inverse.")
    
        inverse_coefficients = {(0,) * self.dimension: np.array([1.0 / self.coefficients.get((0,) * self.dimension)])}
        print(f"Initial inverse_coefficients: {inverse_coefficients}")
        for order_val in range(1, order + 1):
            current_coeff_index = (order_val,) * self.dimension
            current_coeff_sum = np.array([0.0])
            print(f"\nOrder_val: {order_val}, current_coeff_index: {current_coeff_index}")
            for term_order in range(1, order_val + 1):
                index_pair = ((term_order,) * self.dimension, (order_val - term_order,) * self.dimension)
                coeff1 = self.coefficients.get(index_pair[0], np.array([0.0]))
                coeff2 = inverse_coefficients.get(index_pair[1], np.array([0.0]))
                print(f"  Term order: {term_order}, index_pair: {index_pair}, coeff1: {coeff1}, coeff2: {coeff2}")
                current_coeff_sum += coeff1 * coeff2
                print(f"  Current_coeff_sum: {current_coeff_sum}")
            current_coeff = -inverse_coefficients.get((0,) * self.dimension) * current_coeff_sum
            print(f"  b_(0,): {-inverse_coefficients.get((0,) * self.dimension)}, current_coeff_sum: {current_coeff_sum}, current_coeff: {current_coeff}")
            inverse_coefficients[current_coeff_index] = current_coeff
            print(f"  inverse_coefficients updated: {inverse_coefficients}")
        return MultivariateTaylorFunction(coefficients=inverse_coefficients, dimension=self.dimension, expansion_point=self.expansion_point, order=order)

    # def __str__(self):
    #     """
    #     Return a string representation of the MTF.
    #     """
    #     terms = []
    #     for index, coeff in self.coefficients.items():
    #         term_str = f"{coeff[0]:.2f}" # Coefficient value
    #         for i, exp in enumerate(index):
    #             if exp > 0:
    #                 var_name = f"x_{i+1}" # Variable name (x_1, x_2, ...)
    #                 term_str += f"*{var_name}^{exp}"
    #         terms.append(term_str)
    #     return " + ".join(terms) if terms else "0.0"

    def __str__(self):
        """
        Returns a string representation of the Taylor expansion (tabular format).

        This method is called when you use `str(mtf_object)` or `print(mtf_object)`.
        It leverages `get_tabular_string` to produce the tabular output.

        Returns:
            str: Tabular string representation of the Taylor expansion.
        """
        return self.get_tabular_string() # Simply call and return tabular string
    
    def __repr__(self):
        """
        Return a detailed string representation of the MTF, including coefficients and metadata.
        """
        coeff_repr = ", ".join([f"{index}: {coeff}" for index, coeff in self.coefficients.items()])
        return (f"MultivariateTaylorFunction(coefficients={{{coeff_repr}}}, dimension={self.dimension}, "
                f"expansion_point={list(self.expansion_point)}, order={self.order})")


    def get_tabular_string(self, order=None, variable_names=None):
        """
        Returns a string representing the Taylor expansion in a tabular format.

        Parameters:
            order (int, optional): Maximum order of terms to include in the table.
                                    Defaults to global maximum order if None.
            variable_names (list of str, optional): Names of variables for column headers.
                                                     Defaults to ['x_1', 'x_2', ...] if None.

        Returns:
            str: Tabular string representation of the Taylor expansion.
        """
        truncated_function = self.truncate(order) # Truncate for display
        if variable_names is None:
            variable_names = [f"x_{i+1}" for i in range(self.dimension)] # Default variable names

        header_row = "| Term Index | Coefficient | Order | Exponents "
        if variable_names:
            header_row += "| " + " | ".join(variable_names) + " |"
        else:
            header_row += "|"
        header_row += "\n"

        separator_row = "|------------|-------------|-------|-----------"
        if variable_names:
            separator_row += "|" + "-----|" * self.dimension + "-"
        separator_row += "\n"


        table_str = header_row + separator_row
        term_count = 0 # Initialize term counter for Term Index

        # Sort terms by total order for better readability
        sorted_items = sorted(truncated_function.coefficients.items(), key=lambda item: sum(item[0]))


        for multi_index, coefficient in sorted_items:
            term_count += 1
            term_order = sum(multi_index)
            index_str = f"| {term_count:<10} "
            coefficient_str = f"| {coefficient[0]:11.8f} " # Adjusted coefficient width
            order_str = f"| {term_order:<5} "
            exponent_str = f"| { ' '.join(map(str, multi_index)) :<9} " # Adjusted exponent width

            row = index_str + coefficient_str + order_str + exponent_str

            if variable_names:
                for exponent in multi_index:
                    row += f"| {exponent:5} "
                row += "|"
            else:
                row += "|"

            table_str += row + "\n"

        return table_str


    def print_tabular(self, order=None, variable_names=None, coeff_format="{:15.3e}"):
        """
        Prints the tabular representation of the Taylor expansion to the console.

        This method is a convenience wrapper around `get_tabular_string` that directly prints the output.

        Parameters:
            order (int, optional): Maximum order of terms to include in the table.
                                    Defaults to global maximum order if None.
            variable_names (list of str, optional): Names of variables for column headers.
                                                     Defaults to ['x_1', 'x_2', ...] if None.
            coeff_format (str, optional): Format string for coefficients in the table.
        """
        print(self.get_tabular_string(order=order, variable_names=variable_names))
# from variables import Var
# from taylor_operations import set_global_max_order