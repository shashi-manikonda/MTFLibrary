import numpy as np
from collections import defaultdict

# Global setting for maximum order of Taylor expansion - v9.9
_GLOBAL_MAX_ORDER = 10  # Or some other default value

def set_global_max_order(order):
    """
    Set the global maximum order for Taylor expansions.
    """
    global _GLOBAL_MAX_ORDER
    if not isinstance(order, int) or order < 0:
        raise ValueError("Global max order must be a non-negative integer.")
    _GLOBAL_MAX_ORDER = order

def get_global_max_order():
    """
    Get the current global maximum order for Taylor expansions.
    """
    return _GLOBAL_MAX_ORDER

class Var:
    """
    Represents an independent variable in a Multivariate Taylor Function.
    Each variable is uniquely identified by an id.
    """
    _instance_counter = 0

    def __init__(self, var_id=None, dimension=1):
        if var_id is None:
            Var._instance_counter += 1
            self.var_id = Var._instance_counter
        else:
            self.var_id = var_id
        self.dimension = dimension # Dimension of the variable - v9.9

    def __eq__(self, other):
        return isinstance(other, Var) and self.var_id == other.var_id and self.dimension == other.dimension

    def __hash__(self):
        return hash((self.var_id, self.dimension))

    def __str__(self):
        return f"Var(id={self.var_id}, dimension={self.dimension})"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        """Define addition for Var + other (scalar or Var/MTF)."""
        return convert_to_mtf(self) + other

    def __radd__(self, other):
        """Define addition for other (scalar or Var/MTF) + Var."""
        return other + convert_to_mtf(self)

    def __sub__(self, other):
        """Define subtraction for Var - other (scalar or Var/MTF)."""
        return convert_to_mtf(self) - other

    def __rsub__(self, other):
        """Define subtraction for other (scalar or Var/MTF) - Var."""
        return other - convert_to_mtf(self)

    def __mul__(self, other):
        """Define multiplication for Var * other (scalar or Var/MTF)."""
        return convert_to_mtf(self) * other

    def __rmul__(self, other):
        """Define multiplication for other (scalar or Var/MTF) * Var."""
        return other * convert_to_mtf(self)

    # Division operations - only with scalars allowed for Var directly
    def __truediv__(self, other):
        """Define division for Var / scalar."""
        if isinstance(other, (int, float, np.number)):
            return convert_to_mtf(self) / other
        else:
            raise TypeError("Division of Var by non-scalar or MTF is not supported.")

    def __rtruediv__(self, other):
        """Define division for scalar / Var."""
        raise TypeError("Scalar division by Var is not directly supported for Taylor expansion around origin. Inverse of Var at origin is undefined.")    

    def __neg__(self):
        """Define negation of a Var object."""
        return -1 * convert_to_mtf(self) # -1 * MTF
    
    
    def __pow__(self, exponent):
        """Define exponentiation for MTF ** integer exponent."""
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer for MTF**exponent.")
        if exponent < 0:
            raise ValueError("Negative exponents are not supported for MTF**exponent in this simplified version.")
        if exponent == 0:
            return MultivariateTaylorFunction.from_constant(1.0, dimension=self.dimension) # MTF**0 = 1 (constant MTF)
        if exponent == 1:
            return self # MTF**1 = self

        result_mtf = self # Start with base MTF
        for _ in range(exponent - 1): # Multiply 'exponent - 1' times
            result_mtf = result_mtf * self # Use MTF multiplication

        return result_mtf


class MultivariateTaylorFunction:
    """
    Represents a Multivariate Taylor Function.
    coefficients: defaultdict of coefficients, keys are tuples of exponents.
    dimension: number of variables, determined by the length of exponent tuples.
    expansion_point: the point around which the Taylor expansion is defined. Default is origin. - v9.9
    """
    def __init__(self, coefficients, dimension, var_list=None, expansion_point=None):
        self.coefficients = coefficients # defaultdict
        self.dimension = dimension
        self.var_list = var_list if var_list is not None else [Var(i+1, dimension=dimension) for i in range(dimension)] # List of Var objects - v9.9
        self.expansion_point = expansion_point if expansion_point is not None else np.zeros(dimension) # Expansion point, default origin - v9.9


    @classmethod
    def from_constant(cls, constant_value, dimension):
        """
        Create a MultivariateTaylorFunction from a constant value.
        """
        coeffs = defaultdict(lambda: np.array([0.0])) # Defaultdict for coefficients - v9.9
        coeffs[(0,) * dimension] = np.array([float(constant_value)]) # Corrected for multi-dimension - v9.9
        return cls(coefficients=coeffs, dimension=dimension)


    def __call__(self, evaluation_points):
        """
        Evaluate the MTF at given evaluation points.

        Args:
            evaluation_points (np.ndarray): Points to evaluate at.
                                            Shape (dimension,) for single point, or (n_points, dimension) for multiple points.

        Returns:
            np.ndarray: Evaluated values. Shape () for single point, or (n_points,) for multiple points.
        """
        evaluation_points = np.asarray(evaluation_points)

        if evaluation_points.ndim == 1:
            if evaluation_points.shape != (self.dimension,):
                raise ValueError(f"Evaluation point must be of dimension {self.dimension}, but got {evaluation_points.shape}")
            return self._evaluate_single_point(evaluation_points)
        elif evaluation_points.ndim == 2:
            if evaluation_points.shape[1] != self.dimension:
                raise ValueError(f"Evaluation points must have dimension {self.dimension}, but got {evaluation_points.shape[1]}")
            return np.array([self._evaluate_single_point(point) for point in evaluation_points])
        else:
            raise ValueError("Evaluation points must be either 1D or 2D numpy array.")


    def _evaluate_single_point(self, point):
        """Evaluate at a single point (1D numpy array)."""
        value = 0.0
        for exponents, coefficient in self.coefficients.items():
            term_value = coefficient[0] # Get scalar coefficient value - v9.9
            for i in range(self.dimension):
                term_value *= (point[i] - self.expansion_point[i]) ** exponents[i] # Use expansion point - v9.9
            value += term_value
        return value


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
        return f"MultivariateTaylorFunction(dimension={self.dimension}, coefficients={self.coefficients})"


    def __add__(self, other):
        """Addition of two MultivariateTaylorFunction objects or a MTF and a scalar."""
        other_mtf = convert_to_mtf(other, dimension=self.dimension) # Pass dimension
        if self.dimension != other_mtf.dimension:
            raise ValueError("Dimensions must be the same for addition.")
        new_coeffs = defaultdict(lambda: np.array([0.0]))
        all_exponents = set(self.coefficients.keys()) | set(other_mtf.coefficients.keys())
        debug = False # Set to True for debugging info
        if debug:
            print(f"Debug __add__: self.dimension={self.dimension}, other_mtf.dimension={other_mtf.dimension}")
            print(f"Debug __add__: all_exponents={all_exponents}")
        for exponents in all_exponents:
            coeff1 = self.coefficients.get(exponents, np.array([0.0]))
            coeff2 = other_mtf.coefficients.get(exponents, np.array([0.0]))
            if debug:
                print(f"Debug __add__: exponents={exponents}, coeff1={coeff1}, coeff2={coeff2}")
            new_coeffs[exponents] = coeff1 + coeff2
        return MultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension, expansion_point=self.expansion_point)


    def __radd__(self, other):
        """Right-side addition to handle cases like scalar + MTF."""
        return self.__add__(other) # Addition is commutative


    def __sub__(self, other):
        """Subtraction of two MultivariateTaylorFunction objects or a MTF and a scalar."""
        other_mtf = convert_to_mtf(other, dimension=self.dimension) # Pass dimension
        if self.dimension != other_mtf.dimension:
            raise ValueError("Dimensions must be the same for subtraction.")
        new_coeffs = defaultdict(lambda: np.array([0.0]))
        all_exponents = set(self.coefficients.keys()) | set(other_mtf.coefficients.keys())
        for exponents in all_exponents:
            coeff1 = self.coefficients.get(exponents, np.array([0.0]))
            coeff2 = other_mtf.coefficients.get(exponents, np.array([0.0]))
            new_coeffs[exponents] = coeff1 - coeff2
        return MultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension, expansion_point=self.expansion_point)


    def __rsub__(self, other):
        """Right-side subtraction to handle cases like scalar - MTF."""
        other_mtf = convert_to_mtf(other, dimension=self.dimension) # Pass dimension
        if self.dimension != other_mtf.dimension:
            raise ValueError("Dimensions must be the same for subtraction.")
        new_coeffs = defaultdict(lambda: np.array([0.0]))
        all_exponents = set(self.coefficients.keys()) | set(other_mtf.coefficients.keys())
        for exponents in all_exponents:
            coeff1 = convert_to_mtf(other, dimension=self.dimension).coefficients.get(exponents, np.array([0.0])) # scalar converted to MTF here too, with dimension
            coeff2 = self.coefficients.get(exponents, np.array([0.0]))
            new_coeffs[exponents] = coeff1 - coeff2
        return MultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension, expansion_point=self.expansion_point)


    def __mul__(self, other):
        """Multiplication of two MultivariateTaylorFunction objects or a MTF and a scalar."""
        other_mtf = convert_to_mtf(other, dimension=self.dimension) # Pass dimension
        if self.dimension != other_mtf.dimension:
            raise ValueError("Dimensions must be the same for multiplication.")
        new_coeffs = defaultdict(lambda: np.array([0.0]))
        for exp1, coeff1 in self.coefficients.items():
            for exp2, coeff2 in other_mtf.coefficients.items():
                new_exponent = tuple(exp1[i] + exp2[i] for i in range(self.dimension)) # Correct exponent tuple creation - v9.9
                new_coeffs[new_exponent] += coeff1 * coeff2 # Accumulate coefficients - v9.9
        return MultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension, expansion_point=self.expansion_point)


    def __rmul__(self, other):
        """Right-side multiplication to handle cases like scalar * MTF."""
        return self.__mul__(other) # Multiplication is commutative


    def __truediv__(self, other):
        """Division by a scalar."""
        if isinstance(other, (int, float, np.number)): # Handle scalar division directly
            if other == 0:
                raise ZeroDivisionError("Division by zero scalar.")
            new_coeffs = {exp: coeff / other for exp, coeff in self.coefficients.items()}
            return MultivariateTaylorFunction(coefficients=new_coeffs, dimension=self.dimension, expansion_point=self.expansion_point)
        else: # For now, don't handle MTF division in tests, just scalar.
            raise NotImplementedError("Division by MultivariateTaylorFunction is not fully tested yet in this simplified test run. Please focus on scalar division.")


    def __rtruediv__(self, other):
        """Right division (scalar / MTF) - not typically well-defined for Taylor series in general, and inverse is complex."""
        raise NotImplementedError("Scalar division by MultivariateTaylorFunction (scalar / MTF) is not supported in this simplified version.")


    def __neg__(self):
        """Negation of a MultivariateTaylorFunction."""
        neg_coeffs = {exp: -coeff for exp, coeff in self.coefficients.items()}
        return MultivariateTaylorFunction(coefficients=neg_coeffs, dimension=self.dimension, expansion_point=self.expansion_point)
    
    def __pow__(self, exponent):
        """Define exponentiation for MTF ** integer exponent."""
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer for MTF**exponent.")
        if exponent < 0:
            raise ValueError("Negative exponents are not supported for MTF**exponent in this simplified version.")
        if exponent == 0:
            return MultivariateTaylorFunction.from_constant(1.0, dimension=self.dimension) # MTF**0 = 1 (constant MTF)
        if exponent == 1:
            return self # MTF**1 = self

        result_mtf = self # Start with base MTF
        for _ in range(exponent - 1): # Multiply 'exponent - 1' times
            result_mtf = result_mtf * self # Use MTF multiplication

        return result_mtf

    def to_string_table(self):
        """
        Returns a string formatted as a table of exponents and coefficients.
        """
        headers = ["Exponents", "Coefficient"]
        rows = []
        for exponents, coefficient in self.coefficients.items():
            rows.append([str(exponents), str(coefficient)]) # Convert to string for table

        # Calculate column widths - adjust padding as needed
        col_width = [max(len(header), max(len(row[i]) for row in rows)) + 2 for i, header in enumerate(headers)] # +2 for padding

        # Create separator
        separator = "-" * (sum(col_width) + len(headers) - 1) # Correctly calculate separator length

        table_str = separator + "\n"

        # Add headers
        header_row = ""
        for i, header in enumerate(headers):
            header_row += header.ljust(col_width[i]) + "|"
        table_str += header_row[:-1] + "\n" # Remove last "|" and add newline
        table_str += separator + "\n"

        # Add rows
        for row in rows:
            row_str = ""
            for i, cell in enumerate(row):
                row_str += cell.ljust(col_width[i]) + "|"
            table_str += row_str[:-1] + "\n" # Remove last "|" and add newline

        table_str += separator

        return table_str

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
        display_order = order if order is not None else get_global_max_order()
        truncated_function = self.truncate(
            display_order)  # Truncate the MTF to the display order for clarity

        if variable_names is None:
            variable_names = [f"x_{i + 1}" for i in
                              range(self.dimension)]  # Default variable names if none provided

        # Construct header row of the table
        header_row = "| Term Index | Coefficient | Order | Exponents "
        if variable_names:
            header_row += "| " + " | ".join(variable_names) + " |"  # Add variable name columns if provided
        else:
            header_row += "|"
        header_row += "\n"

        # Construct separator row for table formatting
        separator_row = "|------------|-------------|-------|-----------"
        if variable_names:
            separator_row += "|" + "-----|" * self.dimension + "-"  # Add separators for variable name columns
        separator_row += "\n"

        table_str = header_row + separator_row  # Start table string with header and separator
        term_count = 0  # Initialize counter for term index

        # Sort terms by total order for ordered display in table
        sorted_items = sorted(truncated_function.coefficients.items(), key=lambda item: sum(item[0]))

        for multi_index, coefficient in sorted_items:  # Iterate through terms in sorted order
            term_count += 1  # Increment term index for each term
            term_order = sum(multi_index)  # Calculate total order of the term
            index_str = f"| {term_count:<10} "  # Format term index
            coefficient_str = f"| {coefficient[0]:11.8f} "  # Format coefficient value
            order_str = f"| {term_order:<5} "  # Format term order
            exponent_str = f"| {' '.join(map(str, multi_index)):<9} "  # Format exponents as string

            row = index_str + coefficient_str + order_str + exponent_str  # Start row with index, coefficient, order, exponents

            if variable_names:
                for exponent in multi_index:
                    row += f"| {exponent:5} "  # Add column for each variable's exponent with name if variable names provided
                row += "|"
            else:
                row += "|"

            table_str += row + "\n"  # Append row to table string

        return table_str  # Return the complete tabular string


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
        print(self.get_tabular_string(order=order, variable_names=variable_names))  # Generate tabular string and print it to console


    def __repr__(self):  # Added __repr__ for debugging and clarity - v9.8
        return f"MultivariateTaylorFunction(dimension={self.dimension}, order={self.order}, coefficients={self.coefficients}, expansion_point={list(self.expansion_point)})"  # Include expansion_point in repr - v9.8

    def evaluate(self, point):
        if not isinstance(point, np.ndarray) or point.shape != (self.dimension,):  # dimension check - v9.5
            raise ValueError(f"Evaluation point must be a numpy array of shape ({self.dimension},)")  # dimension in error msg - v9.5

        result = np.array([0.0])  # Initialize result as numpy array - v9.4
        for index, coefficient in self.coefficients.items():
            term_value = coefficient
            for var_index, power in enumerate(index):  # Correctly use enumerate for index and power - v9.3
                if power > 0:
                    term_value = term_value * (point[var_index] ** power)  # Use point value - v9.3, corrected power - v9.4
            result += term_value  # Accumulate term value - v9.4
        return result


    def derivative(self, wrt_variable_id):
        """
        Calculate the derivative ... """
        deriv_coeffs = defaultdict(lambda: np.array([0.0]))
        variable_index = wrt_variable_id - 1

        print(f"\n--- Calculating derivative wrt var {wrt_variable_id} ---") # Debug start

        for exponents, coefficient in self.coefficients.items():
            print(f"  Processing term: exponents={exponents}, coefficient={coefficient}") # Debug term input
            if exponents[variable_index] > 0:
                print(f"    Exponent for var {wrt_variable_id} is positive.") # Debug condition met
                new_exponents = list(exponents)
                new_exponents[variable_index] -= 1
                deriv_coefficient_val = coefficient * exponents[variable_index]
                deriv_coeffs[tuple(new_exponents)] = deriv_coefficient_val
                print(f"    Derivative term: new_exponents={tuple(new_exponents)}, deriv_coefficient_val={deriv_coefficient_val}") # Debug derivative term
            else:
                print(f"    Exponent for var {wrt_variable_id} is zero or negative, skipping.") # Debug condition not met
        
        print(f"--- Derivative wrt var {wrt_variable_id} coefficients: {deriv_coeffs} ---") # Debug final coefficients
        return MultivariateTaylorFunction(coefficients=deriv_coeffs, dimension=self.dimension, expansion_point=self.expansion_point)


    def integral(self, wrt_variable_id, integration_constant=0.0):
        """
        Compute the integral of the MTF with respect to a variable.

        Args:
            wrt_variable_id (int): The id of the variable to integrate with respect to (1-indexed).
            integration_constant (float or MultivariateTaylorFunction, optional): The constant of integration. Defaults to 0.0.

        Returns:
            MultivariateTaylorFunction: The integral MTF.
        """
        if not isinstance(wrt_variable_id, int) or not 1 <= wrt_variable_id <= self.dimension:
            raise ValueError(f"Variable ID must be an integer between 1 and {self.dimension}.")

        integrated_coeffs = defaultdict(lambda: np.array([0.0]))
        variable_index = wrt_variable_id - 1 # 0-indexed variable index

        for exponents, coefficient in self.coefficients.items():
            new_exponents = list(exponents)
            new_exponents[variable_index] += 1
            integrated_coeffs[tuple(new_exponents)] = coefficient / new_exponents[variable_index] #Integrate: divide by new exponent

        # Add integration constant term
        constant_mtf = convert_to_mtf(integration_constant, dimension=self.dimension) # Convert constant to MTF of same dimension - v9.9
        return MultivariateTaylorFunction(coefficients=integrated_coeffs, dimension=self.dimension, expansion_point=self.expansion_point) + constant_mtf # Add constant


    def truncate(self, order):
        """
        Truncate the Taylor series to a specified order. Terms with order > 'order' are discarded.

        Args:
            order (int): The maximum order to keep.

        Returns:
            MultivariateTaylorFunction: A new MTF truncated to the specified order.
        """
        if not isinstance(order, int) or order < 0:
            raise ValueError("Truncation order must be a non-negative integer.")
        truncated_coeffs = defaultdict(lambda: np.array([0.0]))
        for exponents, coefficient in self.coefficients.items():
            if sum(exponents) <= order:
                truncated_coeffs[exponents] = coefficient
        return MultivariateTaylorFunction(coefficients=truncated_coeffs, dimension=self.dimension, expansion_point=self.expansion_point)


    def compose(self, substitution_dict, max_order=None):
        """
        Compose the MTF with other MTFs or scalars, substituting variables.

        Args:
            substitution_dict (dict): A dictionary where keys are Var objects to be substituted,
                                        and values are MTF objects or scalars to substitute with.
            max_order (int, optional): Maximum order of the resulting Taylor expansion.
                                        Defaults to global max order.

        Returns:
            MultivariateTaylorFunction: The composed Taylor function.
        """
        if max_order is None:
            max_order = get_global_max_order()
        if not isinstance(max_order, int) or max_order < 0:
            raise ValueError("Max order for composition must be a non-negative integer.")

        # Convert substitution values to MTF
        mtf_substitution_dict = {var: convert_to_mtf(value, dimension=self.dimension) for var, value in substitution_dict.items()} # Ensure dimension match - v9.9

        # Check dimension compatibility - Now checked in convert_to_mtf implicitly via arithmetic ops

        composed_coeffs = defaultdict(lambda: np.array([0.0])) # Initialize coefficients for composed function

        # Calculate constant term by evaluating self at constant terms of substitution MTFs
        constant_term_value = self.coefficients.get((0,) * self.dimension, np.array([0.0])) # Start with constant term of self
        substitution_values = []
        for var_id in range(1, self.dimension + 1): # Iterate through variable IDs of self
            substitution_var = Var(var_id, self.dimension)
            if substitution_var in mtf_substitution_dict:
                substitution_mtf = mtf_substitution_dict[substitution_var]
                substitution_constant_term = substitution_mtf.coefficients.get((0,) * substitution_mtf.dimension, np.array([0.0]))[0] # Get constant term of substitution MTF
                substitution_values.append(substitution_constant_term)
            else:
                substitution_values.append(0.0) # If no substitution for this variable, assume substitution value is 0 at expansion point

        # Evaluate self at substitution constant terms to get the constant term of composed MTF
        constant_term_composed = self(np.array(substitution_values)) # Evaluate MTF at substitution constant terms
        print(f"Substitution values for constant term: {substitution_values}, calculated constant term: {constant_term_composed}") # DEBUG PRINT
        composed_coeffs[(0,) * self.dimension] = np.array([constant_term_composed]) # Set constant term


        initial_order = sum((0,) * self.dimension) # Start from order 0
        terms_queue = [] # Initialize empty queue

        # Queue all terms from self.coefficients up to max_order
        for exponents, coefficient in self.coefficients.items():
            if sum(exponents) <= max_order: # Queue terms within max_order
                terms_queue.append( (coefficient, exponents) ) # Add term to queue


        processed_exponents = set() # Track already processed exponents to avoid re-computation


        while terms_queue:
            current_coefficient, current_exponents = terms_queue.pop(0) # Get term from queue

            print(f"\nProcessing term: exponents={current_exponents}, coefficient={current_coefficient}") # DEBUG PRINT

            if current_exponents in processed_exponents: # Skip if already processed
                continue
            processed_exponents.add(current_exponents) # Mark as processed


            term_order = sum(current_exponents) # Order of current term

            if term_order > max_order: # Skip terms exceeding max order
                continue

            # Handle constant term separately (already initialized)
            if term_order == 0:
                continue

            # Process non-constant terms
            current_coefficient_value = current_coefficient[0] # Get scalar value of coefficient

            linear_contribution = MultivariateTaylorFunction.from_constant(0.0, dimension=self.dimension) # Initialize contribution - MOVED HERE


            for var_id_index in range(self.dimension): # Iterate over each variable dimension
                exponent_order = current_exponents[var_id_index] # Get exponent for current variable

                if exponent_order > 0: # If exponent is positive, it's a derivative term
                    variable_deriv_mtf = self.derivative(wrt_variable_id=var_id_index+1) # Get derivative MTF for this variable
                    print(f"  Derivative MTF coefficients: {variable_deriv_mtf.coefficients}") # DEBUG PRINT - derivative MTF coefficients

                    new_exponents_deriv = list(current_exponents) # Calculate new exponents for derivative term
                    new_exponents_deriv[var_id_index] -= 1
                    deriv_coefficient = variable_deriv_mtf.coefficients.get((0,) * self.dimension, np.array([0.0])) # Fetch CONSTANT term coefficient from derivative MTF
                    print(f"  Deriv coefficient value: {deriv_coefficient}") # DEBUG PRINT - deriv_coefficient value

                    if not np.isclose(deriv_coefficient, 0.0).all(): # If derivative coefficient is non-zero
                        substitution_var = Var(var_id_index + 1, dimension=self.dimension) # Get the Var object by id
                        if substitution_var in mtf_substitution_dict: # Check if variable is in substitution dict
                            substitution_mtf = mtf_substitution_dict[substitution_var] # Get substitution MTF
                            term_contribution = convert_to_mtf(deriv_coefficient, dimension=self.dimension) * substitution_mtf # Multiply derivative coefficient with substitution MTF
                            print(f"  Var index: {var_id_index}, term_contribution coefficients: {term_contribution.coefficients}") # DEBUG PRINT - term_contribution for xy term
                            linear_contribution += term_contribution # Add to linear contribution

            print(f"Linear contribution coefficients: {linear_contribution.coefficients}") # DEBUG PRINT

            composed_coeffs_temp_mtf = MultivariateTaylorFunction(coefficients=composed_coeffs, dimension=self.dimension) # Create MTF for truncate - v9.9
            linear_contribution_truncated = linear_contribution.truncate(order=max_order)

            # Correctly add coefficients from linear_contribution, but skip constant term
            for exponents, coefficient in linear_contribution_truncated.coefficients.items():
                if sum(exponents) > 0: # <--- ADDED CONDITION: Skip constant term (exponents=(0, 0))
                    composed_coeffs[exponents] += coefficient # Accumulate non-constant coefficients

            # Queue up terms from linear_contribution for next iteration, avoid reprocessing constant term
            for exponents, coefficient in linear_contribution.coefficients.items():
                if sum(exponents) > 0 and sum(exponents) <= max_order: # Queue terms within max_order and not constant
                    if exponents not in processed_exponents:
                        print(f"Queueing term from linear_contribution: exponents={exponents}, coefficient={coefficient}") # DEBUG PRINT - Queueing info
                        terms_queue.append( (coefficient, exponents) ) # Add term to queue for processing

        print(f"Final composed coefficients: {composed_coeffs}") # DEBUG PRINT - Final coefficients before return
        return MultivariateTaylorFunction(coefficients=composed_coeffs, dimension=self.dimension, expansion_point=self.expansion_point) # Final truncation and return

    def inverse(self, order=None, initial_guess=None):
        """
        Compute the inverse of the MTF using Taylor expansion up to the given order.
        Algorithm based on recursive determination of Taylor coefficients.

        Args:
            order (int, optional): Order of Taylor expansion for the inverse. Defaults to global max order.
            initial_guess (float, optional): Initial guess for the constant term of the inverse.
                                             If None, defaults to 1/f(expansion_point) where f is self.

        Returns:
            MultivariateTaylorFunction: MTF representing the inverse, truncated to the specified order.

        Raises:
            ValueError: if the constant term of the MTF is zero, as inverse is undefined at origin in that case.
        """
        if order is None:
            order = get_global_max_order()
        if not isinstance(order, int) or order < 0:
            raise ValueError("Order for inverse must be a non-negative integer.")

        # Get the constant term of self
        constant_term_self = self.coefficients.get(tuple([0]*self.dimension), np.array([0.0]))[0] # Scalar value - v9.9

        if np.isclose(constant_term_self, 0.0):
            raise ValueError("Cannot compute inverse for MTF with zero constant term at expansion point.")

        # Determine initial guess for the constant term of the inverse
        if initial_guess is None:
            initial_guess_value = 1.0 / constant_term_self
        else:
            initial_guess_value = float(initial_guess)


        inverse_coeffs = defaultdict(lambda: np.array([0.0]))
        inverse_coeffs[(0,) * self.dimension] = np.array([initial_guess_value]) # Initialize constant term of inverse


        for current_order in range(1, order + 1):
            for exponents in _generate_exponent_combinations(self.dimension, current_order): # Helper function to generate exponents
                if sum(exponents) == current_order: # Consider only exponents of current order
                    term_sum = np.array([0.0])

                    # Summation term calculation
                    for intermediate_order in range(1, current_order + 1):
                        for beta_exponents in _generate_exponent_combinations(self.dimension, intermediate_order):
                            if sum(beta_exponents) == intermediate_order: # exponents beta for self
                                alpha_exponents_tuple_list = _exponents_tuple_generator(exponents, beta_exponents) # Exponents alpha for inverse

                                for alpha_exponents in alpha_exponents_tuple_list: # Iterate over possible alpha exponent tuples
                                    if sum(alpha_exponents) == (current_order - intermediate_order) and sum(alpha_exponents) >=0: # Order check for alpha and beta

                                        coeff_f_beta = self.coefficients.get(beta_exponents, np.array([0.0])) # Coefficient of f for exponent beta
                                        coeff_inverse_alpha = inverse_coeffs.get(alpha_exponents, np.array([0.0])) # Coefficient of inverse for exponent alpha

                                        term_sum += coeff_f_beta * coeff_inverse_alpha

                    # Compute coefficient for current exponent for inverse
                    constant_term_f = self.coefficients.get((0,) * self.dimension, np.array([0.0])) # Constant term of f
                    current_coefficient = - (1.0/constant_term_f[0]) * term_sum # Scalar division - v9.9
                    inverse_coeffs[exponents] = current_coefficient


        return MultivariateTaylorFunction(coefficients=inverse_coeffs, dimension=self.dimension, expansion_point=self.expansion_point).truncate(order=order) # Truncate to requested order


def _generate_exponent_combinations(dimension, order):
    """
    Helper function to generate combinations of exponents for a given order and dimension.
    """
    if dimension == 1:
        for o in range(order + 1):
            yield (o,)
    elif dimension > 1:
        for i in range(order + 1):
            for sub_exponents in _generate_exponent_combinations(dimension - 1, order - i):
                yield (i,) + sub_exponents
    else:
        yield tuple()


def _exponents_tuple_generator(gamma, beta):
    """
    Helper function to generate tuples of exponents alpha such that alpha + beta = gamma (component-wise).
    Handles multi-dimensional exponents. Yields valid tuples of alpha exponents.
    """
    dimension = len(gamma)
    alpha_exponents = [0] * dimension # Initialize alpha exponents for each dimension

    def backtrack(current_dimension_index):
        if current_dimension_index == dimension:
            yield tuple(alpha_exponents) # Found a valid combination, yield it
            return

        for i in range(gamma[current_dimension_index] + 1): # Iterate through possible exponents for current dimension
            alpha_exponents[current_dimension_index] = i
            valid_alpha = True
            if alpha_exponents[current_dimension_index] + beta[current_dimension_index] != gamma[current_dimension_index]:
                valid_alpha = False # current alpha exponent is invalid

            if valid_alpha:
                yield from backtrack(current_dimension_index + 1) # Recurse to next dimension

    yield from backtrack(0) # Start backtracking from the first dimension


def convert_to_mtf(func, dimension=None):
    if isinstance(func, MultivariateTaylorFunction):
        return func
    elif isinstance(func, Var):
        coeffs = defaultdict(lambda: np.array([0.0]))
        coeffs[(1,) + (0,) * (func.dimension - 1)] = np.array([1.0]) # Corrected for multi-dimension - v9.9
        return MultivariateTaylorFunction(coefficients=coeffs, dimension=func.dimension, var_list=[func]) # Include var_list - v9.9
    elif isinstance(func, (int, float, np.number)):
        if dimension is None:
            dimension = 1 # Default dimension for scalars if not specified
        return MultivariateTaylorFunction.from_constant(float(func), dimension=dimension) # Use provided dimension
    elif isinstance(func, np.ndarray): # Handle numpy arrays
        if func.size == 1: # If it's a scalar array
            if dimension is None:
                dimension = 1
            return MultivariateTaylorFunction.from_constant(float(func.item()), dimension=dimension) # Convert to scalar and then to MTF
        else:
            raise TypeError("Unsupported numpy array type for conversion to MultivariateTaylorFunction: array must be scalar.") # Or handle array MTFs if needed in future
    else:
        raise TypeError("Unsupported type for conversion to MultivariateTaylorFunction.")