import numpy as np
from .MTFExtended import MTF
from . import taylor_function as mtf_lib

class TaylorMap:
    """
    Represents a function from an N-dimensional space to an M-dimensional space
    using an array of MultivariateTaylorFunction objects.
    """

    def __init__(self, components: list[MTF]):
        """
        Initializes a TaylorMap object.

        Args:
            components: A list of MTF objects.
        """
        self.components = np.array(components)
        self.map_dim = len(components)

    def __add__(self, other):
        """
        Performs element-wise addition with another TaylorMap.
        """
        if not isinstance(other, TaylorMap):
            raise TypeError("Can only add a TaylorMap to another TaylorMap.")
        if self.map_dim != other.map_dim:
            raise ValueError("TaylorMap dimensions must match for addition.")

        new_components = self.components + other.components
        return TaylorMap(list(new_components))

    def __sub__(self, other):
        """
        Performs element-wise subtraction with another TaylorMap.
        """
        if not isinstance(other, TaylorMap):
            raise TypeError("Can only subtract a TaylorMap from another TaylorMap.")
        if self.map_dim != other.map_dim:
            raise ValueError("TaylorMap dimensions must match for subtraction.")

        new_components = self.components - other.components
        return TaylorMap(list(new_components))

    def __mul__(self, other):
        """
        Performs element-wise multiplication with another TaylorMap.
        """
        if not isinstance(other, TaylorMap):
            raise TypeError("Can only multiply a TaylorMap by another TaylorMap.")
        if self.map_dim != other.map_dim:
            raise ValueError("TaylorMap dimensions must match for multiplication.")

        new_components = self.components * other.components
        return TaylorMap(list(new_components))

    def compose(self, other):
        """
        Performs composition self(other(x)).
        'other' is a map from K -> N, where N is the input dimension of 'self'.
        'self' is a map from N -> M.
        The result is a map from K -> M.
        """
        if not isinstance(other, TaylorMap):
            raise TypeError("Composition is only defined between two TaylorMap objects.")

        if self.map_dim == 0:
            return TaylorMap([])

        self_input_dim = self.components[0].dimension

        if self_input_dim != other.map_dim:
            raise ValueError(f"Cannot compose maps: self input dimension ({self_input_dim}) "
                             f"must equal other output dimension ({other.map_dim}).")

        new_components = []
        # The new dimension will be the input dimension of the 'other' map.
        new_dimension = other.components[0].dimension if other.map_dim > 0 else 0

        for component_mtf in self.components:
            composed_component = MTF.from_constant(0.0, dimension=new_dimension)
            for i in range(len(component_mtf.coeffs)):
                exponent = component_mtf.exponents[i]
                coeff = component_mtf.coeffs[i]

                term_mtf = MTF.from_constant(1.0, dimension=new_dimension)
                for var_idx, power in enumerate(exponent):
                    if power > 0:
                        term_mtf *= other.components[var_idx] ** power

                composed_component += term_mtf * coeff

            new_components.append(composed_component)

        return TaylorMap(new_components)

    def get_component(self, index: int) -> MTF:
        """
        Returns the MTF at the given index.
        """
        return self.components[index]

    def get_coefficient(self, component_index: int, exponent_array: np.ndarray) -> float:
        """
        Returns the coefficient for a specific term in a specific component.
        """
        return self.components[component_index].extract_coefficient(tuple(exponent_array)).item()

    def set_coefficient(self, component_index: int, exponent_array: np.ndarray, new_value: float):
        """
        Sets the coefficient for a specific term in a specific component.
        """
        self.components[component_index].set_coefficient(tuple(exponent_array), new_value)

    def add_component(self, new_component: MTF):
        """
        Adds a new component to the map.
        """
        self.components = np.append(self.components, new_component)
        self.map_dim = len(self.components)

    def remove_component(self, index: int):
        """
        Removes a component from the map.
        """
        self.components = np.delete(self.components, index)
        self.map_dim = len(self.components)

    def truncate(self, order: int):
        """
        Returns a new TaylorMap with all components truncated to a given order.
        """
        new_components = [c.truncate(order) for c in self.components]
        return TaylorMap(new_components)

    def trace(self):
        """
        Calculates the trace of the first-order part of the map.
        This is only defined for maps from N-dim to N-dim space.
        """
        if self.map_dim == 0:
            return 0.0

        if self.map_dim != self.components[0].dimension:
            raise ValueError("Trace is only defined for maps from N-dim to N-dim space.")

        trace_val = 0.0
        for i in range(self.map_dim):
            exponent = [0] * self.map_dim
            exponent[i] = 1
            trace_val += self.get_coefficient(i, np.array(exponent))

        return trace_val

    def map_sensitivity(self, scaling_factors: list[float]):
        """
        Returns a new TaylorMap with coefficients scaled for sensitivity analysis.
        """
        if self.map_dim > 0 and len(scaling_factors) != self.components[0].dimension:
            raise ValueError("Number of scaling factors must match input dimension.")

        new_map_components = []
        for component in self.components:
            new_coeffs = component.coeffs.copy()
            for i in range(len(component.coeffs)):
                exponent = component.exponents[i]
                scaling_factor = np.prod(np.array(scaling_factors) ** exponent)
                new_coeffs[i] *= scaling_factor

            new_component = MTF((component.exponents.copy(), new_coeffs), dimension=component.dimension)
            new_map_components.append(new_component)

        return TaylorMap(new_map_components)

    def substitute(self, variable_map: dict):
        """
        Performs partial or full substitution.

        Args:
            variable_map: A dict of {var_index: value}, where value can be a number.
                          var_index is 1-based.

        Returns:
            A new TaylorMap if the substitution is partial, or a NumPy array of floats
            if the substitution is full.
        """
        if self.map_dim == 0:
            return np.array([])

        is_full_numeric = all(isinstance(v, (int, float, complex, np.number)) for v in variable_map.values()) and \
                          len(variable_map) == self.components[0].dimension

        if is_full_numeric:
            eval_point = [0] * self.components[0].dimension
            for i, val in variable_map.items():
                eval_point[i-1] = val

            return np.array([c.eval(eval_point).item() for c in self.components])

        new_components = []
        for component in self.components:
            new_component = component.copy()
            for var_index, value in variable_map.items():
                if isinstance(value, (int, float, complex, np.number)):
                    new_component.substitute_variable_inplace(var_index, value)
                else:
                    raise TypeError(f"Unsupported substitution type: {type(value)}. Only numeric values are supported in substitute.")
            new_components.append(new_component)

        return TaylorMap(new_components)

    def __repr__(self):
        """
        Returns a concise string representation of the TaylorMap.
        """
        if self.map_dim > 0:
            return f"TaylorMap(map_dim={self.map_dim}, input_dim={self.components[0].dimension})"
        return "TaylorMap(map_dim=0, input_dim=N/A)"

    def __str__(self):
        """
        Returns a detailed string representation of the TaylorMap.
        """
        if self.map_dim == 0:
            return "Empty TaylorMap"

        representation = f"TaylorMap with {self.map_dim} components (input dim: {self.components[0].dimension}):\n"
        for i, component in enumerate(self.components):
            representation += f"--- Component {i+1} ---\n"
            representation += str(component) + "\n"
        return representation

    def invert(self):
        """
        Computes the inverse of the TaylorMap object using a fixed-point iteration.

        This method is based on the inversion of a transfer map using a
        Differential Algebraic (DA) approach. The formula used is the standard
        iterative method F⁻¹_{n+1} = β⁻¹ ∘ (I - G ∘ F⁻¹_n), where β is the
        linear part of the map and G is the non-linear part.

        Returns:
            A new TaylorMap object representing the inverse map F⁻¹.

        Raises:
            ValueError: If the map is not a square map, has constant terms, or
                        if its linear part is not invertible.
        """
        # --- Pre-condition Checks ---
        if self.map_dim == 0:
            return TaylorMap([]) # Inverse of an empty map is an empty map

        dim = self.components[0].dimension
        if self.map_dim != dim:
            raise ValueError(f"Map must be square to be invertible (input_dim={dim}, output_dim={self.map_dim}).")

        # Check for constant terms
        zero_exp = tuple([0] * dim)
        for i, component in enumerate(self.components):
            const_term = component.extract_coefficient(zero_exp).item()
            if abs(const_term) > 1e-14: # Use a tolerance for floating point
                raise ValueError(f"Map must have no constant terms to be invertible. Component {i} has constant term {const_term}.")

        # --- Algorithm Step 1: Extract Linear Part (β) and Jacobian ---
        jacobian = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                exp = tuple([1 if k == j else 0 for k in range(dim)])
                coeff = self.get_coefficient(i, np.array(exp))
                jacobian[i, j] = coeff

        # Check for invertible linear part
        if abs(np.linalg.det(jacobian)) < 1e-14:
            raise ValueError("The linear part of the map is not invertible (Jacobian is singular).")

        # --- Algorithm Step 4 (early): Invert Linear Part (β⁻¹) ---
        inv_jacobian = np.linalg.inv(jacobian)
        inv_linear_components = []
        for i in range(dim):
            comp_mtf = MTF.from_constant(0.0, dimension=dim)
            for j in range(dim):
                if abs(inv_jacobian[i, j]) > 1e-14:
                    var_mtf = MTF.from_variable(j + 1, dim)
                    comp_mtf += inv_jacobian[i, j] * var_mtf
            inv_linear_components.append(comp_mtf)
        beta_inv = TaylorMap(inv_linear_components)

        # --- Algorithm Step 2 & 3: Extract G and create Identity ---
        # We don't need to create beta explicitly. G = F - beta
        # The non-linear part G can be calculated on the fly.
        # Let's get the full non-linear part first.

        non_linear_components = []
        for component in self.components:
            # Create a new MTF with only non-linear terms (order > 1)
            nl_coeffs = {}
            for k in range(len(component.coeffs)):
                exp = tuple(component.exponents[k])
                if sum(exp) > 1:
                    nl_coeffs[exp] = component.coeffs[k]
            non_linear_components.append(MTF(nl_coeffs, dimension=dim))
        G = TaylorMap(non_linear_components)

        identity_components = [MTF.from_variable(i + 1, dim) for i in range(dim)]
        identity_map = TaylorMap(identity_components)

        # --- Algorithm Step 5: Fixed-Point Iteration ---
        F_inv = beta_inv # Initial guess

        max_order = mtf_lib.get_global_max_order()
        for _ in range(max_order):
            composition_G_F_inv = G.compose(F_inv).truncate(max_order)
            inner_map = identity_map - composition_G_F_inv
            F_inv = beta_inv.compose(inner_map).truncate(max_order)

        return F_inv
