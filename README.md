# README.md
# MTFLibrary

## Multivariate Taylor Expansion Library

MTFLibrary is a Python package designed for robust and accurate manipulation of Multivariate Taylor Functions (MTFs). It supports both real and complex coefficients and provides a user-friendly API for defining, operating on, and evaluating MTFs.

### Features

*   **Multivariate Taylor Functions:** Represents and manipulates MTFs with real and complex coefficients.
*   **Comprehensive Operations:** Supports arithmetic operations, inverse, differentiation, integration, composition, truncation, and evaluation.
*   **Elementary Functions:** Includes a set of elementary functions implemented using Taylor series (e.g., sin, cos, exp, log).
*   **Modularity and Extensibility:** Designed for easy maintenance and extension.
*   **User-Friendly API:** Clear, intuitive, and well-documented.
*   **Efficiency:** Optimized for performance, especially in coefficient storage and arithmetic operations using NumPy.
*   **Testability and Reliability:** Comprehensive unit tests to ensure correctness.

### Installation

```bash
pip install numpy  # NumPy is required
pip install .       # Install from the package directory

If you are developing and want to install in editable mode:
pip install -e .

### Usage
First, you need to initialize the global settings for MTFLibrary:

from MTFLibrary.taylor_function import initialize_mtf_globals, set_global_etol, get_global_max_order, get_global_max_dimension
from MTFLibrary.taylor_function import Var, MultivariateTaylorFunction, ComplexMultivariateTaylorFunction
import numpy as np

# Initialize global settings
initialize_mtf_globals(max_order=10, max_dimension=2)
set_global_etol(1e-10)

print(f"Max Order: {get_global_max_order()}")
print(f"Max Dimension: {get_global_max_dimension()}")

# Create variables
x = Var(1)
y = Var(2)

# Create a Multivariate Taylor Function
coefficients_real = {
    (0, 0): np.array([1.0]),
    (1, 0): np.array([2.0]),
    (0, 1): np.array([3.0]),
    (2, 0): np.array([4.0])
}
mtf_real = MultivariateTaylorFunction(coefficients_real, dimension=2, var_list=[x, y])
print("Real MTF:")
mtf_real.print_tabular()

# Evaluate MTF at a point
evaluation_point = np.array([0.1, 0.2])
result_real = mtf_real(evaluation_point)
print(f"Evaluation at {evaluation_point}: {result_real}")

# Create a Complex Multivariate Taylor Function
coefficients_complex = {
    (0, 0): np.array([1.0 + 1j]),
    (1, 0): np.array([2.0 - 1j]),
    (0, 1): np.array([3.0 + 0j]),
}
mtf_complex = ComplexMultivariateTaylorFunction(coefficients_complex, dimension=2, var_list=[x, y])
print("\nComplex MTF:")
mtf_complex.print_tabular()

# Perform operations (example: addition)
mtf_real_2 = MultivariateTaylorFunction.from_constant(2.0, dimension=2)
mtf_sum = mtf_real + mtf_real_2
print("\nSum of MTFs:")
mtf_sum.print_tabular()


# For more examples, refer to the demo scripts and unit tests.