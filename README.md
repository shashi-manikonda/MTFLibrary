# MTFLibrary: Multivariate Taylor Function Library

[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python library for creating, manipulating, and composing
Multivariate Taylor Functions, with extensions for electromagnetism
calculations.**

## Overview

MTFLibrary is a versatile Python library designed to empower symbolic
and numerical computations using Multivariate Taylor Functions (MTFs).
It offers a comprehensive suite of classes and methods for:

-   **Representation:** Creating and managing multivariate Taylor
    functions with real and complex coefficients, up to a user-defined
    order and dimension.
-   **Arithmetic Operations:** Performing essential algebraic operations
    on MTFs, including addition, subtraction, multiplication, division,
    scalar operations, and more.
-   **Calculus Operations:** Computing partial derivatives and
    integration of MTFs.
-   **Composition and Substitution:** Implementing MTF composition
    through variable substitution, effectively realizing the
    multivariate chain rule.
-   **Evaluation:** Efficiently evaluating MTFs at specific numerical
    points.
-   **Truncation:** Reducing the order of MTFs to a desired level.
-   **Elementary Functions:** Providing a library of elementary
    mathematical functions (e.g., cosine, sine, exponential, logarithm,
    etc.) implemented as MTFs, enabling seamless integration with MTF
    operations.
-   **Visualization and Output:** Representing MTFs as formatted tables
    for easy readability and inspection.
-   **Electromagnetism Extension (EMLibrary):** Extending MTFLibrary\'s
    capabilities to electromagnetic field computations, including
    Biot-Savart law calculations for current loops and segments.

### Key Applications:

MTFLibrary is particularly well-suited for applications in diverse
fields such as:

-   **Sensitivity Analysis:** Analyzing the sensitivity of system
    outputs to variations in input parameters.
-   **Uncertainty Quantification:** Propagating uncertainties through
    complex models using Taylor series representations.
-   **Automatic Differentiation:** Implementing automatic
    differentiation techniques for gradient and higher-order derivative
    calculations.
-   **Nonlinear System Analysis:** Studying and simulating nonlinear
    systems using Taylor series approximations.
-   **Electromagnetics Modeling:** Calculating magnetic fields from
    current distributions using the Biot-Savart law within the MTF
    framework.

## Features

-   **Multivariate Taylor Functions (MTFs):**
    -   Robustly represents MTFs with real and complex coefficients.
    -   Supports arbitrary order and dimension (limited by global
        settings).
    -   Efficient storage and manipulation of Taylor coefficients using
        NumPy arrays.
-   **Comprehensive Mathematical Operations:**
    -   **Arithmetic:** Addition, subtraction, multiplication, division,
        negation.
    -   **Scalar Operations:** Multiplication and division by scalars.
    -   **Calculus:** Partial differentiation with respect to any
        variable, integration.
    -   **Composition:** Variable substitution and MTF composition.
    -   **Truncation:** Reducing MTF order.
    -   **Evaluation:** Numerical evaluation of MTFs at given points.
    -   **Inverse:** Calculation of the inverse of an MTF.
-   **Elementary Function Library:**
    -   Includes Taylor series implementations of common elementary
        functions: `cos`, `sin`, `exp`, `log`, `sqrt`, `arctan`, `sinh`,
        `cosh`, `tanh`, `arcsin`, `arccos`, `arctanh`, and `gaussian`.
    -   Allows for the creation of MTFs representing these functions and
        their seamless integration into MTF expressions.
-   **EMLibrary Extension (Electromagnetism):**
    -   Provides functionalities for electromagnetic field computations.
    -   Implements Biot-Savart law calculations for magnetic fields
        generated by current distributions.
    -   Includes functions for:
        -   Calculating magnetic fields from current segments using
            efficient serial and MPI-parallel implementations.
        -   Generating MTF representations of current rings, discretized
            into segments, with control over ring radius, number of
            segments, center point, and axis direction.
-   **Modularity and Extensibility:**
    -   Well-structured codebase for easy maintenance and future
        extensions.
    -   Designed to be modular, allowing for the addition of new
        functionalities and elementary functions.
-   **User-Friendly API:**
    -   Clear and intuitive function and class names.
    -   Consistent and well-documented API.
    -   Example scripts and unit tests provided to demonstrate usage.
-   **Efficiency and Performance:**
    -   Leverages NumPy for optimized numerical operations and efficient
        storage of Taylor coefficients.
    -   MPI-parallel implementation of Biot-Savart law for large-scale
        electromagnetic computations.
-   **Testability and Reliability:**
    -   Includes a comprehensive suite of unit tests to ensure the
        correctness and reliability of MTF operations and elementary
        functions.

## Installation

### Prerequisites:

-   **Python:** Requires Python 3.7 or later.

-   **NumPy:** NumPy is essential for numerical computations. Install
    using:

        pip install numpy

-   **mpi4py (Optional):** For MPI-parallel Biot-Savart calculations,
    install `mpi4py`:

        pip install mpi4py

    MPI environment setup is also required to utilize MPI features.

### Installation of MTFLibrary:

To install MTFLibrary from the package directory, use pip:

    pip install .

For development purposes and to install in editable mode (allowing
changes to the library to be immediately reflected without
reinstalling):

    pip install -e .

## Usage

### Basic MTF Operations:

``` {style="display:block; white-space:pre;"}
from MTFLibrary.taylor_function import initialize_mtf_globals, set_global_etol, get_global_max_order, get_global_max_dimension
from MTFLibrary.taylor_function import Var, MultivariateTaylorFunction, ComplexMultivariateTaylorFunction
import numpy as np

# Initialize global settings for MTFLibrary
initialize_mtf_globals(max_order=10, max_dimension=2)  # Set max order and dimension
set_global_etol(1e-10) # Set error tolerance for coefficient truncation

print(f"Max Order: {get_global_max_order()}") # Get and print max order
print(f"Max Dimension: {get_global_max_dimension()}") # Get and print max dimension

# Create symbolic variables
x = Var(1) # Variable 'x' for dimension 1
y = Var(2) # Variable 'y' for dimension 2

# Create a Real Multivariate Taylor Function
coefficients_real = { # Define coefficients as a dictionary
    (0, 0): np.array([1.0]), # Coefficient for order (0,0)
    (1, 0): np.array([2.0]), # Coefficient for order (1,0)
    (0, 1): np.array([3.0]), # Coefficient for order (0,1)
    (2, 0): np.array([4.0])  # Coefficient for order (2,0)
}
mtf_real = MultivariateTaylorFunction(coefficients_real, dimension=2, var_list=[x, y]) # Create MTF object
print("Real MTF:")
mtf_real.print_tabular() # Print MTF in table format

# Evaluate MTF at a point
evaluation_point = np.array([0.1, 0.2]) # Define evaluation point
result_real = mtf_real(evaluation_point) # Evaluate MTF
print(f"Evaluation at {evaluation_point}: {result_real}") # Print evaluation result

# Create a Complex Multivariate Taylor Function
coefficients_complex = { # Define complex coefficients
    (0, 0): np.array([1.0 + 1j]), # Complex coefficient for order (0,0)
    (1, 0): np.array([2.0 - 1j]), # Complex coefficient for order (1,0)
    (0, 1): np.array([3.0 + 0j]), # Complex coefficient for order (0,1)
}
mtf_complex = ComplexMultivariateTaylorFunction(coefficients_complex, dimension=2, var_list=[x, y]) # Create complex MTF
print("\nComplex MTF:")
mtf_complex.print_tabular() # Print complex MTF

# Perform MTF operations (example: addition)
mtf_real_2 = MultivariateTaylorFunction.from_constant(2.0, dimension=2) # Create constant MTF
mtf_sum = mtf_real + mtf_real_2 # Add two MTFs
print("\nSum of MTFs:")
mtf_sum.print_tabular() # Print the sum MTF
    
```

### EMLibrary Usage (Biot-Savart Example):

``` {style="display:block; white-space:pre;"}
import numpy as np
from MTFLibrary.taylor_function import initialize_mtf_globals, set_global_etol, Var
from MTFLibrary.EMLibrary.current_ring import current_ring
from MTFLibrary.EMLibrary.biot_savart import serial_biot_savart # or mpi_biot_savart for parallel

# Initialize MTF globals
initialize_mtf_globals(max_order=4, max_dimension=4)
set_global_etol(1e-16)

# Define MTF variables
x = Var(1)
y = Var(2)
z = Var(3)
u = Var(4)

# Define current ring parameters
ring_radius = 1.0
num_segments_ring = 5
ring_center_point = np.array([0.5, 0.5, 0.0])
ring_axis_direction = np.array([1, 1, 1])

# Generate ring segment MTFs
segment_mtfs_ring, element_lengths_ring, direction_vectors_ring = current_ring(
    ring_radius, num_segments_ring, ring_center_point, ring_axis_direction)

# Define field points along z-axis as MTFs
num_field_points_axis = 50
z_axis_coords = np.linspace(-2, 2, num_field_points_axis)
field_points_axis = np.array([[x, y, zc+z] for zc in z_axis_coords])

# Calculate Biot-Savart field (serial version)
B_field_ring_axis = serial_biot_savart(segment_mtfs_ring, element_lengths_ring, direction_vectors_ring, field_points_axis)

print("Magnetic field along axis of rotated ring (Example 1 - Element Input, first point):")
print(B_field_ring_axis[0][0]) # Print B-field at the first point
    
```

## Explore More:

For more detailed examples and functionalities, please refer to the
`demo` directory within the MTFLibrary package. The `test` directory
contains comprehensive unit tests that demonstrate various features and
ensure the library\'s correctness.

## License

MTFLibrary is distributed under the MIT License. See the `LICENSE` file
for details.
