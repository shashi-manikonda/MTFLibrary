# mtflib: Multivariate Taylor Function Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python library for creating, manipulating, and composing Multivariate Taylor Functions, with extensions for electromagnetism calculations.**

## Installation

### Prerequisites:

  - **Python:** Requires Python 3.7 or later.

  - **NumPy:** NumPy is essential for numerical computations. Install using:

    ```bash
    pip install numpy
    ```

  - **mpi4py (Optional):** For MPI-parallel Biot-Savart calculations, install `mpi4py`:

    ```bash
    pip install mpi4py
    ```

    MPI environment setup is also required to utilize MPI features.

### Installation of mtflib:

To install `mtflib` from the package directory, use pip:

```bash
pip install .
```

## A Simple Usage Example

Here's a quick example to get you started with `mtflib`:

```python
import numpy as np
from mtflib import initialize_mtf_globals, Var, sin_taylor

# 1. Initialize global settings (must be done once)
initialize_mtf_globals(max_order=5, max_dimension=2)

# 2. Define symbolic variables
x = Var(1)
y = Var(2)

# 3. Perform an operation
# Create a Taylor series for sin(x) + y^2
f = sin_taylor(x) + y**2

# 4. Evaluate the result at a point
# Let's evaluate f at (x=0.5, y=2.0)
eval_point = np.array([0.5, 2.0])
result = f.eval(eval_point)

print(f"f(x, y) = sin(x) + y^2")
print(f"Result of f(0.5, 2.0): {result[0]}")

# For comparison, the exact value is sin(0.5) + 2.0^2
exact_value = np.sin(0.5) + 4.0
print(f"Exact value: {exact_value}")
```

## Overview

`mtflib` is a versatile Python library designed to empower symbolic and numerical computations using Multivariate Taylor Functions (MTFs). It offers a comprehensive suite of classes and methods for:

  - **Representation:** Creating and managing multivariate Taylor functions with real and complex coefficients, up to a user-defined order and dimension.
  - **Arithmetic Operations:** Performing essential algebraic operations on MTFs, including addition, subtraction, multiplication, division, scalar operations, negation, and power (with scalar exponents).
  - **Calculus Operations:** Computing partial derivatives and integration of MTFs using dedicated `derivative` and `integrate` functions.
  - **Composition and Substitution:** Implementing MTF composition through variable substitution, effectively realizing the multivariate chain rule.
  - **Evaluation:** Efficiently evaluating MTFs at specific numerical points.
  - **Truncation:** Reducing the order of MTFs to a desired level.
  - **Elementary Functions:** Providing a comprehensive library of elementary mathematical functions (e.g., cosine, sine, exponential, logarithm, etc.) implemented as MTFs, enabling seamless integration with MTF operations.
  - **Visualization and Output:** Representing MTFs as formatted tables for easy readability and inspection.
  - **NumPy Compatibility:** Seamless integration with NumPy arrays and ufuncs through the `__array_ufunc__` protocol, allowing element-wise operations with NumPy functions.
  - **Electromagnetism Extension:** Extending `mtflib`'s capabilities to electromagnetic field computations, including Biot-Savart law calculations for current loops and segments.

### Key Applications:

`mtflib` is particularly well-suited for applications in diverse fields such as:

  - **Sensitivity Analysis:** Analyzing the sensitivity of system outputs to variations in input parameters.
  - **Uncertainty Quantification:** Propagating uncertainties through complex models using Taylor series representations.
  - **Automatic Differentiation:** Implementing automatic differentiation techniques for gradient and higher-order derivative calculations.
  - **Nonlinear System Analysis:** Studying and simulating nonlinear systems using Taylor series approximations.
  - **Electromagnetics Modeling:** Calculating magnetic fields from current distributions using the Biot-Savart law within the MTF framework.

## Features

  - **Multivariate Taylor Functions (MTFs):**
      - Robustly represents MTFs with real and complex coefficients.
      - Supports arbitrary order and dimension (limited by global settings).
      - Efficient storage and manipulation of Taylor coefficients using NumPy arrays.
  - **Comprehensive Mathematical Operations:**
      - **Arithmetic:** Addition, subtraction, multiplication, division, negation.
      - **Scalar Operations:** Multiplication, division, and power operations with scalars.
      - **Calculus:** Partial differentiation with respect to any variable, integration using `derivative` and `integrate` functions from `elementary_functions`.
      - **Composition:** Variable substitution and MTF composition.
      - **Map Inversion:** The `TaylorMap` class now includes a powerful `invert()` method. It computes the inverse of a square map using a fixed-point iteration algorithm, provided the map has no constant terms and its linear part is invertible.
      - **Truncation:** Reducing MTF order.
      - **Evaluation:** Numerical evaluation of MTFs at given points.
      - **NumPy ufunc Support:**  Integration with NumPy's universal functions (ufuncs) for element-wise operations (e.g., `np.sin`, `np.cos`, `np.exp`, `np.sqrt`, `np.add`, `np.multiply`, etc.) on MTF objects and arrays of MTFs.
  - **Extensive Elementary Function Library:**
      - Includes Taylor series implementations of a wide range of elementary functions: `cos`, `sin`, `tan`, `exp`, `log`, `sqrt`, `arctan`, `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctanh`, and `gaussian`.
      - Enables the creation of MTFs representing these functions for seamless integration into MTF expressions and computations.
  - **Electromagnetism Extension:**
      - Provides specialized functionalities for electromagnetic field computations.
      - Implements Biot-Savart law calculations for magnetic fields generated by current distributions, using `serial_biot_savart` and `mpi_biot_savart` for serial and parallel computations.
      - Includes `current_ring` function to generate MTF representations of current rings, discretized into segments.
  - **3D Field Visualization:**
      - A new `plotting.py` module provides powerful tools to visualize magnetic fields and their source geometries.
      - **`Coil` Class:** A simple container to define current-carrying geometries.
      - **`plot_field_on_line`:** Generates a combined 3D plot of the coil geometry and a 2D plot of a B-field component along a specified line.
      - **`plot_field_on_plane`:** Creates a 3D visualization of the B-field (using quiver, contour, or stream plots) on a specified 2D plane.
      - **`plot_field_vectors_3d`:** Renders a 3D quiver plot of the B-field at specified points in space.
      - See the `demos/em/Field_Plotting_Demo.ipynb` for detailed examples.
  - **Modularity and Extensibility:**
      - Well-structured codebase for easy maintenance and future extensions.
      - Designed to be modular, allowing for the addition of new functionalities and elementary functions.
  - **User-Friendly API:**
      - Clear and intuitive function and class names.
      - Consistent and well-documented API.
      - Example scripts and unit tests provided to demonstrate usage.
  - **Efficiency and Performance:**
      - Leverages NumPy for optimized numerical operations and efficient storage of Taylor coefficients.
      - **Global Coefficient Cleanup Control:** `mtflib` now automatically removes negligible coefficients after arithmetic operations to improve performance and keep Taylor series representations concise. This feature is enabled by default and can be controlled via the `set_truncate_after_operation(enable: bool)` function.
      - MPI-parallel implementation of Biot-Savart law for large-scale electromagnetic computations, utilizing `mpi_biot_savart`.
  - **Testability and Reliability:**
      - Includes a comprehensive suite of unit tests to ensure the correctness and reliability of MTF operations and elementary functions.

## Demo Files
To help you get started with `mtflib`, we have included several demo files that illustrate how to use the library for various applications. These demo files cover a range of topics, including:

- Basic usage of multivariate Taylor functions
- Performing arithmetic operations on MTFs
- Computing derivatives and integrals of MTFs
- Evaluating MTFs at specific points
- Visualizing and exporting MTFs
- Using the electromagnetism extension for field computations

You can find the demo files in the `demos` directory of this repository.