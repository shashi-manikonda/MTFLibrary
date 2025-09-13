# mtflib: Multivariate Taylor Function Library

[![Documentation Status](https://readthedocs.org/projects/mtflibrary/badge/?version=latest)](https://mtflibrary.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for creating, manipulating, and composing Multivariate Taylor Functions, with a C++ backend for performance-critical applications.

## Installation

The recommended way to install `mtflib` is from PyPI:

```bash
pip install mtflib
```

### Installation from Source

Alternatively, you can install `mtflib` directly from the source repository using pip. Ensure you have a C++17 compliant compiler (e.g., GCC, Clang, MSVC) for building the backend extensions.

```bash
pip install .
```

## Quick Start

Here's a simple example to get you started with `mtflib`:

```python
import numpy as np
from mtflib import mtf, Var

# 1. Initialize global settings (must be done once)
# This sets the max order of the Taylor series and the number of variables.
mtf.initialize_mtf(max_order=5, max_dimension=2)

# 2. Define symbolic variables
# Var(1) corresponds to x, Var(2) to y
x = Var(1)
y = Var(2)

# 3. Create a Taylor series expression
# This creates a Taylor series for sin(x) + y^2
f = mtf.sin(x) + y**2

# 4. Evaluate the result at a point
# Let's evaluate f at (x=0.5, y=2.0)
eval_point = np.array([0.5, 2.0])
result = f.eval(eval_point)

print(f"f(x, y) = sin(x) + y^2")
print(f"Result of f(0.5, 2.0): {result[0]}")

# For comparison, the exact value is sin(0.5) + 2.0^2
exact_value = np.sin(0.5) + 4.0
print(f"Exact value: {exact_value}")

# You can also view the Taylor series coefficients
print("\\nTaylor Series Representation:")
print(f)
```

## Running Tests

The project uses `pytest` for testing. First, install the test dependencies:

```bash
pip install -e .[test]
```

Then, run the test suite from the root of the repository:

```bash
pytest
```
