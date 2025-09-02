.. start-getting-started

Getting Started
===============

.. end-getting-started

Installation
------------

First, ensure you have Python 3.8+ installed.

### Basic Installation

You can install `mtflib` and its core dependencies using pip:

.. code-block:: bash

    pip install -r requirements.txt

### Optional: Installation with PyTorch

For GPU acceleration and improved performance, it is highly recommended to install PyTorch. The `neval` function will automatically detect and use the PyTorch backend if it is available.

.. code-block:: bash

    pip install torch

Quick-Start Example
-------------------

Here is a simple example demonstrating how to create and evaluate a
`MultivariateTaylorFunction`.

.. code-block:: python

    import numpy as np
    from mtflib.taylor_function import MultivariateTaylorFunction

    # Define a 2D Taylor function: f(x, y) = 1 + 2x + 3y + 4x^2 + 5xy + 6y^2
    dimension = 2
    coeffs = np.array([1, 2, 3, 4, 5, 6])
    exponents = np.array([
        [0, 0], # Order 0
        [1, 0], # Order 1
        [0, 1],
        [2, 0], # Order 2
        [1, 1],
        [0, 2],
    ])

    f = MultivariateTaylorFunction(dimension, coeffs, exponents)

    # Evaluate at a single point (1, 2)
    single_point = np.array([1.0, 2.0])
    result = f.eval(single_point)
    print(f"Value at single point {single_point}: {result}")

    # Evaluate on a batch of points
    batch_points = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [0.5, 0.5],
    ])
    results_batch = f.neval(batch_points)
    print(f"Values at batch points:\n{results_batch}")
