.. _getting_started:

Getting Started
===============

This guide will walk you through installing `mtflib` and running your
first example.

Installation
------------

You can install `mtflib` using `pip`. There are two installation options
depending on your needs.

Basic Installation
~~~~~~~~~~~~~~~~~~

For standard usage with a NumPy backend, you can install the library
directly from PyPI:

.. code-block:: bash

   pip install mtflib

Optional: PyTorch Backend for GPU Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a CUDA-enabled GPU and want to leverage it for significant
performance improvements, you can install `mtflib` with the optional
PyTorch dependency:

.. code-block:: bash

   pip install mtflib[torch]

This will install the necessary PyTorch libraries alongside `mtflib`,
enabling the GPU-accelerated backend for `neval`.

A Quick-Start Example
---------------------

Here is a simple example to get you started. This script initializes the
library, creates a two-variable function, and evaluates it at a point.

.. code-block:: python

    from mtflib import MultivariateTaylorFunction, Var, sin_taylor

    # 1. Initialize the library's global settings. This is a crucial first step.
    # We'll set a maximum order of 4 and 2 variables (dimensions).
    MultivariateTaylorFunction.initialize_mtf(max_order=4, max_dimension=2)

    # 2. Create variables x and y.
    # Var(1) corresponds to the first variable, Var(2) to the second.
    x = Var(1)
    y = Var(2)

    # 3. Define a function, for example, f(x, y) = sin(x + y**2)
    f = sin_taylor(x + y**2)

    # 4. Print the function's Taylor series coefficients in a readable format.
    print("Taylor series for f(x, y) = sin(x + y^2):")
    print(f.get_tabular_dataframe())

    # 5. Evaluate the function at the point (x=2, y=3).
    evaluation_point = [2, 3]
    result = f.eval(evaluation_point)
    print(f"\\nResult of f(2, 3): {result[0]}")

This example demonstrates the basic workflow of defining a function and
evaluating it. For more complex examples, see the :ref:`examples` page.
