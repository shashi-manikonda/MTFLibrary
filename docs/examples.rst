.. _examples:

Examples
========

This page provides a gallery of examples to showcase the key
functionalities of `mtflib`.

Example 1: Function Composition
-------------------------------

This example demonstrates how to create complex functions by composing
simpler ones. We will create the function `f(x, y) = sin(x + y)`.

.. code-block:: python

   from mtflib import mtf

   # Initialize the library
   mtf.initialize_mtf(max_order=4, max_dimension=2)

   # Create variables
   x = mtf.var(1)
   y = mtf.var(2)

   # Define the inner function g(x, y) = x + y
   g = x + y

   # Compose with sin to get f(x, y) = sin(x + y)
   f = mtf.sin(g)

   print("Taylor series for f(x, y) = sin(x + y):")
   print(f.get_tabular_dataframe())

Example 2: Inverting a Coordinate Transformation with TaylorMap
---------------------------------------------------------------

The `TaylorMap` class is ideal for representing coordinate transformations.
This example shows how to define a map and compute its inverse.

.. code-block:: python

   from mtflib import mtf, TaylorMap

   # Initialize the library
   mtf.initialize_mtf(max_order=3, max_dimension=2)

   # Create variables for the original coordinate system
   x = mtf.var(1)
   y = mtf.var(2)

   # Define a transformation F(x, y) = [u, v] where:
   # u = x + y^2
   # v = y - x^2
   F = TaylorMap([x + y**2, y - x**2])

   print("Original Map F(x, y):")
   print(F)

   # Compute the inverse map G = F_inv, such that G(u, v) = [x, y]
   G = F.invert()

   print("\\nInverse Map G(u, v):")
   print(G)

   # Verification: F(G(u,v)) should be the identity map [u, v]
   # Let's use new variables u, v for clarity
   u = mtf.var(1)
   v = mtf.var(2)
   Identity_check = F.compose(G)

   print("\\nVerification F(G(u,v)):")
   print(Identity_check)


Example 3: Differentiation and Integration
------------------------------------------

This example demonstrates how to use the core differential algebra
operators: `derivative` and `integrate`.

.. code-block:: python

   from mtflib import mtf

   # Initialize the library
   mtf.initialize_mtf(max_order=5, max_dimension=2)

   # Create variables
   x = mtf.var(1)
   y = mtf.var(2)

   # Define a function f(x, y) = x^3 * y^2
   f = (x**3) * (y**2)
   print("Original function f(x, y) = x^3 * y^2:")
   print(f.get_tabular_dataframe())

   # --- Differentiation ---
   # Compute the partial derivative with respect to x
   df_dx = f.derivative(deriv_dim=1)
   print("\\nPartial derivative df/dx:")
   print(df_dx.get_tabular_dataframe()) # Expected: 3 * x^2 * y^2

   # --- Integration ---
   # Compute the indefinite integral of df/dx with respect to x
   # This should recover the original function (up to a constant of integration)
   f_recovered = df_dx.integrate(integration_variable_index=1)
   print("\\nIntegral of df/dx w.r.t x:")
   print(f_recovered.get_tabular_dataframe())
