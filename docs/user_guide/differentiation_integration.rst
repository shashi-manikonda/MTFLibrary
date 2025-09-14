.. _guide_diff_and_int:

Differentiation and Integration
===============================

`mtflib` provides straightforward methods for performing partial
differentiation and integration on `MultivariateTaylorFunction` objects. These
operations are fundamental to the Differential Algebra framework.

Differentiation
---------------

The `derivative` method computes the partial derivative of an MTF with
respect to a specified variable. The variable is identified by its 1-based
index, passed to the `deriv_dim` argument.

.. code-block:: python

   from mtflib import mtf

   # Initialize the library
   mtf.initialize_mtf(max_order=4, max_dimension=2)

   # Create variables
   x = mtf.var(1)
   y = mtf.var(2)

   # Define a function f(x, y) = x^3 * y^2
   f = (x**3) * (y**2)
   print("Original function f(x, y):")
   print(f.get_tabular_dataframe())

   # Compute the partial derivative with respect to x (deriv_dim=1)
   df_dx = f.derivative(deriv_dim=1)
   print("\\nPartial derivative df/dx:")
   print(df_dx.get_tabular_dataframe()) # Expected: 3 * x^2 * y^2

   # Compute the partial derivative with respect to y (deriv_dim=2)
   df_dy = f.derivative(deriv_dim=2)
   print("\\nPartial derivative df/dy:")
   print(df_dy.get_tabular_dataframe()) # Expected: 2 * x^3 * y


Integration
-----------

The `integrate` method computes the indefinite integral of an MTF with
respect to a specified variable. Similar to differentiation, the variable
is identified by its 1-based index.

.. code-block:: python

   from mtflib import mtf

   # Initialize the library
   if not mtf.get_mtf_initialized_status():
       mtf.initialize_mtf(max_order=5, max_dimension=2)

   # Create a function g(x, y) = 3 * x^2 * y^2
   x = mtf.var(1)
   y = mtf.var(2)
   g = 3 * (x**2) * (y**2)
   print("Original function g(x, y):")
   print(g.get_tabular_dataframe())

   # Compute the indefinite integral with respect to x
   integral_g_dx = g.integrate(integration_variable_index=1)
   print("\\nIntegral of g(x, y) w.r.t x:")
   print(integral_g_dx.get_tabular_dataframe()) # Expected: x^3 * y^2

.. note::
   The `integrate` method computes an indefinite integral, and the constant
   of integration is assumed to be zero. If you need to handle a non-zero
   constant of integration, you must add it manually as a constant MTF.
