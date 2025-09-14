.. _guide_creating_mtfs:

Creating a `MultivariateTaylorFunction`
=======================================

There are two primary ways to create a ``MultivariateTaylorFunction`` (MTF)
object in `mtflib`. You can either build it from elementary functions and
arithmetic operations, or construct it directly from its coefficients and
exponents.

Method 1: From Elementary Functions
-----------------------------------

This is the most common and intuitive way to create an MTF. You start by
defining the variables of your function and then combine them using standard
Python arithmetic and `mtflib`'s elementary functions.

.. code-block:: python

   from mtflib import mtf

   # Initialize the library for 2 variables up to order 3
   mtf.initialize_mtf(max_order=3, max_dimension=2)

   # Create variables x and y
   x = mtf.var(1)
   y = mtf.var(2)

   # Define a function using arithmetic and elementary functions
   # f(x, y) = exp(x) + 2*y
   f = mtf.exp(x) + 2*y

   print("MTF created from elementary functions:")
   print(f.get_tabular_dataframe())

The library automatically computes the Taylor series coefficients for the
resulting function `f`.

Method 2: From Coefficients and Exponents
-----------------------------------------

For more advanced use cases, you may need to create an MTF directly from
a known set of Taylor coefficients. This can be done by providing a tuple
containing two NumPy arrays: one for the exponents and one for the
coefficients.

.. code-block:: python

   import numpy as np
   from mtflib import mtf

   # Initialize the library
   if not mtf.get_mtf_initialized_status():
       mtf.initialize_mtf(max_order=3, max_dimension=2)

   # Define the exponents and coefficients for f(x,y) = 1 + 2x + 3y^2
   exponents = np.array([
       [0, 0],  # Constant term
       [1, 0],  # x term
       [0, 2]   # y^2 term
   ])
   coeffs = np.array([1.0, 2.0, 3.0])

   # Create the MTF directly
   g = mtf.MultivariateTaylorFunction((exponents, coeffs))

   print("MTF created from coefficients:")
   print(g.get_tabular_dataframe())

This method gives you precise control over the structure of the Taylor
series and is particularly useful when interfacing with other systems or
when you have pre-computed coefficients.
