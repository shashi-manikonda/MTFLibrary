.. _guide_composition:

Function Composition
====================

Function composition is one of the most powerful features of `mtflib`. It
allows you to build complex functions by substituting one or more variables
of a function with other `MultivariateTaylorFunction` objects. This is
achieved using the `compose` method.

The `compose` method takes a dictionary as input, where the keys are the
1-based variable indices to be substituted, and the values are the MTF
objects to substitute in their place.

Basic Composition
-----------------

Here is an example of composing two functions, `f(g(x, y))`.

.. code-block:: python

   from mtflib import mtf

   # Initialize the library
   mtf.initialize_mtf(max_order=3, max_dimension=2)

   # Create variables
   x = mtf.var(1)
   y = mtf.var(2)

   # Define an outer function f(a) = exp(a)
   # Note: We can define f in terms of a single variable 'a'
   a = mtf.var(1, dimension=1) # Create a 1D variable
   f = mtf.exp(a)

   # Define an inner function g(x, y) = x + y
   g = x + y

   # Compose f with g: h(x, y) = f(g(x, y)) = exp(x + y)
   # We substitute the first variable of f (a) with g
   h = f.compose({1: g})

   print("Composed function h(x, y) = exp(x + y):")
   print(h.get_tabular_dataframe())

Composition with Multiple Variables
-----------------------------------

You can also compose functions with multiple variables. For example, if you
have a function `f(a, b)`, you can substitute `a` with `g1(x, y)` and `b`
with `g2(x, y)`.

.. code-block:: python

   from mtflib import mtf

   # Initialize the library
   mtf.initialize_mtf(max_order=2, max_dimension=2)

   # Create variables for the outer function f(a, b)
   a = mtf.var(1)
   b = mtf.var(2)
   f = a * b
   print("Outer function f(a, b) = a * b")

   # Create variables for the inner functions
   x = mtf.var(1)
   y = mtf.var(2)

   # Define inner functions
   g1 = x + y
   g2 = x - y
   print("Inner functions: g1(x,y) = x+y, g2(x,y) = x-y")

   # Compose f with g1 and g2: h(x, y) = f(g1, g2) = (x+y)*(x-y) = x^2 - y^2
   h = f.compose({1: g1, 2: g2})

   print("\\nComposed function h(x, y):")
   print(h.get_tabular_dataframe())
