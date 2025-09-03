.. _api_reference:

API Reference
=============

This page provides a detailed reference for the `mtflib` public API. The
documentation is automatically generated from the docstrings in the source
code.

Core Classes
------------

These are the main classes for representing and manipulating Taylor series.

.. autoclass:: mtflib.taylor_function.MultivariateTaylorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mtflib.complex_taylor_function.ComplexMultivariateTaylorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mtflib.taylor_map.TaylorMap
   :members:
   :undoc-members:
   :show-inheritance:

Factory and Utility Functions
-----------------------------

These functions are used to create or manipulate core objects.

.. automodule:: mtflib.taylor_function
   :members: Var, mtfarray

.. automodule:: mtflib.elementary_coefficients
   :members: load_precomputed_coefficients

Elementary Functions & Operators
--------------------------------

These functions provide the core mathematical operations of the Differential
Algebra.

.. automodule:: mtflib.elementary_functions
   :members:
   :undoc-members:

Backend System
--------------

These classes and functions manage the computational backend for `mtflib`.

.. automodule:: mtflib.backend
   :members: get_backend, NumpyBackend, TorchBackend
   :undoc-members:
   :show-inheritance:
