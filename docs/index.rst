.. mtflib documentation master file.

.. include:: getting_started.rst
   :start-after: .. start-getting-started
   :end-before: .. end-getting-started

.. include:: advanced.rst
   :start-after: .. start-advanced
   :end-before: .. end-advanced

Welcome to mtflib's documentation!
==================================

`mtflib` is a Python library for creating, manipulating, and composing
Multivariate Taylor Functions, with a C++ backend for performance-critical
applications.

Features
--------

- **High Performance:** `mtflib` utilizes a C++ backend for key operations, ensuring fast, efficient computation.
- **Backend Flexibility:** Supports both NumPy and PyTorch, automatically switching backends to leverage GPU acceleration when PyTorch tensors are used.
- **Comprehensive Functionality:** Includes a wide range of elementary functions and core operations like composition, differentiation, and integration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   getting_started
   advanced

.. toctree::
   :maxdepth: 1
   :caption: Indices and tables:

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
