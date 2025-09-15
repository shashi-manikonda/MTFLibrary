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

.. toctree::
   :maxdepth: 1

   ../demos/quick_start_demo

This example demonstrates the basic workflow of defining a function and
evaluating it. For more complex examples, see the :ref:`examples` page.
