.. _theory:

Advanced Topics
======================

Backends and Performance
------------------------

`mtflib` is designed for high performance by abstracting key numerical
operations to a flexible backend system. This allows the library to leverage
optimized libraries like NumPy and PyTorch.

### The Backend System

The `backend.py` module defines a set of static methods for common
tensor operations (e.g., `power`, `prod`, `dot`). The `neval`
method in `MultivariateTaylorFunction` automatically selects the appropriate
backend based on the input type.

### NumPy Backend

This is the default backend and is used when the input to `neval` is a
NumPy array. All operations are performed using NumPy's highly optimized
C and Fortran libraries, providing excellent performance for CPU-based computation.

### PyTorch Backend

When a PyTorch `Tensor` is passed to `neval`, `mtflib` automatically
switches to the PyTorch backend. This provides two key advantages:

* **GPU Acceleration:** PyTorch's native CUDA support allows operations to be
    executed on an NVIDIA GPU, dramatically speeding up computations on large
    batches of data.
* **Vector and Matrix Operations:** PyTorch's backend is specifically
    optimized for deep learning, making it exceptionally fast for the types of
    vectorized operations used in `mtflib`.

---
