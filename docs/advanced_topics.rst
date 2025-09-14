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

.. code-block:: python
   :caption: Example of using the PyTorch backend for GPU acceleration

    import numpy as np
    import torch
    from mtflib import mtf

    # Initialize the library
    mtf.initialize_mtf(max_order=5, max_dimension=2)

    # Create a function
    x, y = mtf.var(1), mtf.var(2)
    f = mtf.exp(x * y)

    # Create a large batch of evaluation points
    num_points = 1_000_000
    points_np = np.random.rand(num_points, 2)

    # --- NumPy Backend (CPU) ---
    # The backend is automatically selected based on the input type
    result_np = f.neval(points_np)
    print(f"NumPy backend result type: {type(result_np)}")


    # --- PyTorch Backend (GPU) ---
    # Check if a CUDA-enabled GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Convert the points to a PyTorch tensor on the GPU
        points_torch = torch.from_numpy(points_np).to(device)

        # The PyTorch backend is automatically selected
        result_torch = f.neval(points_torch)

        print(f"PyTorch backend result type: {type(result_torch)}")
        print(f"PyTorch tensor device: {result_torch.device}")

.. note::
    To leverage the PyTorch backend, you must have a CUDA-compatible GPU
    and the `torch` library installed. You can install it with `mtflib`
    using `pip install mtflib[torch]`.

Future Work
-----------

This section outlines potential future directions for `mtflib`, ranging from
incremental improvements to more ambitious features.

### Performance Benchmarking and Optimization

* **Comprehensive Benchmarking:** Create a dedicated benchmark suite to
  compare the performance of the Python, C++, and PyTorch backends for
  various operations and problem sizes.
* **C++ Backend Profiling:** Profile the C++ backend to identify and
  optimize performance bottlenecks.
* **Advanced Data Structures:** Investigate using more advanced data
  structures for storing coefficients, such as hash maps or sparse matrices,
  to improve performance for very sparse Taylor series.

### Expanded Functionality

* **Automatic Differentiation (AD):** Implement support for AD to allow for
  the computation of gradients, Jacobians, and Hessians of Taylor maps.
* **Advanced Elementary Functions:** Add support for more advanced
  elementary functions, such as Bessel functions, gamma functions, and error
  functions.
* **Non-linear Equation Solver:** Implement a solver for systems of
  non-linear equations using Taylor series methods (e.g., Newton's method
  with high-order corrections).

### Improved User Experience

* **Symbolic API:** Develop a more intuitive API for creating and
  manipulating Taylor maps, perhaps with a more "symbolic" feel.
* **Example Gallery:** Create a "gallery" of examples in the documentation
  showcasing real-world applications of `mtflib` in physics, engineering,
  and other fields.
* **Improved Error Messages:** Add more detailed error messages and warnings
  to help users debug their code.
