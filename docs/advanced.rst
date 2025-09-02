.. start-advanced

Advanced Topics
===============

.. end-advanced

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

Theoretical Background: The Differential Algebra :math:`_{n}D_{v}`
--------------------------------------------------------------------------------

For a real analytic function :math:`f` in :math:`v` variables, we can form a vector that
contains all its Taylor expansion coefficients at :math:`\vec{x}=\vec{0}` up to a
certain order :math:`n`. This vector is called the DA (Differential Algebra) vector.
Knowing the DA vector for two real analytic functions :math:`f` and :math:`g`
allows us to compute the respective vector for :math:`f+g` and :math:`f \cdot g`,
since the derivatives of the sum and product are uniquely defined by the
derivatives of :math:`f` and :math:`g`.

The resulting operations of addition and multiplication lead to an algebra, the
so-called Truncated Power Series Algebra (TPSA) [ATTPSA]. The power of TPSA can
be enhanced by introducing derivations :math:`\partial` and their inverses,
corresponding to differentiation and integration in the space of functions. This
leads to the recognition of the underlying differential algebraic structure.

This structure is based on the following commuting diagrams for addition,
multiplication, and differentiation, where :math:`T` is the operation that
extracts the Taylor coefficients of a function:

.. math::
   :label: CD_eqn

   T(f \pm g) = T(f) \oplus T(g)

   T(f \cdot/ g) = T(f) \odot T(g)

   T(\partial f) = \partial_{\bigcirc} T(f)

   T(\partial^{-1} f) = \partial^{-1}_{\bigcirc} T(f)


In the equations above, the operation :math:`T` extracts the Taylor
coefficients of a prespecified order :math:`n`. The symbols :math:`\oplus,
\ominus, \odot, \oslash, \partial_{\bigcirc}` and :math:`\partial_{\bigcirc}^{-1}`
denote operations on the space of DA objects which are defined such that the
commuting relations hold.

Using Differential Algebra, we can compute the :math:`n`-th order derivative of
univariate elementary functions (e.g., :math:`\sin x, \cos x, \log x, \tan x`),
and by extension, we can compute the derivatives up to order :math:`n` of
multivariate functions. The detailed description of this process is described
in [AIEP108book]. Composition of two multivariate functions can also be defined
using DA. Many problems involving DA techniques can be formulated as
fixed-point problems, such as the inversion of a multivariate function. Here,
fixed-point theorems can be applied to prove the existence of a solution and
provide a practical means to obtain it [PMCAP04].

DA techniques have been widely applied in beam physics, asteroid problems, and
other problems involving the solution of ODEs or PDEs. The focus of the work we
present is finding a solution to the Laplace and Poisson equation using DA
techniques. In this context, it is worthwhile to look at the techniques that
already use DA to solve PDEs [AIE_P108book].
