# Performance Profiling Report for `mtflib` Electromagnetism Demos

## 1. Executive Summary

This report details the performance analysis of the electromagnetism (EM) demo scripts within the `mtflib` library. The primary goal was to identify performance bottlenecks by separating the core Biot-Savart law calculations from the plotting and visualization routines, and to compare the performance of the pure Python implementation against the compiled (C++/Cython) backend.

**Key Findings:**

*   **The primary performance bottleneck is overwhelmingly in the core calculation stage, not the plotting routines.** For a representative workload, the calculation phase took several minutes, whereas the plotting phase completed in approximately 0.1 seconds.
*   **The compiled C++ backend is not being effectively utilized and provides no performance benefit.** Profiling reveals that even when the compiled extensions are enabled, the most time-consuming functions are still the pure Python implementations of Taylor series arithmetic. Paradoxically, the "compiled" version ran slightly slower than the pure Python version due to overhead.
*   **The main bottlenecks within the calculation are:**
    1.  The `numpy.unique` function, which consumes ~45% of the total execution time.
    2.  The pure Python `__mul__` method for `MultivariateTaylorFunction` objects, consuming ~38% of the time.
*   **The MPI implementation for parallelization scales well.** It achieves a near-linear speedup of **~3.5x on 4 cores** with low communication overhead (~2.5%). However, it is effectively parallelizing a slow, un-optimized calculation.

**Conclusion:** The performance issues do not lie with the EM demo scripts themselves, but with a fundamental implementation issue in the `mtflib` core that prevents the C++ backend from accelerating critical arithmetic operations.

## 2. Detailed Analysis: Calculation Phase

The calculation phase was profiled using the `serial_biot_savart` function with a workload of 200 field points and 100 total wire segments.

### 2.1. Python Implementation (Backend Disabled)

*   **Total Execution Time:** 226.5 seconds

*   **Top 5 Time-Consuming Functions:**
    | Function | Cumulative Time | Percentage |
    | :--- | :--- | :--- |
    | `_numpy_biot_savart_core` | 226.4s | ~100% |
    | `numpy.unique` | 102.9s | 45.4% |
    | `taylor_function.py:__mul__` | 86.0s | 37.9% |
    | `taylor_function.py:__pow__` | 82.7s | 36.5% |
    | `MTFExtended.py:__init__` | 74.4s | 32.8% |

### 2.2. Compiled Implementation (Backend Enabled)

*   **Total Execution Time:** 232.9 seconds

*   **Top 5 Time-Consuming Functions:**
    | Function | Cumulative Time | Percentage |
    | :--- | :--- | :--- |
    | `_numpy_biot_savart_core` | 232.8s | ~100% |
    | `numpy.unique` | 105.5s | 45.3% |
    | `taylor_function.py:__mul__` | 89.0s | 38.2% |
    | `taylor_function.py:__pow__` | 85.6s | 36.8% |
    | `MTFExtended.py:__init__` | 76.7s | 32.9% |

**Analysis:** The profiles are nearly identical. The "compiled" run shows no evidence of offloading critical operations like `__mul__` or `__pow__` to C++ code. The minor increase in total time for the compiled version likely stems from Python import system overheads or other minor differences.

### 2.3. MPI-Parallel Implementation Analysis

The MPI calculation was run on 4 processes with the same workload.

*   **Total Execution Time (4 processes):** 66.3 seconds
*   **Speedup vs. Serial:** 232.9s / 66.3s = **3.51x** (Excellent scaling, close to the ideal 4x)
*   **Communication vs. Computation:** On rank 0, the local computation (`numpy_biot_savart`) took 64.67s, while the total time in the `mpi_biot_savart` wrapper was 66.35s. This leaves ~1.68s (or **2.5%**) as MPI communication and logic overhead, which is very efficient.

## 3. Detailed Analysis: Plotting Phase

The plotting phase was profiled by first calculating the B-field data and then profiling only the `matplotlib` rendering calls.

*   **Total Execution Time (Compiled Backend):** ~0.113 seconds
*   **Total Execution Time (Python Backend):** ~0.105 seconds

**Analysis:** The choice of backend has no material impact on plotting performance. The time is dominated by calls to `matplotlib.axes3d.plot` and its associated helper functions for scaling and drawing. The plotting stage is highly efficient and is **not** a source of performance bottlenecks.

## 4. Recommendations

The results clearly point to the `mtflib` core library as the source of the performance issues. The following recommendations are prioritized to address these findings:

1.  **High Priority - Fix Backend Dispatch:** The most critical issue to resolve is the failure of the core arithmetic functions (`__mul__`, `__add__`, `__pow__`, etc.) in `taylor_function.py` to dispatch to their compiled C++/Cython counterparts. The Python code should be modified to check for the presence of the compiled extensions and, if available, delegate the computationally intensive work to them.

2.  **High Priority - Investigate `numpy.unique` Usage:** The fact that `numpy.unique` consumes ~45% of the runtime is a major finding. This function is likely used for managing Taylor series exponents during arithmetic operations. Its performance should be investigated and optimized. Potential solutions include:
    *   **Caching:** Memoize the results of exponent combination.
    *   **Alternative Data Structures:** Instead of relying on NumPy sorting and uniqueing, use a more efficient structure like a hash set (`set`) for managing exponents within the C++ extensions.

3.  **Medium Priority - Optimize Plotting Calculation Loop:** The current plotting functions call `serial_biot_savart` once for every point in a Python loop. This is inefficient. The code should be refactored to call `serial_biot_savart` a single time with an array of all field points, leveraging its vectorized design.

4.  **Guidance for Users:**
    *   For performance-critical applications, users should **use the MPI implementation**, as it provides significant, near-linear speedup.
    *   Users should be made aware that there is currently **no benefit to using the compiled backend** over the pure Python installation until the backend dispatch issue is resolved.
