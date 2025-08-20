# Performance Profiling Report for `mtflib` Electromagnetism Demos

This report details the performance analysis of the electromagnetism (EM) demo scripts within the `mtflib` library. The data was generated on the `fix-taylor-map-inversion-and-deps` branch after fixing the bug in the `invert` method.

## 1. Executive Summary

This report details the performance analysis of the electromagnetism (EM) demo scripts within the `mtflib` library. The primary goal was to identify performance bottlenecks by separating the core Biot-Savart law calculations from the plotting and visualization routines.

**Key Findings:**

*   **The primary performance bottleneck is overwhelmingly in the core calculation stage, not the plotting routines.** For a representative workload, the calculation phase took ~28 seconds, whereas the plotting phase completed in a fraction of a second.
*   **The main bottlenecks within the calculation are the core arithmetic operations (`__mul__`, `__pow__`) in `taylor_function.py`.** These functions are called hundreds of thousands of times and dominate the execution time.

## 2. Detailed Analysis: Calculation Phase

The calculation phase was profiled using the `serial_biot_savart` function.

### 2.1. Python Implementation

*   **Total Execution Time:** ~28 seconds

*   **Top 5 Time-Consuming Functions:**
    | Function                      | Cumulative Time |
    | :---------------------------- | :-------------- |
    | `serial_biot_savart`          | 27.784s         |
    | `__pow__`                     | 17.794s         |
    | `__mul__`                     | 17.016s         |
    | `compose_one_dim`             | 8.607s          |
    | `isqrt_taylor`                | 5.503s          |

## 3. Detailed Analysis: Plotting Phase

The plotting phase was profiled by first calculating the B-field data and then profiling only the `matplotlib` rendering calls.

*   **Total Execution Time (Python Backend):** ~10 seconds for calculation, plotting is much faster (~0.1s).

**Analysis:** The plotting stage is highly efficient and is **not** a source of performance bottlenecks.

## 4. Recommendations

The results clearly point to the `mtflib` core library as the source of the performance issues. The following recommendations are prioritized to address these findings:

1.  **High Priority - Optimize `__mul__` and `__pow__`:** These two functions are the main performance bottlenecks. They should be optimized, likely by improving the C++ backend and ensuring it is used correctly. The failure of the compiled backend to be used in the test environment should also be investigated.
2.  **Medium Priority - Optimize Plotting Calculation Loop:** The current plotting functions call `serial_biot_savart` once for every point in a Python loop. This is inefficient. The code should be refactored to call `serial_biot_savart` a single time with an array of all field points, leveraging its vectorized design.
