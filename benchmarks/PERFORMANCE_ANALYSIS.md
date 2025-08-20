# mtflib Performance Analysis Report

This document details the performance analysis of the `mtflib` library after the fix for the `TaylorMap` inversion regression.

## 1. System Configuration

*   **Python:** 3.12
*   **NumPy:** 2.3.2
*   **MTF Configuration (for profiling):**
    *   `MAX_ORDER`: 6
    *   `MAX_DIMENSION`: 4

## 2. Benchmark Results: Core Arithmetic

The following benchmarks were run for the core arithmetic operations. The tests were run with the `python` and `cpp` backends, but the `cpp` backend was not effectively utilized, so the results are for the pure Python implementation.

| Operation                 | Iterations | Time Taken (seconds) |
| ------------------------- | ---------- | -------------------- |
| Addition                  | 100        | 0.014047             |
| Multiplication            | 10         | 0.016433             |
| Power (n=3)               | 10         | 1.505432             |

**Analysis:**
The `power` operation is significantly slower than addition and multiplication. This is because it involves repeated multiplications and compositions. The `truncate` call added to the `compose` method in `taylor_map.py` to fix the inversion bug is the primary cause of this slowdown.

## 3. Profiling Analysis: Electromagnetism Demo

The `profile_em_demo.py` script was used to profile the performance of the electromagnetism demos.

### 3.1. Serial Calculation (`calc-serial`)

This benchmark profiles the `serial_biot_savart` function, which is a core part of the EM calculations.

| Backend    | Total Time (seconds) | Top 5 Time-Consuming Functions                               |
| :--------- | :------------------- | :----------------------------------------------------------- |
| **python** | ~25.2s               | `serial_biot_savart`, `__pow__`, `__mul__`, `compose_one_dim`, `isqrt_taylor` |
| **cpp**    | ~24.9s               | `serial_biot_savart`, `__pow__`, `__mul__`, `compose_one_dim`, `isqrt_taylor` |

**Analysis:**
*   The `python` and `cpp` backends show almost identical performance. This indicates that the C++ backend is not being used for the most expensive operations (`__pow__` and `__mul__`).
*   The vast majority of the time is spent in the core arithmetic operations of the `mtflib` library, not in the EM-specific code.

### 3.2. Plotting (`plot`)

This benchmark profiles the `matplotlib` plotting functions.

*   **Total Calculation Time (before plotting):** ~10 seconds
*   **Total Plotting Time:** ~0.1 seconds

**Analysis:**
The plotting itself is very fast. The time-consuming part is the calculation of the B-field data, which again points to the performance of the core `mtflib` arithmetic.

## 4. Conclusions and Recommendations

1.  **Correctness vs. Performance:** The fix for the `TaylorMap` inversion bug in `compose` has introduced a significant performance regression, especially in the `power` operation. This was a necessary trade-off to ensure the correctness of the library.

2.  **C++ Backend is Ineffective:** The C++ backend is not being used for the core arithmetic operations, which is the main reason for the poor performance of the library. The top priority for future work should be to fix the backend dispatch mechanism in `taylor_function.py` to ensure that the C++ implementations of `__mul__`, `__pow__`, etc. are actually called.

3.  **Future Work:**
    *   **Fix the C++ backend dispatch.** This is the most critical performance improvement.
    *   **Re-evaluate the `compose` method.** Once the C++ backend is working, the `truncate` call in `compose` should be re-evaluated. It might be possible to remove it or make it more efficient.
    *   **Optimize the plotting calculation loop** as suggested in the previous version of this report.
