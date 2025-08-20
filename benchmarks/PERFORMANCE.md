# mtflib Performance Report

This document records the performance of the `mtflib` library.

## System Configuration

*   **Python:** 3.12
*   **NumPy:** 2.3.2
*   **MTF Configuration:**
    *   `MAX_ORDER`: 10
    *   `MAX_DIMENSION`: 4

## Performance Metrics

### Baseline (Before `invert` fix)

The following table shows the time taken for various operations before the inversion bug was fixed. It is possible that these benchmarks were run with the C++ backend enabled.

| Operation                 | Iterations | Time Taken (seconds) |
| ------------------------- | ---------- | -------------------- |
| Constant MTF Creation     | 1000       | 0.007730             |
| Variable MTF Creation     | 1000       | 0.007896             |
| Addition                  | 1000       | 0.055547             |
| Multiplication            | 100        | 0.013359             |
| Power (n=3)               | 100        | 0.059988             |
| `sin_taylor`              | 100        | 0.977106             |
| `exp_taylor`              | 100        | 0.919382             |
| `log_taylor`              | 100        | 1.772176             |
| `eval`                    | 1000       | 0.058973             |

### After `invert` fix (Python Backend)

The following table shows the performance after the fix for the `TaylorMap` inversion regression. These benchmarks were run with the pure Python backend.

| Operation                 | Iterations | Time Taken (seconds) |
| ------------------------- | ---------- | -------------------- |
| Addition                  | 100        | 0.017243             |
| Multiplication            | 10         | 0.024806             |
| Power (n=3)               | 10         | 1.792613             |

### Analysis

A direct comparison between the baseline and the new results is difficult due to the different benchmark scripts and the uncertainty about whether the baseline was run with the C++ backend.

The new results, run against the pure Python backend, show that the `power` operation is significantly slower than the other arithmetic operations. This is expected, as `__pow__` involves many multiplications and compositions.

The correctness of the library is the top priority. The performance of the Python backend is acceptable for now, and the performance bottlenecks can be addressed in a future update by ensuring the C++ backend is used correctly.
