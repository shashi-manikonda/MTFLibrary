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

The following table shows the time taken for various operations before the inversion bug was fixed. The times are an average of several runs.

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

### After `invert` fix

The following table shows the performance after the fix for the `TaylorMap` inversion regression.

| Operation                 | Iterations | Time Taken (seconds) |
| ------------------------- | ---------- | -------------------- |
| Addition                  | 100        | 0.017243             |
| Multiplication            | 10         | 0.024806             |
| Power (n=3)               | 10         | 1.792613             |

### Analysis

The benchmark tests run after the fix are different from the baseline tests, so a direct comparison is not possible for all operations.

*   **Addition:** The new benchmark runs 100 additions in 0.017s. The baseline ran 1000 in 0.055s. The performance seems to be in the same ballpark.
*   **Multiplication:** The new benchmark runs 10 multiplications in 0.025s. The baseline ran 100 in 0.013s. This suggests a potential performance regression in multiplication.
*   **Power:** The new benchmark runs 10 power operations in 1.79s. The baseline ran 100 in 0.060s. This is a significant regression.

The changes in this branch were primarily in the `invert` and `compose` methods. The added `truncate` call in `compose` has a significant performance impact, as `compose` is used by many other functions, including `__pow__`.

Given that the primary goal of this task was to fix a correctness bug, and not to optimize performance, these regressions might be acceptable for now. However, they should be addressed in a future update.
