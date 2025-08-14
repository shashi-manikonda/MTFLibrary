# MTFLibrary Performance Baseline

This document records the baseline performance of the MTFLibrary before any optimizations have been applied. The metrics were gathered by running the `benchmark.py` script.

## System Configuration

*   **Python:** 3.12
*   **NumPy:** 2.3.2
*   **MTF Configuration:**
    *   `MAX_ORDER`: 10
    *   `MAX_DIMENSION`: 4

## Baseline Performance Metrics

The following table shows the time taken for various operations. The times are an average of several runs.

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

## Profiling Summary

Profiling was conducted using `cProfile`. The results confirm that the performance bottlenecks are concentrated in a few key areas:

*   **`__mul__` (Multiplication):** This is the single most time-consuming function. It is called thousands of times, especially during function composition. The pure Python implementation with dictionary lookups is highly inefficient.
*   **`compose_one_dim` (Function Composition):** This is the second major bottleneck. Its high cumulative time is a result of it repeatedly calling `__pow__`, which in turn calls `__mul__`. All elementary functions (`sin`, `exp`, etc.) rely on this function, making its optimization critical.

The optimization plan will focus on rewriting these two functions and their underlying data structures in Cython to achieve significant performance gains.
