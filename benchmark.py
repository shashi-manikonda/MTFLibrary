import time
import numpy as np
import MTFLibrary
from MTFLibrary import (
    initialize_mtf_globals,
    set_global_etol,
    get_global_max_dimension,
    get_global_max_order,
    Var,
    MultivariateTaylorFunction,
    sin_taylor,
    exp_taylor,
    log_taylor,
)

# --- Benchmark Configuration ---
MAX_ORDER = 10
MAX_DIMENSION = 4
ETOL = 1e-12

def setup_environment():
    """Initializes the MTF library globals."""
    if MTFLibrary.taylor_function._INITIALIZED:
        MTFLibrary.taylor_function._INITIALIZED = False
    initialize_mtf_globals(max_order=MAX_ORDER, max_dimension=MAX_DIMENSION)
    set_global_etol(ETOL)

def benchmark_initialization():
    """Benchmarks the creation of MTF objects."""
    print("\n--- Benchmarking Initialization ---")

    # Time creation of a constant MTF
    start_time = time.time()
    for _ in range(1000):
        MultivariateTaylorFunction.from_constant(1.0)
    end_time = time.time()
    print(f"Time for 1000 constant MTF creations: {end_time - start_time:.6f} seconds")

    # Time creation of a variable MTF
    start_time = time.time()
    for _ in range(1000):
        Var(1)
    end_time = time.time()
    print(f"Time for 1000 variable MTF creations: {end_time - start_time:.6f} seconds")

def benchmark_arithmetic():
    """Benchmarks arithmetic operations."""
    print("\n--- Benchmarking Arithmetic Operations ---")
    x = Var(1)
    y = Var(2)

    # Create two moderately complex MTFs
    mtf1 = 1 + x + 0.5 * x**2 + 0.2 * y**2
    mtf2 = 2 - 0.5 * x + y**2

    # Benchmark Addition
    start_time = time.time()
    for _ in range(1000):
        _ = mtf1 + mtf2
    end_time = time.time()
    print(f"Time for 1000 additions: {end_time - start_time:.6f} seconds")

    # Benchmark Multiplication
    start_time = time.time()
    for _ in range(100):
        _ = mtf1 * mtf2
    end_time = time.time()
    print(f"Time for 100 multiplications: {end_time - start_time:.6f} seconds")

    # Benchmark Power
    start_time = time.time()
    for _ in range(100):
        _ = mtf1 ** 3
    end_time = time.time()
    print(f"Time for 100 power operations (3): {end_time - start_time:.6f} seconds")

def benchmark_elementary_functions():
    """Benchmarks elementary functions."""
    print("\n--- Benchmarking Elementary Functions ---")
    x = Var(1)
    y = Var(2)
    mtf = 1 + 0.1 * x + 0.2 * y

    # Benchmark sin_taylor
    start_time = time.time()
    for _ in range(100):
        _ = sin_taylor(mtf)
    end_time = time.time()
    print(f"Time for 100 sin_taylor operations: {end_time - start_time:.6f} seconds")

    # Benchmark exp_taylor
    start_time = time.time()
    for _ in range(100):
        _ = exp_taylor(mtf)
    end_time = time.time()
    print(f"Time for 100 exp_taylor operations: {end_time - start_time:.6f} seconds")

    # Benchmark log_taylor
    mtf_log = 1 + 0.1 * x + 0.2 * y # Ensure constant part is positive
    start_time = time.time()
    for _ in range(100):
        _ = log_taylor(mtf_log)
    end_time = time.time()
    print(f"Time for 100 log_taylor operations: {end_time - start_time:.6f} seconds")


def benchmark_evaluation():
    """Benchmarks the eval method."""
    print("\n--- Benchmarking Evaluation ---")
    x = Var(1)
    y = Var(2)
    z = Var(3)
    mtf = 1 + x + 0.5*y**2 + 0.1*z**3

    eval_point = [0.5, 0.2, 0.1] + [0] * (get_global_max_dimension() - 3)

    start_time = time.time()
    for _ in range(1000):
        mtf.eval(eval_point)
    end_time = time.time()
    print(f"Time for 1000 evaluations: {end_time - start_time:.6f} seconds")

def main():
    """Main function to run all benchmarks."""
    print("Setting up benchmark environment...")
    setup_environment()
    print(f"Benchmark Configuration: MAX_ORDER={MAX_ORDER}, MAX_DIMENSION={MAX_DIMENSION}")

    benchmark_initialization()
    benchmark_arithmetic()
    benchmark_elementary_functions()
    benchmark_evaluation()

    print("\nBenchmark run complete.")

if __name__ == "__main__":
    main()
