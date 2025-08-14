import time
import numpy as np
import mtflib
from mtflib import (
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
import argparse
import cProfile
import pstats
import random

# --- Default Benchmark Configuration ---
DEFAULT_ORDER = 10
DEFAULT_DIMENSION = 4
DEFAULT_NUM_TERMS = 20
DEFAULT_ETOL = 1e-12

def generate_random_mtf(dimension, max_order, num_terms, implementation='python'):
    """
    Generates a random MultivariateTaylorFunction for benchmarking.
    This implementation avoids generating all possible exponents to save memory.
    """
    if dimension == 0:
        return MultivariateTaylorFunction.from_constant(np.random.rand())

    exponents = set()
    # Add a constant term to ensure logs and other functions are well-behaved
    exponents.add(tuple([0] * dimension))

    while len(exponents) < num_terms:
        exp = [0] * dimension
        order = random.randint(0, max_order)
        for _ in range(order):
            d = random.randint(0, dimension - 1)
            exp[d] += 1

        # Ensure the total order does not exceed max_order
        if sum(exp) <= max_order:
            exponents.add(tuple(exp))

    coeffs = np.random.rand(len(exponents))

    return MultivariateTaylorFunction(
        (np.array(list(exponents), dtype=np.int32), coeffs),
        dimension=dimension,
        implementation=implementation
    )

def setup_environment(max_order, max_dimension, etol):
    """Initializes the MTF library globals."""
    # Reset initialization flag to allow re-running in the same session
    if hasattr(mtflib.taylor_function, '_INITIALIZED') and mtflib.taylor_function._INITIALIZED:
        mtflib.taylor_function._INITIALIZED = False
    initialize_mtf_globals(max_order=max_order, max_dimension=max_dimension)
    set_global_etol(etol)

def benchmark_arithmetic(mtf1, mtf2, implementation='python'):
    """Benchmarks arithmetic operations."""
    print(f"\n--- Benchmarking Arithmetic Operations (Implementation: {implementation}) ---")

    # Benchmark Addition
    start_time = time.time()
    for _ in range(100):
        _ = mtf1 + mtf2
    end_time = time.time()
    print(f"Time for 100 additions: {end_time - start_time:.6f} seconds")

    # Benchmark Multiplication
    start_time = time.time()
    for _ in range(10):
        _ = mtf1 * mtf2
    end_time = time.time()
    print(f"Time for 10 multiplications: {end_time - start_time:.6f} seconds")

    # Benchmark Power
    start_time = time.time()
    for _ in range(10):
        _ = mtf1 ** 3
    end_time = time.time()
    print(f"Time for 10 power operations (n=3): {end_time - start_time:.6f} seconds")

def run_benchmarks(args):
    """Run all the benchmarks based on the provided arguments."""
    print("Setting up benchmark environment...")
    setup_environment(args.order, args.dimension, DEFAULT_ETOL)
    print(f"Benchmark Configuration: Order={args.order}, Dimension={args.dimension}, Terms={args.num_terms}, Implementation={args.implementation}")

    # Generate random MTFs for benchmarking
    print("\nGenerating random MTFs for benchmarking...")
    mtf1 = generate_random_mtf(args.dimension, args.order, args.num_terms, args.implementation)
    mtf2 = generate_random_mtf(args.dimension, args.order, args.num_terms, args.implementation)
    print("MTF generation complete.")

    benchmark_arithmetic(mtf1, mtf2, args.implementation)

def main():
    """Main function to parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark script for mtflib.")
    parser.add_argument('--order', type=int, default=DEFAULT_ORDER, help='Maximum order of Taylor series.')
    parser.add_argument('--dimension', type=int, default=DEFAULT_DIMENSION, help='Number of variables.')
    parser.add_argument('--num-terms', type=int, default=DEFAULT_NUM_TERMS, help='Number of non-zero terms in the random MTFs.')
    parser.add_argument('--implementation', type=str, default='python', choices=['python', 'cython', 'cpp'], help='Implementation to benchmark.')
    parser.add_argument('--profile', action='store_true', help='Enable cProfile profiling.')

    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

        run_benchmarks(args)

        profiler.disable()
        stats_file = f"profile_results_{args.implementation}_order{args.order}_dim{args.dimension}_terms{args.num_terms}.prof"
        profiler.dump_stats(stats_file)
        print(f"\nProfiling results saved to {stats_file}")
        print(f"To view stats, run: python -m pstats {stats_file}")
    else:
        run_benchmarks(args)

    print("\nBenchmark run complete.")

if __name__ == "__main__":
    main()
