import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import mtflib
from mtflib.taylor_function import (
    MultivariateTaylorFunction,
)
from mtflib.elementary_functions import (
    sin_taylor,
    exp_taylor,
    log_taylor,
)
from src.applications.em.biot_savart import serial_biot_savart
import argparse
import cProfile
import pstats
import random
import os

# --- Default Benchmark Configuration ---
DEFAULT_ORDER = 10           # Options: any positive integer
DEFAULT_DIMENSION = 4        # Options: any positive integer
DEFAULT_NUM_TERMS = 20       # Options: any positive integer
DEFAULT_ETOL = 1e-12         # Options: any small positive float
DEFAULT_IMPLEMENTATION = 'cpp' # Options: 'python', 'cython', 'cpp'
DEFAULT_NUM_POINTS = 100     # Options: any positive integer
ADD_ITERATIONS = 100         # Options: any positive integer
MUL_ITERATIONS = 10          # Options: any positive integer
POW_ITERATIONS = 10          # Options: any positive integer
POW_EXPONENT = 3             # Options: any integer

def generate_random_mtf(dimension, max_order, num_terms):
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
        dimension=dimension
    )

def setup_environment(max_order, max_dimension, etol, implementation):
    """Initializes the MTF library globals."""
    MultivariateTaylorFunction.initialize_mtf(max_order=max_order, max_dimension=max_dimension, implementation=implementation)
    MultivariateTaylorFunction.set_etol(etol)

def benchmark_arithmetic(mtf1, mtf2, implementation='python'):
    """Benchmarks arithmetic operations."""
    print(f"\n--- Benchmarking Arithmetic Operations (Implementation: {implementation}) ---")

    # Benchmark Addition
    start_time = time.time()
    for _ in range(ADD_ITERATIONS):
        _ = mtf1 + mtf2
    end_time = time.time()
    print(f"Time for {ADD_ITERATIONS} additions: {end_time - start_time:.6f} seconds")

    # Benchmark Multiplication
    start_time = time.time()
    for _ in range(MUL_ITERATIONS):
        _ = mtf1 * mtf2
    end_time = time.time()
    print(f"Time for {MUL_ITERATIONS} multiplications: {end_time - start_time:.6f} seconds")

    # Benchmark Power
    start_time = time.time()
    for _ in range(POW_ITERATIONS):
        _ = mtf1 ** POW_EXPONENT
    end_time = time.time()
    print(f"Time for {POW_ITERATIONS} power operations (n={POW_EXPONENT}): {end_time - start_time:.6f} seconds")

def benchmark_biot_savart(implementation, num_points):
    """Benchmarks the serial_biot_savart function."""
    print(f"\n--- Benchmarking serial_biot_savart (Implementation: {implementation}) ---")

    element_centers = np.array([[0, 0, 0], [1, 0, 0]])
    element_lengths = np.array([0.1, 0.1])
    element_directions = np.array([[1, 0, 0], [0, 1, 0]])
    field_points = np.random.rand(num_points, 3)

    start_time = time.time()
    _ = serial_biot_savart(element_centers, element_lengths, element_directions, field_points)
    end_time = time.time()
    print(f"Time for serial_biot_savart with {num_points} points: {end_time - start_time:.6f} seconds")

def run_benchmarks(args):
    """Run all the benchmarks based on the provided arguments."""
    print("Setting up benchmark environment...")
    setup_environment(args.order, args.dimension, DEFAULT_ETOL, args.implementation)
    print(f"Benchmark Configuration: Order={args.order}, Dimension={args.dimension}, Terms={args.num_terms}, Implementation={args.implementation}")

    # Generate random MTFs for benchmarking
    print("\nGenerating random MTFs for benchmarking...")
    mtf1 = generate_random_mtf(args.dimension, args.order, args.num_terms)
    mtf2 = generate_random_mtf(args.dimension, args.order, args.num_terms)
    print("MTF generation complete.")

    benchmark_arithmetic(mtf1, mtf2, args.implementation)
    benchmark_biot_savart(args.implementation, args.num_points)


def main():
    """Main function to parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark script for mtflib.")
    parser.add_argument('--order', type=int, default=DEFAULT_ORDER, help='Maximum order of Taylor series.')
    parser.add_argument('--dimension', type=int, default=DEFAULT_DIMENSION, help='Number of variables.')
    parser.add_argument('--num-terms', type=int, default=DEFAULT_NUM_TERMS, help='Number of non-zero terms in the random MTFs.')
    parser.add_argument('--num-points', type=int, default=DEFAULT_NUM_POINTS, help='Number of field points for Biot-Savart benchmark.')
    parser.add_argument('--implementation', type=str, default=DEFAULT_IMPLEMENTATION, choices=['python', 'cpp'], help='Implementation to benchmark.')
    parser.add_argument('--profile', action='store_true', help='Enable cProfile profiling.')

    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

        run_benchmarks(args)

        profiler.disable()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        profiling_dir = os.path.join(script_dir, "profiling_results")
        os.makedirs(profiling_dir, exist_ok=True)
        stats_file = os.path.join(profiling_dir, f"profile_results_{args.implementation}_order{args.order}_dim{args.dimension}_terms{args.num_terms}.prof")
        profiler.dump_stats(stats_file)
        print(f"\nProfiling results saved to {stats_file}")
        print(f"To view stats, run: python -m pstats {stats_file}")
    else:
        run_benchmarks(args)

    print("\nBenchmark run complete.")

if __name__ == "__main__":
    main()
