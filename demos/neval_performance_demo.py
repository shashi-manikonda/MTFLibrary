import numpy as np
import time
from mtflib import MultivariateTaylorFunction

def old_eval_loop(mtf, points):
    """
    Evaluates the MTF at each point in a loop using the old eval logic.
    """
    results = np.empty(points.shape[0])
    for i, point in enumerate(points):
        term_values = np.prod(np.power(point, mtf.exponents), axis=1)
        results[i] = np.einsum('j,j->', mtf.coeffs, term_values)
    return results

def main():
    """
    Main function to run the performance demo.
    """
    # --- Setup ---
    print("Setting up the performance demo...")
    MultivariateTaylorFunction.initialize_mtf(max_order=5, max_dimension=3)

    # Create a sample MTF
    x = MultivariateTaylorFunction.from_variable(1, 3)
    y = MultivariateTaylorFunction.from_variable(2, 3)
    z = MultivariateTaylorFunction.from_variable(3, 3)
    mtf = np.sin(x) + np.cos(y) + np.exp(z)

    # Generate a large number of random points
    n_points = 100_000
    dimension = 3
    points = np.random.rand(n_points, dimension)
    print(f"Generated {n_points} random points in {dimension} dimensions.")
    print("-" * 30)

    # --- Benchmark old eval loop ---
    print("Benchmarking the old eval method (looping)...")
    start_time = time.time()
    results_old = old_eval_loop(mtf, points)
    end_time = time.time()
    time_old = end_time - start_time
    print(f"Time taken: {time_old:.4f} seconds")
    print("-" * 30)

    # --- Benchmark new neval method ---
    print("Benchmarking the new neval method (vectorized)...")
    start_time = time.time()
    results_new = mtf.neval(points)
    end_time = time.time()
    time_new = end_time - start_time
    print(f"Time taken: {time_new:.4f} seconds")
    print("-" * 30)

    # --- Comparison ---
    print("Performance Comparison:")
    if time_new > 0:
        speedup = time_old / time_new
        print(f"The new `neval` method is {speedup:.2f}x faster than the old `eval` loop.")
    else:
        print("Could not calculate speedup (neval was too fast).")

    # Verify that the results are consistent
    assert np.allclose(results_old, results_new), "Results from old and new methods do not match."
    print("\nResults from both methods are consistent.")

if __name__ == "__main__":
    main()
