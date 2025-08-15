import numpy as np
import mtflib
from mtflib import TaylorMap, MTF

def run_demo():
    """
    A simple demo to showcase the functionality of the TaylorMap class.
    """
    # 1. Initialize mtflib global parameters
    # This is a mandatory step before using any mtflib functionality.
    # We set the maximum order of the Taylor series and the number of variables.
    try:
        mtflib.initialize_mtf_globals(max_order=4, max_dimension=2)
    except RuntimeError:
        # Globals might already be initialized if running in an interactive session
        # where this demo has been run before.
        pass

    print("--- TaylorMap Demo ---")
    print("mtflib initialized for max_order=4, max_dimension=2\n")

    # 2. Create two TaylorMap objects
    # Let's define two maps from R^2 to R^2.

    # Map 1: F(x,y) = [sin(x), cos(y)]
    print("--- Creating Map 1: F(x,y) = [sin(x), cos(y)] ---")
    x = MTF.from_variable(1, 2)
    y = MTF.from_variable(2, 2)
    sin_x = mtflib.sin_taylor(x)
    cos_y = mtflib.cos_taylor(y)
    map_F = TaylorMap([sin_x, cos_y])
    print(map_F)

    # Map 2: G(x,y) = [x + y, x - y]
    print("--- Creating Map 2: G(x,y) = [x + y, x - y] ---")
    map_G = TaylorMap([x + y, x - y])
    print(map_G)

    # 3. Demonstrate Operations

    # Operation 1: Addition (F + G)
    print("--- Operation 1: Addition (F + G) ---")
    map_sum = map_F + map_G
    print(map_sum)

    # Operation 2: Composition (F(G(x,y)))
    print("--- Operation 2: Composition F(G(x,y)) ---")
    # This computes sin(x+y) and cos(x-y)
    map_composed = map_F.compose(map_G)
    print(map_composed)

    # Operation 3: Trace
    # The trace is the sum of the diagonal elements of the Jacobian matrix's linear part.
    # For F(x,y) = [sin(x), cos(y)], the Jacobian is [[cos(x), 0], [0, -sin(y)]].
    # At (0,0), the linear part is [[1, 0], [0, 0]], so the trace is 1.
    print("--- Operation 3: Trace of F ---")
    trace_F = map_F.trace()
    print(f"Trace of F at (0,0): {trace_F}") # Expected: cos(0) + 0 = 1

    # Operation 4: Substitution
    # Let's evaluate the composed map F(G(x,y)) at x=0.5, y=0.2
    print("\n--- Operation 4: Full Substitution into composed map at (0.5, 0.2) ---")
    # We are evaluating [sin(x+y), cos(x-y)] at x=0.5, y=0.2
    # This is sin(0.7) and cos(0.3)
    eval_point = {1: 0.5, 2: 0.2}
    result_array = map_composed.substitute(eval_point)
    print(f"F(G(0.5, 0.2)) from TaylorMap: {result_array}")

    # Compare with numpy to verify
    numpy_result = [np.sin(0.7), np.cos(0.3)]
    print(f"NumPy equivalent for comparison: {numpy_result}")

    # The Taylor series approximation should be close to the actual value.
    # The accuracy depends on the order of the series.

if __name__ == "__main__":
    run_demo()
