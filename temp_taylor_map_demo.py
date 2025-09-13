import numpy as np
import mtflib
from mtflib import TaylorMap, mtf

mtf.initialize_mtf(max_order=4, max_dimension=2)

x = mtf.from_variable(1, 2)
y = mtf.from_variable(2, 2)
sin_x = mtf.sin(x)
cos_y = mtf.cos(y)
map_F = TaylorMap([sin_x, cos_y])
print(map_F)

map_G = TaylorMap([x + y, x - y])
print(map_G)

map_sum = map_F + map_G
print(map_sum)

map_composed = map_F.compose(map_G)
print(map_composed)

trace_F = map_F.trace()
print(f"Trace of F at (0,0): {trace_F}")

eval_point = {1: 0.5, 2: 0.2}
result_array = map_composed.substitute(eval_point)
print(f"F(G(0.5, 0.2)) from TaylorMap: {result_array}")

# Compare with numpy to verify
numpy_result = [np.sin(0.7), np.cos(0.3)]
print(f"NumPy equivalent for comparison: {numpy_result}")

# Create an invertible map
x_inv = mtf.from_variable(1, 2)
y_inv = mtf.from_variable(2, 2)
f1_inv = x_inv + 0.1 * y_inv**2
f2_inv = y_inv - 0.1 * x_inv**2
map_to_invert = TaylorMap([f1_inv, f2_inv])
print("--- Original Map to Invert ---")
print(map_to_invert)

# Invert the map
inverted_map = map_to_invert.invert()
print("\n--- Inverted Map ---")
print(inverted_map)

# Verify by composing F and F_inv
composition = inverted_map.compose(map_to_invert)
print("\n--- Composition of F_inv o F ---")
print(composition)
print("\n(Result should be close to the identity map [x, y])")
