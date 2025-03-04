#test_MFT.py

from MTFLibrary import *

import numpy as np

initialize_mtf_globals(max_order=6, max_dimension=3)
set_global_etol(1e-16)

# Set global max order for Taylor expansions
# global_max_order = 10
# set_global_max_order(global_max_order)
# print(f"Global max order set to: {global_max_order}\n")

# Define variables
x = Var(1) # Var_id=1, dimension=2 - v9.5
y = Var(2) # Var_id=2, dimension=2 - v9.5
z = Var(3) # Var_id=2, dimension=2 - v9.5

print(f"Variable x: {x}") # Print var details - v9.5
print(f"Variable y: {y}") # Print var details - v9.5

evaluation_point = np.array([0.5, 0.2]) # 2D evaluation point - v9.5
print(f"Evaluation point: {evaluation_point}\n") # Print evaluation point - v9.5

# --- Elementary Functions ---
print("--- Elementary Functions ---\n")

cos_x_tf = cos_taylor(x)
print(f"Dimension of cos_x_tf: {cos_x_tf.dimension}")  # Debug print for dimension
print(f"cos_taylor(x):\n{cos_x_tf}")

sin_y_tf = sin_taylor(y)
print(f"\nsin_taylor(y):\n{sin_y_tf}") 

exp_x_tf = exp_taylor(x)
print(f"\nexp_taylor(x):\n{exp_x_tf}") 

gaussian_x_tf = gaussian_taylor(x)
print(f"\ngaussian_taylor(x):\n{gaussian_x_tf}") 

sqrt_x_tf = sqrt_taylor(1+x)
print(f"\nsqrt_taylor(x):\n{sqrt_x_tf}") 

log_y_tf = log_taylor(1+y)
print(f"\nlog_taylor(y):\n{log_y_tf}") 

arctan_x_tf = arctan_taylor(x+2*y+.5*z)
print(f"\narctan_taylor(x):\n{arctan_x_tf}") 

print(1/(1+x**2+y))