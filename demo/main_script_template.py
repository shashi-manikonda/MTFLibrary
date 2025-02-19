from MTFLibrary.taylor_function import MultivariateTaylorFunction, Var, set_global_max_order # Var is now imported from taylor_function
from MTFLibrary.elementary_functions import cos_taylor, sin_taylor, exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor, sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, arctanh_taylor
import numpy as np
import math

# Set global max order for Taylor expansions
global_max_order = 10
set_global_max_order(global_max_order)
print(f"Global max order set to: {global_max_order}\n")

# Define variables
x = Var(1, dimension=2) # Var_id=1, dimension=2 - v9.5
y = Var(2, dimension=2) # Var_id=2, dimension=2 - v9.5
print(f"Variable x: {x}") # Print var details - v9.5
print(f"Variable y: {y}") # Print var details - v9.5

evaluation_point = np.array([0.5, 0.2]) # 2D evaluation point - v9.5
print(f"Evaluation point: {evaluation_point}\n") # Print evaluation point - v9.5

# --- Elementary Functions ---
print("--- Elementary Functions ---\n")

cos_x_tf = cos_taylor(x, order=global_max_order)
print(f"Dimension of cos_x_tf: {cos_x_tf.dimension}")  # Debug print for dimension
print(f"cos_taylor(x):\n{cos_x_tf}")

sin_y_tf = sin_taylor(y, order=global_max_order)
print(f"\nsin_taylor(y):\n{sin_y_tf}") 

exp_x_tf = exp_taylor(x, order=global_max_order)
print(f"\nexp_taylor(x):\n{exp_x_tf}") 

gaussian_x_tf = gaussian_taylor(x, order=global_max_order)
print(f"\ngaussian_taylor(x):\n{gaussian_x_tf}") 

sqrt_x_tf = sqrt_taylor(x, order=global_max_order)
print(f"\nsqrt_taylor(x):\n{sqrt_x_tf}") 

log_y_tf = log_taylor(y, order=global_max_order)
print(f"\nlog_taylor(y):\n{log_y_tf}") 

arctan_x_tf = arctan_taylor(x, order=global_max_order)
print(f"\narctan_taylor(x):\n{arctan_x_tf}") 

sinh_y_tf = sinh_taylor(y, order=global_max_order)
print(f"\nsinh_taylor(y):\n{sinh_y_tf}") 

cosh_x_tf = cosh_taylor(x, order=global_max_order)
print(f"\ncosh_taylor(x):\n{cosh_x_tf}") 

tanh_y_tf = tanh_taylor(y, order=global_max_order)
print(f"\ntanh_taylor(y):\n{tanh_y_tf}") 

arcsin_x_tf = arcsin_taylor(x, order=global_max_order)
print(f"\narcsin_taylor(x):\n{arcsin_x_tf}") 

arccos_y_tf = arccos_taylor(y, order=global_max_order)
print(f"\narccos_taylor(y):\n{arccos_y_tf}") 

arctanh_x_tf = arctanh_taylor(x, order=global_max_order)
print(f"\narctanh_taylor(x):\n{arctanh_x_tf}") 


# --- Arithmetic Operations ---
print("\n--- Arithmetic Operations ---\n")

f1 = cos_x_tf * sin_y_tf # Multiplication of two MTFs
f2 = f1 / 3.0 # Division by a scalar
print(f"Dimension of f2 after scalar division: {f2.dimension}") # Print dimension of f2
print(f"\nCombined function f2:\n{f2}") 


# --- Derivative and Integration ---
print("\n--- Derivative and Integration ---\n")

# Derivative of f2 with respect to x (var_id=1)
deriv_f2_x = f2.derivative(wrt_variable_id=1)
print(f"Derivative of f2 wrt x (var_id=1):\n{deriv_f2_x}") 

# Derivative of f2 with respect to y (var_id=2)
deriv_f2_y = f2.derivative(wrt_variable_id=2)
print(f"\nDerivative of f2 wrt y (var_id=2):\n{deriv_f2_y}") 

# Integral of f2 wrt x (var_id=1) with integration constant 0
integ_f2_x = f2.integral(wrt_variable_id=1, integration_constant=0.0)
print(f"\nIntegral of f2 wrt x (var_id=1):\n{integ_f2_x}") 

# Integral of f2 wrt y (var_id=2) with integration constant 0
integ_f2_y = f2.integral(wrt_variable_id=2, integration_constant=0.0) # Integral of f2 wrt y (var_id=2) with constant 0
print(f"\nIntegral of f2 wrt y (var_id=2):\n{integ_f2_y}") 


# --- Composition ---
print("\n--- Composition ---\n")

# Example composition: cos(x + sin(y))
composed_func = cos_taylor(x + sin_taylor(y, order=global_max_order), order=global_max_order)
print(f"Composed function cos(x + sin(y)):\n{composed_func}") 

# --- Evaluation ---
print("\n--- Evaluation ---\n")

f2_evaluated = f2.evaluate(evaluation_point)
print(f"Evaluated f2 at {evaluation_point}: {f2_evaluated}")

deriv_f2_x_evaluated = deriv_f2_x.evaluate(evaluation_point)
print(f"Evaluated derivative of f2 wrt x at {evaluation_point}: {deriv_f2_x_evaluated}")

deriv_f2_y_evaluated = deriv_f2_y.evaluate(evaluation_point)
print(f"Evaluated derivative of f2 wrt y at {evaluation_point}: {deriv_f2_y_evaluated}")

integ_f2_x_evaluated = integ_f2_x.evaluate(evaluation_point)
print(f"Evaluated integral of f2 wrt x at {evaluation_point}: {integ_f2_x_evaluated}")

integ_f2_y_evaluated = integ_f2_y.evaluate(evaluation_point)
print(f"Evaluated integral of f2 wrt y at {evaluation_point}: {integ_f2_y_evaluated}")

composed_func_evaluated = composed_func.evaluate(evaluation_point)
print(f"Evaluated composed function at {evaluation_point}: {composed_func_evaluated}")