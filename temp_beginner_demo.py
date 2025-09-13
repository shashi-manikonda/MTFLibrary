from mtflib import mtf, var, integrate, derivative, mtfarray
import numpy as np

# Check if MTF globals are already initialized
if not mtf.get_mtf_initialized_status():
    mtf.initialize_mtf(max_order=8, max_dimension=3)
    mtf.set_etol(1e-16)
else:
    print("MTF globals are already initialized. To change max_order or max_dimension, please restart the session.")

# Define variables for a 3-dimensional space (since max_dimension is 3)
x = var(1)
y = var(2)
z = var(3)

print(f"Variable x:\n{x}")
print(f"Variable y:\n{y}")
print(f"Variable z:\n{z}")
print("--Explanation of Taylor expansion table view columns--")
print("I: Index of the term in the Taylor series expansion (for internal tracking)")
print("Coefficient: The numerical value of the Taylor expansion coefficient")
print("Order: The total order (sum of exponents) of the Taylor term")
print("Exponents: The powers of each variable in the term, corresponding to (power of x, power of y, power of z)")

# Taylor expansion of sin(x)
sin_x = mtf.sin(x)
print(f"Taylor expansion of sin(x):\n{sin_x}")

# Calculate the first derivative of sin(x) with respect to x (variable index 1)
derivative_sin_x = derivative(sin_x, 1)
print(f"Derivative of sin(x) with respect to x:\n{derivative_sin_x}")

# Compare with the Taylor expansion of cos(x)
cos_x = mtf.cos(x)
print(f"Taylor expansion of cos(x):\n{cos_x}")

# Calculate the indefinite integral of sin(x) with respect to x (variable index 1)
indef_integral_sin_x = integrate(sin_x, 1)
print(f"Indefinite integral of sin(x) with respect to x:\n{indef_integral_sin_x}")

# Calculate the definite integral of sin(x) with respect to x from 0 to π/2
def_integral_sin_x = integrate(sin_x, 1, 0, np.pi/2)
print(f"Definite integral of sin(x) from 0 to π/2:\n{def_integral_sin_x}")

# Define a Taylor function with three variables: exp(x + y**4 + z**6)
exp_xyz = mtf.exp(x + y**4 + z**6)
print(f"Taylor expansion of exp(x + y**4 + z**6):\n{exp_xyz}")

# Calculate the indefinite integral with respect to y (variable index 2)
indef_integral_exp_xyz = integrate(exp_xyz, 2)
print(f"Indefinite integral of exp(x + y**4 + z**6) with respect to y:\n{indef_integral_exp_xyz}")

# Calculate the definite integral with respect to y (variable index 2) from 0 to 1
def_integral_exp_xyz = integrate(exp_xyz, 2, 0, 1)
print(f"Definite integral of exp(x + y**4 + z**6) with respect to y from 0 to 1:\n{def_integral_exp_xyz}")

# Calculate the first derivative with respect to z (variable index 3)
derivative_exp_xyz = derivative(exp_xyz, 3)
print(f"Derivative of exp(x + y**4 + z**6) with respect to z:\n{derivative_exp_xyz}")

# Evaluate the function at the point [x=1, y=0, z=0]. The `eval` method takes a NumPy array representing the point.
eval_point = np.array([1, 0, 0])
eval_result = exp_xyz.eval(eval_point)
print(f"Evaluation of exp(x + y**4 + z**6) at [1, 0, 0]: {eval_result}")

# Define a simple MTF: f(x, y, z) = x + y + z**4
fxyz = x + y + z**4
print(f"Original MTF: f(x, y, z) =\n{fxyz}")

# Substitute x (variable index 1) with the constant 0.1
gxyz = fxyz.substitute_variable(1, 0.1)
print(f"After substituting x with 0.1: g(y, z) =\\n{gxyz}")

# Define the outer MTF: f(x, y) = x**2 + y**3 (in a space of at least 2 dimensions)
fxy = x**2 + y**3
print(f'Outer MTF before substitution: f(x, y) =\\n{fxy}')

# Define the substituting MTFs:
# Substitute x (index 1) with g(y) = 1 + y
g_y = 1 + y
# Substitute y (index 2) with h(y, z) = 1 + y * z
h_yz = 1 + y * z

# Create the substitution dictionary
substitution_dict = {
    1: g_y,  # Substitute x with g(y)
    2: h_yz    # Substitute y with h(y, z)
}

# Perform the composition using the new compose method
composed_f = fxy.compose(substitution_dict)
print(f'Outer MTF after substitution: f(g(y), h(y, z)) =\\n{composed_f}')

# Define some MTFs to work with
sin_x = mtf.sin(x)
cos_x = mtf.cos(x)

# Addition
sum_tf = sin_x + cos_x
print(f"sin(x) + cos(x) =\\n{sum_tf}")

# Subtraction
diff_tf = sin_x - cos_x
print(f"sin(x) - cos(x) =\\n{diff_tf}")

# Multiplication by a scalar
scaled_sin_x = 2 * sin_x
print(f"2 * sin(x) =\\n{scaled_sin_x}")

# Multiplication of two MTFs
product_tf = x * sin_x
print(f"x * sin(x) =\\n{product_tf}")

# Power of an MTF
squared_x = x**2
print(f"x^2 =\\n{squared_x}")

# Comparing MTFs for equality
sin_x_again = mtf.sin(x)
are_equal = (sin_x == sin_x_again)
print(f"Is sin(x) == sin(x) again? {are_equal}")

cos_x_shifted = mtf.cos(x + 0.1)
are_unequal = (mtf.sin(x) == cos_x_shifted)
print(f"Is sin(x) == cos(x + 0.1)? {are_unequal}")

# Accessing the coefficients of an MTF
sin_x_for_coeffs = mtf.sin(x)
coefficients_sin_x = {tuple(exp): coeff for exp, coeff in zip(sin_x_for_coeffs.exponents, sin_x_for_coeffs.coeffs)}
print(f"Coefficients of sin(x):\\n{coefficients_sin_x}")
print("Iterating through coefficients:")
for exponents, coefficient in coefficients_sin_x.items():
    exponents = tuple(int(x) for x in exponents) # convert from np.int64 to int
    print(f"Exponent tuple: {exponents}, Coefficient: {coefficient}")

# Creating a constant MTF
# We will use "mtf" which is an alias for MultivariateTaylorFunction class
constant_tf = mtf.from_constant(5)
print(f"Constant MTF (value 5):\\n{constant_tf}")

# Creating a zero MTF
zero_tf = 0 * x
print(f"Zero MTF:\\n{zero_tf}")

# Taylor expansion of exp(x)
exp_x_tf = mtf.exp(x)
print(f"\nTaylor expansion of exp(x):\n{exp_x_tf}")

# Taylor expansion of gaussian(x) (exp(-x^2))
gaussian_x_tf = mtf.gaussian(x)
print(f"\nTaylor expansion of gaussian(x):\n{gaussian_x_tf}")

# Taylor expansion of sqrt(1+x)
sqrt_x_tf = mtf.sqrt(1+x)
print(f"\nTaylor expansion of sqrt(1+x):\n{sqrt_x_tf}")

# Taylor expansion of log(1+y)
log_y_tf = mtf.log(1+y)
print(f"\nTaylor expansion of log(1+y):\n{log_y_tf}")

# Taylor expansion of arctan(x + 2y + 0.5z)
arctan_x_tf = mtf.arctan(x+2*y+.5*z)
print(f"\nTaylor expansion of arctan(x + 2y + 0.5z):\n{arctan_x_tf}")

# Taylor expansion of 1/(1+x**2+y+.3*z)
fxyz = 1/(1+x**2+y+.3*z)
print(f"\nTaylor expansion of 1/(1+x**2+y+.3*z):\n{fxyz}")

# Recall that 'x' was defined earlier as var(1)

# Since we don't have a direct '1/(1-x)' elementary function, we can try to approximate it
# by using (1-x)**(-1). This might behave differently due to the nature of the expansion.
# A more direct way would be to manually create the coefficients for a few terms.

def manual_geometric_series(order):
    coeffs = {}
    dim = mtf.get_max_dimension()
    for i in range(order + 1):
        exponent = [0] * dim
        exponent[0] = i
        coeffs[tuple(exponent)] = 1.0
    return mtf(coeffs, dimension=dim)

geometric_series_approx = manual_geometric_series(8) # Using max_order = 8
print("Taylor series approximation of 1/(1-x) around x=0:\n", geometric_series_approx)

# Let's evaluate at a point within the radius of convergence (e.g., x = 0.5)
eval_point_within = np.array([0.5,0,0])
result_within = geometric_series_approx.eval(eval_point_within)
exact_within = 1 / (1 - 0.5)
print(f"\nEvaluation at x = 0.5: Approximation = {result_within}, Exact = {exact_within}")

# Let's evaluate at a point outside the radius of convergence (e.g., x = 1.5)
eval_point_outside = np.array([1.5,0,0])
result_outside = geometric_series_approx.eval(eval_point_outside)
exact_outside = 1 / (1 - 1.5)
print(f"Evaluation at x = 1.5: Approximation = {result_outside}, Exact = {exact_outside}")

# You will likely observe that the approximation is good within the radius of convergence
# but may deviate significantly outside it, especially as you move further away.

# import numpy as np
# import matplotlib.pyplot as plt
# from mtflib import var, mtf

# # Values of x to evaluate
# x_values = [0.01, 0.1, 0.2, 0.5]
# num_plots = len(x_values)
# num_rows = (num_plots + 1) // 2
# num_cols = 2

# # Maximum order of Taylor series to consider
# max_order = mtf.get_max_order()
# orders = range(1, max_order + 1)

# # Define the variable for Taylor expansion
# x = var(1)

# # Functions to evaluate
# functions = {
#     "sin(x)": (np.sin, mtf.sin, "red"),
#     "cos(x)": (np.cos, mtf.cos, "blue"),
#     "exp(x)": (np.exp, mtf.exp, "green"),
#     "gaussian(x)": (lambda x: np.exp(-x**2), mtf.gaussian, "purple"),
#     "sqrt(1+x)": (lambda x: np.sqrt(1 + x), lambda var: mtf.sqrt(1 + var), "orange"),
#     "log(1+x)": (lambda x: np.log(1 + x), lambda var: mtf.log(1 + var), "brown"),
#     "arctan(x)": (np.arctan, mtf.arctan, "cyan"),
#     "sinh(x)": (np.sinh, mtf.sinh, "magenta"),
#     "cosh(x)": (np.cosh, mtf.cosh, "lime"),
#     "tanh(x)": (np.tanh, mtf.tanh, "teal"),
#     "arcsin(x)": (np.arcsin, mtf.arcsin, "olive")
# }

# # Create subplots
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
# axes = axes.flatten()

# # Generate plots for each x value
# for i, x_val in enumerate(x_values):
#     accuracy_data = {}
#     for name, (exact_func, taylor_func, color) in functions.items():
#         errors = []
#         for order in orders:
#             mtf.set_max_order(order)
#             taylor_series = taylor_func(x)
#             approximation = taylor_series.eval(np.array([x_val, 0, 0]))
#             try:
#                 exact_value = exact_func(x_val)
#                 absolute_error = np.abs(approximation - exact_value)
#                 errors.append(-np.log10(absolute_error + 1e-18))
#             except ValueError as e:
#                 # Handle cases where the exact function is not defined for x_val
#                 errors.append(np.nan)
#                 print(f"Warning: {e} for {name} at x = {x_val}")

#         accuracy_data[name] = errors

#     # Plotting the results for the current x_val in the corresponding subplot
#     ax = axes[i]
#     for name, errors in accuracy_data.items():
#         color = functions[name][2]
#         ax.plot(orders, errors, marker='o', linestyle='-', color=color, label=name)

#     ax.set_xlabel("Order of Taylor Series")
#     ax.set_ylabel("Logarithm of Accuracy (-log10(Absolute Error))")
#     ax.set_title(f"Accuracy vs. Order at x = {x_val}")
#     ax.grid(True)
#     ax.legend()

# # Remove any unused subplots if the number of x_values is odd
# if num_plots < num_rows * num_cols:
#     for i in range(num_plots, num_rows * num_cols):
#         fig.delaxes(axes[i])

# plt.tight_layout()
# # plt.show()
# mtf.set_max_order(max_order)

# This cell requires sympy to be installed.
# You can install it using: pip install sympy
import sympy as sp
sp.init_printing(use_unicode=True, order='grevlex')

# --- Example with MultivariateTaylorFunction ---
print("--- Example: mtf.sin(x+2*y) ---")
x = var(1)
y = var(2)
mtf_sin = mtf.sin(x + 2*y)
sympy_sin_expr = mtf_sin.symprint()
print("The following is a SymPy object, which will render beautifully in a notebook.")
print(sympy_sin_expr)

# --- Example with ComplexMultivariateTaylorFunction ---
print("--- Example: mtf.exp(i*x) ---")
i = 1j
from mtflib import ComplexMultivariateTaylorFunction
x_complex = ComplexMultivariateTaylorFunction.from_variable(var_index=1, dimension=1)
mtf_complex_exp = mtf.exp(i * x_complex)
sympy_complex_expr = mtf_complex_exp.symprint(symbols=['x'])
print("The following is a SymPy object for a complex function.")
print(sympy_complex_expr)

# --- Example with custom coefficient formatting ---
print("--- Example: Custom coefficient formatting ---")
def custom_formatter(c, p):
    # A simple rational formatter
    if np.iscomplexobj(c):
        return sp.Rational(c.real).limit_denominator(10**p) + sp.I * sp.Rational(c.imag).limit_denominator(10**p)
    else:
        return sp.Rational(c).limit_denominator(10**p)

# Re-initialize for the 2D mtf_sin
x = var(1)
y = var(2)
mtf_sin = mtf.sin(x+2*y)

sympy_custom_format_expr = mtf_sin.symprint(precision=3, coeff_formatter=custom_formatter)
print("The following is a SymPy object with custom rational formatting.")
print(sympy_custom_format_expr)
