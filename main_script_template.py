import numpy as np # Added import - v9.2, v9.4
import math
from variables import Var
from taylor_function import MultivariateTaylorFunction, set_global_max_order
from elementary_functions import cos_taylor, sin_taylor, exp_taylor, sqrt_taylor, log_taylor, arctan_taylor

# def main():
# Set a lower global max order for cleaner output (optional)
set_global_max_order(20)

dimension = 2 # Example dimension

# Define variables
x = Var(1, dimension) # Variable 'x' (var_1) in 2D
y = Var(2, dimension) # Variable 'y' (var_2) in 2D

print("--- Taylor Package Example Script ---")

# --- Example 1: Basic Arithmetic Operations ---
print("\n--- Example 1: Basic Arithmetic ---")
f1 = 2*x + 3*y - 5
print("f1 = 2*x + 3*y - 5:")
print(f1) # Using default __str__ which calls print_tabular
print(f"Evaluate f1 at [0.5, 0.1]: {f1.evaluate([0.5, 0.1])}")

f2 = f1 * (x + 1) # Multiplication
print("\nf2 = f1 * (x + 1):")
print(f2) # Using default __str__ which calls print_tabular

f3 = f1 / 2.0 # Division by scalar
print("\nf3 = f1 / 2.0:")
print(f3) # Using default __str__ which calls print_tabular


# --- Example 2: Non-negative Integer Exponentiation ---
print("\n--- Example 2: Exponentiation (non-negative integer) ---")
f4 = (x + y)**2 # Exponentiation to power 2
print("f4 = (x + y)**2:")
print(f4) # Using default __str__ which calls print_tabular

f5 = (x**3) - (y**0) # Exponentiation to powers 3 and 0
print("\nf5 = (x**3) - (y**0):") # y**0 is constant 1
print(f5) # Using default __str__ which calls print_tabular


# --- Example 3: Differentiation and Integration ---
print("\n--- Example 3: Differentiation & Integration ---")
f6 = 6*x**2*y + 2*x + 1 # Example function
print("f6 = 6*x**2*y + 2*x + 1:")
print(f6) # Using default __str__ which calls print_tabular

f6_dx = f6.derivative(wrt_variable_id=1) # Derivative w.r.t x
print("\nDerivative of f6 w.r.t x:")
print(f6_dx) # Using default __str__ which calls print_tabular

f6_dy = f6.derivative(wrt_variable_id=2) # Derivative w.r.t y
print("\nDerivative of f6 w.r.t y:")
print(f6_dy) # Using default __str__ which calls print_tabular

f6_int_x = f6.integrate(wrt_variable_id=1, integration_constant=0) # Integral w.r.t x
print("\nIntegral of f6 w.r.t x (constant=0):")
print(f6_int_x) # Using default __str__ which calls print_tabular

f6_int_y = f6.integrate(wrt_variable_id=2, integration_constant=5) # Integral w.r.t y with constant
print("\nIntegral of f6 w.r.t y (constant=5):")
print(f6_int_y) # Using default __str__ which calls print_tabular


# --- Example 4: Function Composition ---
print("\n--- Example 4: Function Composition ---")
f7_base = x*y # Base function f(x,y) = xy
print("Base function f7_base = x*y:")
print(f7_base) # Using default __str__ which calls print_tabular

substitution_functions = {
    x: x + 1, # Substitute x with (x+1) - also a Taylor function
    y: MultivariateTaylorFunction(coefficients={(0, 2): np.array(1.0)}, dimension=dimension, expansion_point=np.zeros(dimension)) # Explicit MTF for y**2 - CORRECTED v9.4, order removed to use global default
}
f7_composed = f7_base.compose(substitution_functions) # f((x+1), (y**2)) = (x+1)*(y**2)
print("\nComposed function f7_composed = f7_base.compose({x: x+1, y: y**2}):")
print(f7_composed) # Using default __str__ which calls print_tabular


# --- Example 5: Elementary Functions (Taylor Expansions) ---
print("\n--- Example 5: Elementary Functions ---")
f_cos_x = cos_taylor(x) # Taylor expansion of cos(x) - order removed to use global default
print("cos(x) Taylor expansion:")
print(f_cos_x) # Using default __str__ which calls print_tabular

f_sin_y = sin_taylor(y) # Taylor expansion of sin(y) - order removed to use global default
print("\nsin(y) Taylor expansion:")
print(f_sin_y) # Using default __str__ which calls print_tabular

f_exp_sum = exp_taylor(x+y) # Taylor expansion of exp(x+y) - order removed to use global default
print("\nexp(x+y) Taylor expansion:")
print(f_exp_sum) # Using default __str__ which calls print_tabular

f_sqrt_1_plus_x = sqrt_taylor(x) # Taylor expansion of sqrt(1+x) - order removed to use global default
print("\msqrt(1+x) Taylor expansion:")
print(f_sqrt_1_plus_x) # Using default __str__ which calls print_tabular

f_log_1_plus_y = log_taylor(y) # Taylor expansion of ln(1+y) - order removed to use global default
print("\nln(1+y) Taylor expansion:")
print(f_log_1_plus_y) # Using default __str__ which calls print_tabular

f_arctan_x = arctan_taylor(x) # Taylor expansion of arctan(x) - order removed to use global default
print("\narctan(x) Taylor expansion:")
print(f_arctan_x) # Using default __str__ which calls print_tabular


print("\n--- End of Examples ---")

fn = sin_taylor(math.pi/6+x+y) # Order removed to use global default
print(fn) # Using default __str__ which calls print_tabular


# if __name__ == "__main__":
#     main()