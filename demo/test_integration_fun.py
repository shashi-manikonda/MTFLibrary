# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:37:36 2025

@author: manik
"""

# demo/magtest.py
import numpy as np  # NumPy for numerical operations, especially array handling
from MTFLibrary import *


# --- Global MTF Settings ---
initialize_mtf_globals(max_order=8, max_dimension=4)  # Initialize global settings for MTF calculations
                                                        # max_order: Maximum order of Taylor series to be used
                                                        # max_dimension: Maximum number of variables in MTFs
set_global_etol(1e-16)  # Set global error tolerance for MTF calculations (coefficients smaller than this are truncated)

# --- Define Variables for MTF ---
# Create symbolic variables x, y, z, u as Multivariate Taylor Functions (MTFs)
# These variables will be used to define functions and expressions in terms of MTFs
x = Var(1)  # Variable 'x' associated with dimension 1
y = Var(2)  # Variable 'y' associated with dimension 2
z = Var(3)  # Variable 'z' associated with dimension 3
u = Var(4)  # Variable 'u' associated with dimension 4 - used as elemental arc length parameter
# v = Var(5)  # Example of another variable if needed
# w = Var(6)  # Example of another variable if needed

# fx = 2+x + x**2+2*x*y+3*x**2*z
# print(f'Before integration:\n{fx}')
# ifx=integrate(fx,1,0,.5)
# print(f'After definite integration Integrate(fx, 2, 0, .5):\n{ifx}')


fxy = exp_taylor(1+x+y)
print(f'Before integration:\n{fxy}')
ifxy=integrate(fxy,1,-1,1)
print(f'After definite integration Integrate(fx,1,-1,1):\n{ifxy}')


fxy = cos_taylor(0.33*x**2+0.127*y)
print(f'Before integration:\n{fxy}')
ifxy=integrate(fxy,1,-1,1)
print(f'After definite integration Integrate(fx,1,-1,1):\n{ifxy}')

# initialize_mtf_globals(max_order=3, max_dimension=2)
# set_global_etol(1e-10)

# # Provide var_names when initializing Var objects or MTFs:
# x = Var(1) # Var is now directly accessible within MTFExtended.py, no var_name
# y = Var(2) # Var is now directly accessible within MTFExtended.py, no var_name

# # Example MTF: f(x, y) = xy, now without var_names
# coefficients_definite_int = {
#     (1, 1): np.array([1.0]),
#     (0, 0, 0): np.array([0.0])
# }
# mtf_definite_integrate = MultivariateTaylorFunction(coefficients_definite_int, dimension=2) # No var_names
# print("Original MTF to Definite Integrate (f(x,y) = xy):")
# mtf_definite_integrate.print_tabular()

# # Definite Integration wrt x (dimension 1) from 0 to 2
# definite_integral_mtf_x = integrate(mtf_definite_integrate, 1, lower_limit=0.0, upper_limit=2.0)
# print("\nDefinite Integral wrt x from 0 to 2:")
# definite_integral_mtf_x.print_tabular()
# # Expected: Integrate (xy) dx from 0 to 2 = [(1/2)x^2*y] from 0 to 2 = (1/2)*(2^2)*y - 0 = 2y

# # Definite Integration wrt y (dimension 2) from 1 to 3
# definite_integral_mtf_y = integrate(mtf_definite_integrate, 2, lower_limit=1.0, upper_limit=3.0)
# print("\nDefinite Integral wrt y from 1 to 3:")
# definite_integral_mtf_y.print_tabular()
# # Expected: Integrate (xy) dy from 1 to 3 = [x*(1/2)y^2] from 1 to 3 = x*(1/2)*(3^2) - x*(1/2)*(1^2) = 4x

# print("\nOriginal Max Order:", get_global_max_order())

# # Example with a constant term and x^2 term, without var_names
# coefficients_const_x2 = {
#     (0, 0): np.array([5.0]),
#     (2, 0): np.array([2.0]),
#     (0, 0, 0): np.array([0.0])
# }
# mtf_const_x2 = MultivariateTaylorFunction(coefficients_const_x2, dimension=2) # No var_names
# print("\nOriginal MTF to Definite Integrate (f(x,y) = 2x^2 + 5):")
# mtf_const_x2.print_tabular()
# definite_integral_mtf_x_const_x2 = integrate(mtf_const_x2, 1, lower_limit=-1.0, upper_limit=1.0)
# print("\nDefinite Integral wrt x from -1 to 1:")
# definite_integral_mtf_x_const_x2.print_tabular()
# # Expected: Integrate (2x^2 + 5) dx from -1 to 1 = [(2/3)x^3 + 5x] from -1 to 1 = [(2/3)*(1)^3 + 5*(1)] - [(2/3)*(-1)^3 + 5*(-1)] = 4/3 + 10 = 34/3 = ~11.333
# # In terms of MTF in y (dimension 2): constant term ~ 11.333


