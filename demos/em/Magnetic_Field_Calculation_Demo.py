import numpy as np
from mtflib import *
from applications.em.biot_savart import mpi_biot_savart, serial_biot_savart
from applications.em.current_ring import current_ring
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Global MTF Settings ---
initialize_mtf_globals(max_order=6, max_dimension=4)
set_global_etol(1e-16)

# --- Define Variables for MTF ---
x = Var(1)
y = Var(2)
z = Var(3)
u = Var(4)

# --- MPI Setup ---
mpi_installed = True
try:
    from mpi4py import MPI
except ImportError:
    mpi_installed = False
    MPI = None

mu_0_4pi = 1e-7

if mpi_installed:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Running in MPI parallel mode with {size} processes.")
    parallel_mode = True
else:
    print("Running in serial mode (mpi4py not installed).")
    parallel_mode = False
    rank = 0
    size = 1

# --- Example Field Points ---
num_field_points_axis = 3
z_axis_coords = np.linspace(-2, 2, num_field_points_axis)
field_points_axis = np.array([[x, y, zc+z] for zc in z_axis_coords], dtype=object)

# --- Example 1: Current Ring with Specified Center and Axis (element input) ---
print("\n--- Example 1: Current Ring with Specified Center and Axis (Element Input) ---")
ring_radius = 0.4
num_segments_ring = 10
ring_center_point = np.array([0.0, 0.0, 0.0])
ring_axis_direction = np.array([0, 0, 1])

segment_mtfs_ring, element_lengths_ring, direction_vectors_ring = current_ring(
    ring_radius, num_segments_ring, ring_center_point, ring_axis_direction)

if parallel_mode:
    B_field_ring_axis = mpi_biot_savart(segment_mtfs_ring, element_lengths_ring, direction_vectors_ring, field_points_axis)
else:
    B_field_ring_axis = serial_biot_savart(segment_mtfs_ring, element_lengths_ring, direction_vectors_ring, field_points_axis)


for i in range(num_field_points_axis):
    B_field_ring_axis[i] = [integrate(bfld, 4, -1, 1) for bfld in B_field_ring_axis[i]]
    # fmap = mtfarray(list(B_field_ring_axis[i]),['Bx','By','Bz'])
    # print(f'{fmap}\n')

if rank == 0:
    mid_point = num_field_points_axis//2
    print("Magnetic field along axis of rotated ring (Example 1 - Element Input, first point):")
    print(mtfarray(B_field_ring_axis[mid_point]))

    # Bzfldxyz = integrate(B_field_ring_axis[mid_point][2],4,-1,1)
    Bzfldxyz = B_field_ring_axis[mid_point][2]
    Bzfld = Bzfldxyz.substitute_variable(1,0).substitute_variable(2,0)
    print('Bz field along z: \n', Bzfld)

    current = 1

    import math
    analytic_fun_expr = (mu_0_4pi*2*math.pi*(ring_radius**2)*current)/((z**2+ring_radius**2)*sqrt_taylor(z**2+ring_radius**2))
    print('Analytic_fun expression:\n',analytic_fun_expr)

    # --- Coefficient Comparison Table ---
    mtf_exponents = Bzfld.exponents
    mtf_coeffs = Bzfld.coeffs
    mtf_map = {tuple(exp): coeff for exp, coeff in zip(mtf_exponents, mtf_coeffs)}

    analytic_exponents = analytic_fun_expr.exponents
    analytic_coeffs = analytic_fun_expr.coeffs
    analytic_map = {tuple(exp): coeff for exp, coeff in zip(analytic_exponents, analytic_coeffs)}

    all_exponents = set(mtf_map.keys()) | set(analytic_map.keys()) # Union of exponents

    print("\nCoefficient Comparison Table:")
    print("----------------------------------------------------------------------------------")
    print(f"{'Exponent':<15} | {'Bzfld Coeff':<20} | {'Analytic Coeff':<20} | {'Error':<20}") # Wider columns
    print("----------------------------------------------------------------------------------")

    for exponent in sorted(list(all_exponents)): # Iterate through exponents in sorted order
        mtf_coeff_raw = mtf_map.get(exponent, 0.0) # Get MTF coefficient, default to 0 if not present
        analytic_coeff_raw = analytic_map.get(exponent, 0.0) # Get analytic coefficient, default to 0 if not present
        error_raw = mtf_coeff_raw - analytic_coeff_raw

        # Safely convert to float, handling potential NumPy arrays and removing DeprecationWarning
        mtf_coeff = float(mtf_coeff_raw.item()) if isinstance(mtf_coeff_raw, np.ndarray) else float(mtf_coeff_raw)
        analytic_coeff = float(analytic_coeff_raw.item()) if isinstance(analytic_coeff_raw, np.ndarray) else float(analytic_coeff_raw)
        error = float(error_raw.item()) if isinstance(error_raw, np.ndarray) else float(error_raw)


        # Convert exponent tuple elements to standard Python int before string conversion
        exponent_list_int = [int(val) for val in exponent] # Convert np.int64 to int
        exponent_str = str(tuple(exponent_list_int)) # Convert tuple of ints to string

        print(f"{exponent_str:<15} | {mtf_coeff:<20.8e} | {analytic_coeff:<20.8e} | {error:<20.8e}") # Wider columns and consistent formatting

    print("----------------------------------------------------------------------------------")
