# demo/magtest.py
import numpy as np  # NumPy for numerical operations, especially array handling

# Import necessary components from the MTFLibrary package
from MTFLibrary.taylor_function import initialize_mtf_globals, set_global_etol, convert_to_mtf  # Core MTF functionalities
from MTFLibrary.elementary_functions import cos_taylor, sin_taylor, exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor, sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, arctanh_taylor  # Elementary Taylor functions
from MTFLibrary.MTFExtended import Var  # Variable class for Multivariate Taylor Functions (MTF)
# Import Biot-Savart functions from the EMLibrary subpackage
from MTFLibrary.EMLibrary.biot_savart import mpi_biot_savart, serial_biot_savart  # Biot-Savart calculation functions (MPI and serial versions)
from MTFLibrary.EMLibrary.current_ring import current_ring  # Function to create current ring geometry for Biot-Savart calculation

# --- Global MTF Settings ---
initialize_mtf_globals(max_order=4, max_dimension=4)  # Initialize global settings for MTF calculations
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

# --- MPI Setup (for parallel processing) ---
mpi_installed = True   # Assume MPI is installed initially
try:
    from mpi4py import MPI  # Try to import the mpi4py library for parallel computation
except ImportError:
    mpi_installed = False  # If import fails, MPI is not installed
    MPI = None   # Set MPI to None so we can check for it later

mu_0_4pi = 1e-7   # Permeability of free space / 4pi (constant mu_0 / 4pi in SI units)

if mpi_installed:
    comm = MPI.COMM_WORLD  # Get the MPI communicator for all processes
    rank = comm.Get_rank()  # Get the rank (ID) of the current MPI process
    size = comm.Get_size()  # Get the total number of MPI processes running
    print(f"Running in MPI parallel mode with {size} processes.") # Indicate MPI mode
    parallel_mode = True  # Set flag for parallel mode
else:
    print("Running in serial mode (mpi4py not installed).") # Indicate serial mode
    parallel_mode = False  # Set flag for serial mode
    rank = 0  # For serial execution, treat as rank 0 for output consistency
    size = 1  # For serial, the size is 1 (single process)

# --- Example Field Points (common for all examples to keep it simple) ---
num_field_points_axis = 50  # Number of field points along the z-axis
z_axis_coords = np.linspace(-2, 2, num_field_points_axis)  # Create 50 points along z-axis from -2 to 2
field_points_axis = np.array([[x, y, zc+z] for zc in z_axis_coords]) # Define field points as MTFs, varying z-coordinate along z-axis
                                                                     # x and y are symbolic MTF variables, zc+z creates MTF for z-coordinate

# --- Example 1: Magnetic field of a current ring with specified center and axis (element input) ---
print("\n--- Example 1: Current Ring with Specified Center and Axis (Element Input) ---")
ring_radius = 1.0  # Radius of the current ring
num_segments_ring = 5  # Number of segments to discretize the ring into
ring_center_point = np.array([0.5, 0.5, 0.0]) # Define the center point of the ring
ring_axis_direction = np.array([1, 1, 1]) # Define the direction vector of the ring's axis (normal to the ring plane)

# --- Call current_ring function to generate ring geometry ---
# The current_ring function discretizes a circular current loop into segments.
# It returns information about each segment for Biot-Savart calculation.
segment_mtfs_ring, element_lengths_ring, direction_vectors_ring = current_ring( # Renamed variable here
    ring_radius, num_segments_ring, ring_center_point, ring_axis_direction) # Updated function arguments
#
# Output of current_ring:
# - segment_mtfs_ring: (numpy.ndarray of MTFs) Array of Multivariate Taylor Functions (MTFs). # Updated variable name in comments
#                       Each element in this NumPy array is itself an MTF,
#                       representing the ring segment as a Taylor expansion
#                       around the center point of the segment/element.
#                       Dimension 4 (variable 'u') is used as the parameter for elemental arc length,
#                       with unit scaling in the expansion. This MTF describes the location of points
#                       along the ring segment as a function of this arc length parameter.
# - element_lengths_ring: (numpy.ndarray) Length of each ring segment (dl). These are NumPy numbers,
#                         representing the discretized length of each segment.
# - direction_vectors_ring: (numpy.ndarray) Unit vectors representing the direction of current
#                         flow along each ring segment, tangent to the ring at the center point.
#                         Normalized to length 1. These are NumPy arrays representing vectors.

# Calculate the magnetic field along the defined z-axis field points using the element input method
if parallel_mode:
    # Use MPI parallel Biot-Savart calculation if MPI is available
    B_field_ring_axis = mpi_biot_savart(segment_mtfs_ring, element_lengths_ring, direction_vectors_ring, field_points_axis) # Updated variable name here
else:
    # Use serial Biot-Savart calculation if MPI is not available
    B_field_ring_axis = serial_biot_savart(segment_mtfs_ring, element_lengths_ring, direction_vectors_ring, field_points_axis) # Updated variable name here

# Output results (only on rank 0 in MPI mode to avoid duplicate output)
if rank == 0:
    print("Magnetic field along axis of rotated ring (Example 1 - Element Input, first point):")
    print(B_field_ring_axis[0][0]) # Print the x-component of the B-field at the first field point
    # The following lines are commented out, but show how to print more detailed MTF results if needed:
    # for i in range(min(5, len(field_points_axis))):
    #     print(f"Field Point {field_points_axis[i]}: B-field = {B_field_ring_axis[i][1].__str__}")
    # _=([print(item) for item in fn]) # Example of printing a MTF object (fn is commented out example MTF expression)

# # --- Example of creating and printing a simple MTF expression (commented out) ---
# # fn = x**2+y+z+4*u**4+exp_taylor(x-y**4+z-u+3*v+w**2)
# # print(fn)

# # --- Example of creating MTF arrays and operations (commented out, for potential future examples) ---
# # davecx = np.array([1/(1+x)]*10)
# # davecy = np.array([sin_taylor(y+z+w)]*10)
# # fn = (davecx+davecy)


# # demo/magtest.py
# import numpy as np  # NumPy for numerical operations, especially array handling

# # Import necessary components from the MTFLibrary package
# from MTFLibrary.taylor_function import initialize_mtf_globals, set_global_etol, convert_to_mtf  # Core MTF functionalities
# from MTFLibrary.elementary_functions import cos_taylor, sin_taylor, exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor, sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, arctanh_taylor  # Elementary Taylor functions
# from MTFLibrary.MTFExtended import Var  # Variable class for Multivariate Taylor Functions (MTF)
# # Import Biot-Savart functions from the EMLibrary subpackage
# from MTFLibrary.EMLibrary.biot_savart import mpi_biot_savart, serial_biot_savart  # Biot-Savart calculation functions (MPI and serial versions)
# from MTFLibrary.EMLibrary.current_ring import current_ring  # Function to create current ring geometry for Biot-Savart calculation

# # --- Global MTF Settings ---
# initialize_mtf_globals(max_order=4, max_dimension=4)  # Initialize global settings for MTF calculations
#                                                         # max_order: Maximum order of Taylor series to be used
#                                                         # max_dimension: Maximum number of variables in MTFs
# set_global_etol(1e-16)  # Set global error tolerance for MTF calculations (coefficients smaller than this are truncated)

# # --- Define Variables for MTF ---
# # Create symbolic variables x, y, z, u as Multivariate Taylor Functions (MTFs)
# # These variables will be used to define functions and expressions in terms of MTFs
# x = Var(1)  # Variable 'x' associated with dimension 1
# y = Var(2)  # Variable 'y' associated with dimension 2
# z = Var(3)  # Variable 'z' associated with dimension 3
# u = Var(4)  # Variable 'u' associated with dimension 4 - used as elemental arc length parameter
# # v = Var(5)  # Example of another variable if needed
# # w = Var(6)  # Example of another variable if needed

# # --- MPI Setup (for parallel processing) ---
# mpi_installed = True   # Assume MPI is installed initially
# try:
#     from mpi4py import MPI  # Try to import the mpi4py library for parallel computation
# except ImportError:
#     mpi_installed = False  # If import fails, MPI is not installed
#     MPI = None   # Set MPI to None so we can check for it later

# mu_0_4pi = 1e-7   # Permeability of free space / 4pi (constant mu_0 / 4pi in SI units)

# if mpi_installed:
#     comm = MPI.COMM_WORLD  # Get the MPI communicator for all processes
#     rank = comm.Get_rank()  # Get the rank (ID) of the current MPI process
#     size = comm.Get_size()  # Get the total number of MPI processes running
#     print(f"Running in MPI parallel mode with {size} processes.") # Indicate MPI mode
#     parallel_mode = True  # Set flag for parallel mode
# else:
#     print("Running in serial mode (mpi4py not installed).") # Indicate serial mode
#     parallel_mode = False  # Set flag for serial mode
#     rank = 0  # For serial execution, treat as rank 0 for output consistency
#     size = 1  # For serial, the size is 1 (single process)

# # --- Example Field Points (common for all examples to keep it simple) ---
# num_field_points_axis = 50  # Number of field points along the z-axis
# z_axis_coords = np.linspace(-2, 2, num_field_points_axis)  # Create 50 points along z-axis from -2 to 2
# field_points_axis = np.array([[x, y, zc+z] for zc in z_axis_coords]) # Define field points as MTFs, varying z-coordinate along z-axis
#                                                                      # x and y are symbolic MTF variables, zc+z creates MTF for z-coordinate

# # --- Example 1: Magnetic field of a current ring oriented in an arbitrary direction (using element input) ---
# print("\n--- Example 1: Current Ring in Arbitrary Direction (Element Input) ---")
# ring_radius = 1.0  # Radius of the current ring
# num_segments_ring = 5  # Number of segments to discretize the ring into
# rotation_axis = np.array([0, 1, 0])  # Define the axis of rotation (y-axis in this case)
# rotation_angle = np.pi / 4   # Define the rotation angle (45 degrees or pi/4 radians)

# # --- Call current_ring function to generate ring geometry ---
# # The current_ring function discretizes a circular current loop into segments.
# # It returns information about each segment for Biot-Savart calculation.
# segment_mtfs_ring, element_lengths_ring, direction_vectors_ring = current_ring( # Renamed variable here
#     ring_radius, num_segments_ring, rotation_axis, rotation_angle)
# #
# # Output of current_ring:
# # - segment_mtfs_ring: (numpy.ndarray of MTFs) Array of Multivariate Taylor Functions (MTFs). # Updated variable name in comments
# #                       Each element in this NumPy array is itself an MTF,
# #                       representing the ring segment as a Taylor expansion
# #                       around the center point of the segment/element.
# #                       Dimension 4 (variable 'u') is used as the parameter for elemental arc length,
# #                       with unit scaling in the expansion. This MTF describes the location of points
# #                       along the ring segment as a function of this arc length parameter.
# # - element_lengths_ring: (numpy.ndarray) Length of each ring segment (dl). These are NumPy numbers,
# #                         representing the discretized length of each segment.
# # - direction_vectors_ring: (numpy.ndarray) Unit vectors representing the direction of current
# #                         flow along each ring segment, tangent to the ring at the center point.
# #                         Normalized to length 1. These are NumPy arrays representing vectors.

# # Calculate the magnetic field along the defined z-axis field points using the element input method
# if parallel_mode:
#     # Use MPI parallel Biot-Savart calculation if MPI is available
#     B_field_ring_axis = mpi_biot_savart(segment_mtfs_ring, element_lengths_ring, direction_vectors_ring, field_points_axis) # Updated variable name here
# else:
#     # Use serial Biot-Savart calculation if MPI is not available
#     B_field_ring_axis = serial_biot_savart(segment_mtfs_ring, element_lengths_ring, direction_vectors_ring, field_points_axis) # Updated variable name here

# # Output results (only on rank 0 in MPI mode to avoid duplicate output)
# if rank == 0:
#     print("Magnetic field along axis of rotated ring (Example 1 - Element Input, first point):")
#     print(B_field_ring_axis[0][0]) # Print the x-component of the B-field at the first field point
#     # The following lines are commented out, but show how to print more detailed MTF results if needed:
#     # for i in range(min(5, len(field_points_axis))):
#     #     print(f"Field Point {field_points_axis[i]}: B-field = {B_field_ring_axis[i][1].__str__}")
#     # _=([print(item) for item in fn]) # Example of printing a MTF object (fn is commented out example MTF expression)

# # # --- Example of creating and printing a simple MTF expression (commented out) ---
# # # fn = x**2+y+z+4*u**4+exp_taylor(x-y**4+z-u+3*v+w**2)
# # # print(fn)

# # # --- Example of creating MTF arrays and operations (commented out, for potential future examples) ---
# # # davecx = np.array([1/(1+x)]*10)
# # # davecy = np.array([sin_taylor(y+z+w)]*10)
# # # fn = (davecx+davecy)