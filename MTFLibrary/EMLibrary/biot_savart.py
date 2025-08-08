import numpy as np
from MTFLibrary import *

mpi_installed = True  # Assume MPI is installed initially
try:
    from mpi4py import MPI
except ImportError:
    mpi_installed = False
    MPI = None  # Set MPI to None so we can check for it later

mu_0_4pi = 1e-7 # Define mu_0_4pi if it's not already globally defined


def _numpy_biot_savart_core(source_points, dl_vectors, field_points, order=None):
    """
    Core vectorized Biot-Savart calculation.
    """
    source_points_reshaped = source_points[:, np.newaxis, :]
    field_points_reshaped = field_points[np.newaxis, :, :]

    r_vectors = field_points_reshaped - source_points_reshaped
    r_squared = np.sum(r_vectors**2, axis=2)

    # Avoid division by zero at the source point location
    r_squared = np.where(r_squared == 0, 1e-18, r_squared)

    dl_vectors_reshaped = dl_vectors[:, np.newaxis, :]
    cross_products = np.cross(dl_vectors_reshaped, r_vectors, axis=2)

    # Calculate 1/r^3 for magnitude scaling.
    # This is done as 1/r^2 * 1/r to work with MTF ufunc overrides.
    inv_r_cubed = np.reciprocal(r_squared) * r_squared**(-0.5)
    inv_r_cubed_expanded = np.expand_dims(inv_r_cubed, axis=2)

    dB_contributions = (mu_0_4pi * cross_products) * inv_r_cubed_expanded

    B_field = np.sum(dB_contributions, axis=0)

    # If an order is specified, truncate each MTF in the result
    if order is not None:
        for i in range(B_field.shape[0]):
            for j in range(B_field.shape[1]):
                if isinstance(B_field[i, j], MultivariateTaylorFunctionBase):
                    B_field[i, j] = B_field[i, j].truncate(order)

    return B_field


def numpy_biot_savart(element_centers, element_lengths, element_directions, field_points, order=None):
    """
    NumPy vectorized Biot-Savart calculation using element inputs.

    Calculates the magnetic field at specified field points due to a set of current
    elements, using NumPy vectorization for efficiency. This function is designed for
    serial (single-processor) execution and takes element-based descriptions of the
    current source.

    Args:
        element_centers (numpy.ndarray): (N, 3) array of center coordinates of current elements.
            Each row represents the (x, y, z) coordinates of the center of a current element.
        element_lengths (numpy.ndarray): (N,) array of lengths of current elements (dl). Each
            element represents the length of the corresponding current element.
        element_directions (numpy.ndarray): (N, 3) array of unit vectors representing the
            direction of current flow for each element. Each row is a unit vector.
        field_points (numpy.ndarray): (M, 3) array of field point coordinates. Each row
            represents the (x, y, z) coordinates where the magnetic field is to be calculated.

    Returns:
        numpy.ndarray: (M, 3) array of magnetic field vectors at each field point. Each row
            is a vector representing the (Bx, By, Bz) components of the magnetic field
            at the corresponding field point.

    Raises:
        ValueError: If input arrays do not have the expected dimensions or shapes.

    Example:
        >>> element_centers = np.array([[0, 0, 0], [1, 0, 0]])
        >>> element_lengths = np.array([0.1, 0.1])
        >>> element_directions = np.array([[1, 0, 0], [0, 1, 0]])
        >>> field_points = np.array([[0, 1, 0], [1, 1, 0]])
        >>> B_field = numpy_biot_savart(element_centers, element_lengths, element_directions, field_points)
        >>> print(B_field)
        [[ 0.00000000e+00  0.00000000e+00  1.00000000e-08]
         [ 0.00000000e+00  0.00000000e+00  5.00000000e-09]]
    """
    element_centers = np.array(element_centers)
    element_lengths = np.array(element_lengths)
    element_directions = np.array(element_directions)
    field_points = np.array(field_points)

    if element_centers.ndim != 2 or element_centers.shape[1] != 3:
        raise ValueError("element_centers must be a NumPy array of shape (N, 3)")
    if element_lengths.ndim != 1 or element_lengths.shape[0] != element_centers.shape[0]:
        raise ValueError("element_lengths must be a NumPy array of shape (N,) and have the same length as element_centers")
    if element_directions.ndim != 2 or element_directions.shape[1] != 3 or element_directions.shape[0] != element_centers.shape[0]:
        raise ValueError("element_directions must be a NumPy array of shape (N, 3) and have the same length as element_centers")
    if field_points.ndim != 2 or field_points.shape[1] != 3:
        raise ValueError("field_points must be a NumPy array of shape (M, 3)")

    source_points = element_centers  # Center points are used as source points
    # The 0.5 factor is necessary for the MTF integration workflow, where the
    # result is integrated over a parameter that ranges from -1 to 1 (an interval of length 2).
    dl_vectors = 0.5 * element_lengths[:, np.newaxis] * element_directions  # dl_vector = dl * direction

    return _numpy_biot_savart_core(source_points, dl_vectors, field_points, order)


def mpi_biot_savart(element_centers, element_lengths, element_directions, field_points, order=None):
    """
    Parallel Biot-Savart calculation using mpi4py with element inputs.

    Computes the magnetic field in parallel using MPI (mpi4py library). This function
    distributes the calculation of the magnetic field at different field points across
    multiple MPI processes to speed up computation.

    Note:
        - Requires `mpi4py` to be installed. If not installed, it will raise an ImportError.
        - Must be run in an MPI environment (e.g., using `mpiexec` or `mpirun`).
        - Input arrays `element_centers`, `element_lengths`, and `element_directions` are
          assumed to be the complete datasets and are broadcasted to all MPI processes.
        - `field_points` are distributed among processes. The result is gathered on rank 0.

    Args:
        element_centers (numpy.ndarray): (N, 3) array of center coordinates of current elements.
            (Broadcasted to all MPI processes).
        element_lengths (numpy.ndarray): (N,) array of lengths of current elements (dl).
            (Broadcasted to all MPI processes).
        element_directions (numpy.ndarray): (N, 3) array of unit vectors for current directions.
            (Broadcasted to all MPI processes).
        field_points (numpy.ndarray): (M, 3) array of field point coordinates.
            (Distributed across MPI processes).

    Returns:
        numpy.ndarray or None:
            On MPI rank 0: (M, 3) array of magnetic field vectors at each field point.
            On MPI ranks > 0: None.

    Raises:
        ImportError: if mpi4py is not installed.

    Example (Run in MPI environment, e.g., `mpiexec -n 4 python your_script.py`):
        >>> import numpy as np
        >>> from MTFLibrary.EMLibrary.biot_savart import mpi_biot_savart
        >>> element_centers = np.array([[0, 0, 0], [1, 0, 0]])
        >>> element_lengths = np.array([0.1, 0.1])
        >>> element_directions = np.array([[1, 0, 0], [0, 1, 0]])
        >>> field_points = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0]]) # More field points for MPI to distribute
        >>> B_field = mpi_biot_savart(element_centers, element_lengths, element_directions, field_points)
        >>> if MPI.COMM_WORLD.Get_rank() == 0: # Only rank 0 has the full result
        >>>     print(B_field)
    """
    if not mpi_installed:
        raise ImportError("mpi4py is not installed, cannot run in MPI mode.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    element_centers = np.array(element_centers)
    element_lengths = np.array(element_lengths)
    element_directions = np.array(element_directions)
    field_points = np.array(field_points)

    num_field_points = field_points.shape[0]
    chunk_size = num_field_points // size
    remainder = num_field_points % size
    start_index = rank * chunk_size + min(rank, remainder)
    end_index = start_index + chunk_size + (1 if rank < remainder else 0)
    local_field_points = field_points[start_index:end_index]

    local_B_field = numpy_biot_savart(element_centers, element_lengths, element_directions, local_field_points)  # Use serial numpy version for local computation

    all_B_field_chunks = comm.gather(local_B_field, root=0)

    if rank == 0:
        B_field = np.concatenate(all_B_field_chunks, axis=0)
        return B_field
    else:
        return None


def serial_biot_savart(element_centers, element_lengths, element_directions, field_points, order=None):
    """
    Serial Biot-Savart calculation with element inputs.

    Computes the magnetic field serially (non-parallel), taking current element
    center points, lengths, and directions as input. This function is suitable for
    single-processor execution and serves as a serial counterpart to the
    `mpi_biot_savart` function.

    Args:
        element_centers (numpy.ndarray): (N, 3) array of center coordinates of current elements.
        element_lengths (numpy.ndarray): (N,) array of lengths of current elements (dl).
        element_directions (numpy.ndarray): (N, 3) array of unit vectors for current directions.
        field_points (numpy.ndarray): (M, 3) array of field point coordinates.
        order (int, optional): The maximum order of the Taylor series to compute.
                               If None, the global max order is used. Defaults to None.

    Returns:
        numpy.ndarray: (M, 3) array of magnetic field vectors at each field point.

    Example:
        >>> element_centers = np.array([[0, 0, 0], [1, 0, 0]])
        >>> element_lengths = np.array([0.1, 0.1])
        >>> element_directions = np.array([[1, 0, 0], [0, 1, 0]])
        >>> field_points = np.array([[0, 1, 0], [1, 1, 0]])
        >>> B_field = serial_biot_savart(element_centers, element_lengths, element_directions, field_points, order=0)
        >>> print(B_field)
        [[ 0.00000000e+00  0.00000000e+00  1.00000000e-08]
         [ 0.00000000e+00  0.00000000e+00  5.00000000e-09]]
    """
    return numpy_biot_savart(element_centers, element_lengths, element_directions, field_points, order=order)
