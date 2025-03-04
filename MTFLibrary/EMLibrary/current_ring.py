import numpy as np
from ..taylor_function import initialize_mtf_globals, set_global_etol, convert_to_mtf
from ..elementary_functions import cos_taylor, sin_taylor, exp_taylor, gaussian_taylor, sqrt_taylor, log_taylor, arctan_taylor, sinh_taylor, cosh_taylor, tanh_taylor, arcsin_taylor, arccos_taylor, arctanh_taylor
from ..MTFExtended import Var

def current_ring(ring_radius, num_segments_ring, ring_center_point, ring_axis_direction):
    """
    Generates MTF representations for segments of a current ring defined by its center point
    and axis direction.

    Args:
        ring_radius (float): Radius of the current ring.
        num_segments_ring (int): Number of segments to discretize the ring into.
        ring_center_point (numpy.ndarray): (3,) array defining the center coordinates (x, y, z) of the ring.
        ring_axis_direction (numpy.ndarray): (3,) array defining the direction vector of the ring's axis
                                            (normal to the plane of the ring).

    Returns:
        tuple: A tuple containing:
            - segment_mtfs_ring (numpy.ndarray): (N,) array of MTFs, where each MTF represents a ring segment
                                                as a Taylor expansion around the segment's center point.
            - element_lengths_ring (numpy.ndarray): (N,) array of lengths of each ring segment (dl).
            - direction_vectors_ring (numpy.ndarray): (N, 3) array of unit vectors representing the direction
                                                    of current flow for each segment.
    """
    x = Var(1)
    y = Var(2)
    z = Var(3)
    u = Var(4)

    d_phi = 2 * np.pi / num_segments_ring
    ring_axis_direction_unit = ring_axis_direction / np.linalg.norm(ring_axis_direction) # Normalize the axis direction

    # Rotation matrix to align z-axis with ring_axis_direction
    def rotation_matrix_align_vectors(v1, v2):
        """Generates rotation matrix to rotate vector v1 to align with v2."""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        v_cross = np.cross(v1_u, v2_u)
        if np.allclose(v_cross, 0): # Vectors are parallel or anti-parallel, no rotation needed or rotation by 180 degree
            if np.dot(v1_u, v2_u) < 0: # Anti-parallel, rotate by 180 degree around x-axis for example
                return rotation_matrix(np.array([1, 0, 0]), np.pi)
            else: # Parallel, no rotation needed
                return np.eye(3)

        rotation_axis = v_cross / np.linalg.norm(v_cross)
        rotation_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) # Clip to handle potential floating point inaccuracies

        return rotation_matrix(rotation_axis, rotation_angle)


    def rotation_matrix(axis, angle):
        """Rotation matrix about arbitrary axis using quaternion parameters."""
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle/2)
        b,c,d = -axis*np.sin(angle/2)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, bd, cd = b*c, b*d, c*d
        ad, ac, ab = a*d, a*c, a*b
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    # Rotation to align z-axis with ring_axis_direction
    rotation_align_z_axis = rotation_matrix_align_vectors(np.array([0, 0, 1.0]), ring_axis_direction_unit)

    segment_mtfs_ring = []
    element_lengths_ring = []
    direction_vectors_ring = []

    for i in range(num_segments_ring):
        phi = (i + 0.5 + 0.5*u) * d_phi # Center point at midpoint of segment
        # Base ring in xy-plane
        x_center = ring_radius * cos_taylor(phi)
        y_center = ring_radius * sin_taylor(phi)
        z_center = 0.0

        # Rotate and translate center point
        center_point_rotated = rotation_align_z_axis @ np.array([x_center, y_center, z_center])
        center_point_translated = center_point_rotated + ring_center_point # Translate to the desired center
        segment_mtfs_ring.append(center_point_translated)

        element_lengths_ring.append(ring_radius * d_phi) # Arc length is approx. dl = r * d_phi

        # Tangent direction at center point (for base ring in xy-plane):
        direction_base = np.array([-sin_taylor(phi), cos_taylor(phi), 0]) # Tangent in xy-plane
        direction_rotated = rotation_align_z_axis @ direction_base
        # MTF Norm calculation: sqrt(Bx^2 + By^2 + Bz^2) using MTF operations
        norm_mtf_squared = direction_rotated[0]**2 + direction_rotated[1]**2 + direction_rotated[2]**2
        norm_mtf = sqrt_taylor(norm_mtf_squared)

        # Normalize direction_rotated by the MTF norm
        direction_normalized_mtf = direction_rotated / norm_mtf  # Element-wise division of MTF array by MTF scalar (should work due to __array_ufunc__)
        direction_vectors_ring.append(direction_normalized_mtf)

    return np.array(segment_mtfs_ring), np.array(element_lengths_ring), np.array(direction_vectors_ring)

