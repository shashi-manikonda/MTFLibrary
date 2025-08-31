import numpy as np
from mtflib import MultivariateTaylorFunction
from src.applications.em.biot_savart import serial_biot_savart

def assert_mtf_arrays_equal(arr1, arr2):
    """Helper function to assert that two arrays of MTF objects are equal."""
    assert arr1.shape == arr2.shape
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            mtf1 = arr1[i, j].copy()
            mtf2 = arr2[i, j].copy()
            mtf1._cleanup_after_operation()
            mtf2._cleanup_after_operation()

            # Sort by exponents to ensure consistent order
            sort_indices1 = np.lexsort(mtf1.exponents.T)
            mtf1.exponents = mtf1.exponents[sort_indices1]
            mtf1.coeffs = mtf1.coeffs[sort_indices1]

            sort_indices2 = np.lexsort(mtf2.exponents.T)
            mtf2.exponents = mtf2.exponents[sort_indices2]
            mtf2.coeffs = mtf2.coeffs[sort_indices2]

            assert np.allclose(mtf1.coeffs, mtf2.coeffs)
            assert np.array_equal(mtf1.exponents, mtf2.exponents)


def test_cpp_backend_consistency():
    """
    Tests that the C++ backend produces the same results as the Python backend.
    """
    # Initialize MTF for Python backend
    MultivariateTaylorFunction.initialize_mtf(max_order=4, max_dimension=4, implementation='python')

    element_centers = np.array([[MultivariateTaylorFunction.from_constant(0), 0, 0], [1, 0, 0]], dtype=object)
    element_lengths = np.array([0.1, 0.1])
    element_directions = np.array([[1, 0, 0], [0, 1, 0]])
    field_points = np.array([[0, 1, 0], [1, 1, 0]])

    # Run with Python backend
    B_field_py = serial_biot_savart(element_centers, element_lengths, element_directions, field_points)

    # Initialize MTF for C++ backend
    MultivariateTaylorFunction.initialize_mtf(max_order=4, max_dimension=4, implementation='cpp')

    # Run with C++ backend
    B_field_cpp = serial_biot_savart(element_centers, element_lengths, element_directions, field_points)

    # Compare the results
    assert_mtf_arrays_equal(B_field_py, B_field_cpp)
