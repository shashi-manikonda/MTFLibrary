# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libcpp.map cimport map
from libcpp.vector cimport vector

# Define C++ types for Cython
ctypedef vector[int] vector_int
ctypedef map[vector_int, double] cpp_map

@cython.boundscheck(False)
@cython.wraparound(False)
def add_mtf_cython(np.ndarray[np.int32_t, ndim=2] exps1,
                   np.ndarray[np.float64_t, ndim=1] coeffs1,
                   np.ndarray[np.int32_t, ndim=2] exps2,
                   np.ndarray[np.float64_t, ndim=1] coeffs2):
    """
    Adds two MTF objects using Cython with a C++ map for aggregation.
    """
    cdef cpp_map combined_coeffs
    cdef int i, j
    cdef int n_terms1 = exps1.shape[0]
    cdef int n_terms2 = exps2.shape[0]
    cdef int dimension = exps1.shape[1] if n_terms1 > 0 else (exps2.shape[1] if n_terms2 > 0 else 0)
    cdef vector_int exp_vec

    # Process the first polynomial
    for i in range(n_terms1):
        exp_vec.clear()
        for j in range(dimension):
            exp_vec.push_back(exps1[i, j])
        combined_coeffs[exp_vec] = coeffs1[i]

    # Process the second polynomial
    for i in range(n_terms2):
        exp_vec.clear()
        for j in range(dimension):
            exp_vec.push_back(exps2[i, j])
        # This will add if the key exists, or insert if it does not
        combined_coeffs[exp_vec] += coeffs2[i]

    # Convert back to NumPy arrays
    if combined_coeffs.empty():
        return np.empty((0, dimension), dtype=np.int32), np.empty((0,), dtype=np.float64)

    cdef Py_ssize_t size = combined_coeffs.size()
    cdef np.ndarray[np.int32_t, ndim=2] new_exponents = np.empty((size, dimension), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] new_coeffs = np.empty(size, dtype=np.float64)

    cdef int idx = 0
    for exp_vec, coeff in combined_coeffs:
        for j in range(dimension):
            new_exponents[idx, j] = exp_vec[j]
        new_coeffs[idx] = coeff
        idx += 1

    return new_exponents, new_coeffs

@cython.boundscheck(False)
@cython.wraparound(False)
def multiply_mtf_cython(np.ndarray[np.int32_t, ndim=2] exps1,
                        np.ndarray[np.float64_t, ndim=1] coeffs1,
                        np.ndarray[np.int32_t, ndim=2] exps2,
                        np.ndarray[np.float64_t, ndim=1] coeffs2):
    """
    Multiplies two MTF objects using Cython with a C++ map for aggregation.
    """
    cdef cpp_map combined_coeffs
    cdef int i, j, k
    cdef int n_terms1 = exps1.shape[0]
    cdef int n_terms2 = exps2.shape[0]

    if n_terms1 == 0 or n_terms2 == 0:
        return np.empty((0, 0), dtype=np.int32), np.empty((0,), dtype=np.float64)

    cdef int dimension = exps1.shape[1]
    cdef vector_int new_exp_vec

    for i in range(n_terms1):
        for j in range(n_terms2):
            new_exp_vec.clear()
            for k in range(dimension):
                new_exp_vec.push_back(exps1[i, k] + exps2[j, k])

            new_coeff = coeffs1[i] * coeffs2[j]
            combined_coeffs[new_exp_vec] += new_coeff

    # Convert back to NumPy arrays
    cdef Py_ssize_t size = combined_coeffs.size()
    cdef np.ndarray[np.int32_t, ndim=2] new_exponents = np.empty((size, dimension), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] new_coeffs = np.empty(size, dtype=np.float64)

    cdef int idx = 0
    for exp_vec, coeff in combined_coeffs:
        for j in range(dimension):
            new_exponents[idx, j] = exp_vec[j]
        new_coeffs[idx] = coeff
        idx += 1

    return new_exponents, new_coeffs

@cython.boundscheck(False)
@cython.wraparound(False)
def compose_one_dim_cython(np.ndarray[np.int32_t, ndim=2] outer_exps,
                           np.ndarray[np.float64_t, ndim=1] outer_coeffs,
                           np.ndarray[np.int32_t, ndim=2] inner_exps,
                           np.ndarray[np.float64_t, ndim=1] inner_coeffs,
                           int inner_dimension):
    """
    Performs function composition self(other_mtf(x)) using Cython.
    """
    cdef np.ndarray[np.int32_t, ndim=1] outer_orders = np.sum(outer_exps, axis=1)

    cdef int max_order = 0
    if outer_orders.shape[0] > 0:
        max_order = np.max(outer_orders)

    # Pre-calculate powers of the inner MTF
    cdef list powers_of_inner_mtf = [(np.array([[0] * inner_dimension], dtype=np.int32), np.array([1.0], dtype=np.float64))]
    if max_order > 0:
        current_power_exps, current_power_coeffs = inner_exps.copy(), inner_coeffs.copy()
        powers_of_inner_mtf.append((current_power_exps, current_power_coeffs))
        for i in range(1, max_order):
            current_power_exps, current_power_coeffs = multiply_mtf_cython(current_power_exps, current_power_coeffs, inner_exps, inner_coeffs)
            powers_of_inner_mtf.append((current_power_exps, current_power_coeffs))

    # Start with a zero MTF
    cdef np.ndarray[np.int32_t, ndim=2] composed_exps = np.empty((0, inner_dimension), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] composed_coeffs = np.empty((0,), dtype=np.float64)

    for i in range(outer_exps.shape[0]):
        order = outer_orders[i]
        coeff = outer_coeffs[i]

        if coeff == 0:
            continue

        if order == 0:
            const_exp = np.array([[0] * inner_dimension], dtype=np.int32)
            const_coeff = np.array([coeff], dtype=np.float64)
            composed_exps, composed_coeffs = add_mtf_cython(composed_exps, composed_coeffs, const_exp, const_coeff)
        else:
            term_exps, term_coeffs = powers_of_inner_mtf[order]

            # Scale by the outer coefficient
            scaled_term_coeffs = term_coeffs * coeff

            composed_exps, composed_coeffs = add_mtf_cython(composed_exps, composed_coeffs, term_exps, scaled_term_coeffs)

    return composed_exps, composed_coeffs
