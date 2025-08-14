#include "mtf_ops.hpp"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

std::pair<py::array_t<int32_t>, py::array_t<double>> add_mtf_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps1,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs1,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps2,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs2) {

    auto e1 = exps1.unchecked<2>();
    auto c1 = coeffs1.unchecked<1>();
    auto e2 = exps2.unchecked<2>();
    auto c2 = coeffs2.unchecked<1>();

    ssize_t n_terms1 = e1.shape(0);
    ssize_t n_terms2 = e2.shape(0);
    ssize_t dimension = (n_terms1 > 0) ? e1.shape(1) : ((n_terms2 > 0) ? e2.shape(1) : 0);

    std::unordered_map<std::vector<int>, double, VectorHasher> combined_coeffs;

    for (ssize_t i = 0; i < n_terms1; ++i) {
        std::vector<int> exp_vec(dimension);
        for (ssize_t j = 0; j < dimension; ++j) {
            exp_vec[j] = e1(i, j);
        }
        combined_coeffs[exp_vec] = c1(i);
    }

    for (ssize_t i = 0; i < n_terms2; ++i) {
        std::vector<int> exp_vec(dimension);
        for (ssize_t j = 0; j < dimension; ++j) {
            exp_vec[j] = e2(i, j);
        }
        combined_coeffs[exp_vec] += c2(i);
    }

    ssize_t size = combined_coeffs.size();
    py::array_t<int32_t> new_exponents({size, dimension});
    py::array_t<double> new_coeffs(size);
    auto new_exps_ptr = new_exponents.mutable_unchecked<2>();
    auto new_coeffs_ptr = new_coeffs.mutable_unchecked<1>();

    ssize_t idx = 0;
    for (auto const& [key, val] : combined_coeffs) {
        for (ssize_t j = 0; j < dimension; ++j) {
            new_exps_ptr(idx, j) = key[j];
        }
        new_coeffs_ptr(idx) = val;
        idx++;
    }

    return std::make_pair(new_exponents, new_coeffs);
}

std::pair<py::array_t<int32_t>, py::array_t<double>> multiply_mtf_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps1,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs1,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps2,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs2) {

    auto e1 = exps1.unchecked<2>();
    auto c1 = coeffs1.unchecked<1>();
    auto e2 = exps2.unchecked<2>();
    auto c2 = coeffs2.unchecked<1>();

    ssize_t n_terms1 = e1.shape(0);
    ssize_t n_terms2 = e2.shape(0);
    ssize_t dimension = (n_terms1 > 0) ? e1.shape(1) : ((n_terms2 > 0) ? e2.shape(1) : 0);

    if (n_terms1 == 0 || n_terms2 == 0) {
        // Correct way to create an empty array with a specific shape
        py::array::ShapeContainer shape({0, (ssize_t)dimension});
        return std::make_pair(py::array_t<int32_t>(shape), py::array_t<double>(0));
    }
    std::unordered_map<std::vector<int>, double, VectorHasher> combined_coeffs;

    #pragma omp parallel
    {
        std::unordered_map<std::vector<int>, double, VectorHasher> local_coeffs;
        #pragma omp for nowait
        for (ssize_t i = 0; i < n_terms1; ++i) {
            for (ssize_t j = 0; j < n_terms2; ++j) {
                std::vector<int> new_exp_vec(dimension);
                for (ssize_t k = 0; k < dimension; ++k) {
                    new_exp_vec[k] = e1(i, k) + e2(j, k);
                }
                local_coeffs[new_exp_vec] += c1(i) * c2(j);
            }
        }

        #pragma omp critical
        {
            for (auto const& [key, val] : local_coeffs) {
                combined_coeffs[key] += val;
            }
        }
    }

    ssize_t size = combined_coeffs.size();
    py::array_t<int32_t> new_exponents({size, dimension});
    py::array_t<double> new_coeffs(size);
    auto new_exps_ptr = new_exponents.mutable_unchecked<2>();
    auto new_coeffs_ptr = new_coeffs.mutable_unchecked<1>();

    ssize_t idx = 0;
    for (auto const& [key, val] : combined_coeffs) {
        for (ssize_t j = 0; j < dimension; ++j) {
            new_exps_ptr(idx, j) = key[j];
        }
        new_coeffs_ptr(idx) = val;
        idx++;
    }

    return std::make_pair(new_exponents, new_coeffs);
}
