#ifndef MTF_DATA_HPP
#define MTF_DATA_HPP

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <map>
#include <complex>
#include "precomputed_coefficients.hpp"

namespace py = pybind11;

// Custom hasher for std::vector<int32_t> so it can be used as a key in std::unordered_map
struct VectorHasher {
    std::size_t operator()(const std::vector<int32_t>& vec) const {
        std::size_t seed = vec.size();
        for(int32_t i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class MtfData {
public:
    std::vector<std::vector<int32_t>> exponents;
    std::vector<std::complex<double>> coeffs;
    int dimension;

    // Default constructor
    MtfData();

    // Constructor from numpy arrays
    MtfData(py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps_arr,
            py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coeffs_arr);

    // Constructor from C++ vectors
    MtfData(std::vector<std::vector<int32_t>> exps, std::vector<std::complex<double>> cs, int dim);

    // Method to convert back to python dictionary of numpy arrays
    py::dict to_dict() const;

    MtfData add(const MtfData& other) const;
    void add_inplace(const MtfData& other);
    MtfData multiply(const MtfData& other) const;
    void multiply_inplace(const MtfData& other);
    MtfData compose(const std::map<int, MtfData>& other_function_dict) const;
    MtfData negate() const;
    MtfData power(double exponent) const;
};

const std::vector<double>& get_precomputed_coeffs(const std::string& name);

MtfData sqrt_taylor_1D_expansion(const MtfData& variable, int order);
MtfData isqrt_taylor_1D_expansion(const MtfData& variable, int order);
MtfData inverse_taylor_1D_expansion(const MtfData& variable, int order);


std::pair<py::array_t<int32_t>, py::array_t<std::complex<double>>> add_mtf_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps1,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coeffs1,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps2,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coeffs2);

std::pair<py::array_t<int32_t>, py::array_t<std::complex<double>>> multiply_mtf_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps1,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coeffs1,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps2,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coeffs2);

std::complex<double> extract_coefficient_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coeffs,
    std::vector<int32_t> exp_to_find);

#endif // MTF_DATA_HPP
