#ifndef MTF_OPS_HPP
#define MTF_OPS_HPP

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Custom hasher for std::vector<int> so it can be used as a key in std::unordered_map
struct VectorHasher {
    std::size_t operator()(const std::vector<int>& vec) const {
        std::size_t seed = vec.size();
        for(int i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

std::pair<py::array_t<int32_t>, py::array_t<double>> add_mtf_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps1,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs1,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps2,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs2);

std::pair<py::array_t<int32_t>, py::array_t<double>> multiply_mtf_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps1,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs1,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps2,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs2);

#endif // MTF_OPS_HPP
