#ifndef BIOT_SAVART_OPS_HPP
#define BIOT_SAVART_OPS_HPP

#include "mtf_data.hpp"
#include <vector>

// Each point is a vector of 3 MtfData objects (x, y, z)
using MtfVector = std::vector<MtfData>;

std::vector<MtfVector> biot_savart_core_cpp(
    const std::vector<MtfVector>& source_points,
    const std::vector<MtfVector>& dl_vectors,
    const std::vector<MtfVector>& field_points);

py::list biot_savart_from_numpy(
    py::list source_points_exps, py::list source_points_coeffs,
    py::list dl_vectors_exps, py::list dl_vectors_coeffs,
    py::list field_points_exps, py::list field_points_coeffs
);

py::list biot_savart_from_flat_numpy(
    py::array_t<int32_t> all_exponents,
    py::array_t<std::complex<double>> all_coeffs,
    py::array_t<int32_t> shapes
);

#endif // BIOT_SAVART_OPS_HPP
