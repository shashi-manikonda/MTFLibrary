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

#endif // BIOT_SAVART_OPS_HPP
