#include "biot_savart_ops.hpp"
#include "mtf_data.hpp"
#include <stdexcept>
#include <cmath>

// Helper for vector subtraction
MtfVector subtract_vectors(const MtfVector& a, const MtfVector& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::runtime_error("Vectors must have 3 components.");
    }
    return {a[0].add(b[0].negate()),
            a[1].add(b[1].negate()),
            a[2].add(b[2].negate())};
}

// Helper for vector cross product
MtfVector cross_product(const MtfVector& a, const MtfVector& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::runtime_error("Vectors must have 3 components.");
    }
    MtfData cx = a[1].multiply(b[2]).add(a[2].multiply(b[1]).negate());
    MtfData cy = a[2].multiply(b[0]).add(a[0].multiply(b[2]).negate());
    MtfData cz = a[0].multiply(b[1]).add(a[1].multiply(b[0]).negate());
    return {cx, cy, cz};
}

std::vector<MtfVector> biot_savart_core_cpp(
    const std::vector<MtfVector>& source_points,
    const std::vector<MtfVector>& dl_vectors,
    const std::vector<MtfVector>& field_points) {

    if (source_points.size() != dl_vectors.size()) {
        throw std::runtime_error("Number of source points and dl vectors must be the same.");
    }

    if (field_points.empty()) {
        return {};
    }

    int dim = field_points[0][0].dimension;
    std::vector<MtfVector> B_fields;
    B_fields.reserve(field_points.size());

    double mu_0_4pi = 1e-7;
    MtfData scale_factor({{std::vector<int32_t>(dim, 0)}}, {mu_0_4pi}, dim);

    for (const auto& field_point : field_points) {
        MtfVector B_field_total = {MtfData({}, {}, dim), MtfData({}, {}, dim), MtfData({}, {}, dim)};
        for (size_t i = 0; i < source_points.size(); ++i) {
            const auto& source_point = source_points[i];
            const auto& dl_vector = dl_vectors[i];

            MtfVector r_vector = subtract_vectors(field_point, source_point);

            MtfData r_squared = r_vector[0].multiply(r_vector[0])
                              .add(r_vector[1].multiply(r_vector[1]))
                              .add(r_vector[2].multiply(r_vector[2]));

            MtfData inv_r_cubed = r_squared.power(-1.5);

            MtfVector cross_prod = cross_product(dl_vector, r_vector);

            B_field_total[0] = B_field_total[0].add(cross_prod[0].multiply(inv_r_cubed).multiply(scale_factor));
            B_field_total[1] = B_field_total[1].add(cross_prod[1].multiply(inv_r_cubed).multiply(scale_factor));
            B_field_total[2] = B_field_total[2].add(cross_prod[2].multiply(inv_r_cubed).multiply(scale_factor));
        }
        B_fields.push_back(B_field_total);
    }

    return B_fields;
}
