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

            MtfData r = r_squared.power(0.5);
            MtfData r_inv = r.power(-1.0);
            MtfData inv_r_cubed = r_inv.multiply(r_inv).multiply(r_inv);

            MtfVector cross_prod = cross_product(dl_vector, r_vector);

            MtfData inv_r_cubed_scaled = inv_r_cubed.multiply(scale_factor);

            MtfData dBx = cross_prod[0].multiply(inv_r_cubed_scaled);
            B_field_total[0].add_inplace(dBx);

            MtfData dBy = cross_prod[1].multiply(inv_r_cubed_scaled);
            B_field_total[1].add_inplace(dBy);

            MtfData dBz = cross_prod[2].multiply(inv_r_cubed_scaled);
            B_field_total[2].add_inplace(dBz);
        }
        B_fields.push_back(B_field_total);
    }

    return B_fields;
}

py::list biot_savart_from_numpy(
    py::list source_points_exps, py::list source_points_coeffs,
    py::list dl_vectors_exps, py::list dl_vectors_coeffs,
    py::list field_points_exps, py::list field_points_coeffs
) {
    auto to_mtf_vector = [](py::list exps_list, py::list coeffs_list) {
        std::vector<MtfVector> mtf_vectors;
        for (size_t i = 0; i < exps_list.size(); ++i) {
            py::list p_exps = exps_list[i].cast<py::list>();
            py::list p_coeffs = coeffs_list[i].cast<py::list>();
            MtfVector vec;
            for (size_t j = 0; j < p_exps.size(); ++j) {
                vec.push_back(MtfData(p_exps[j].cast<py::array_t<int32_t>>(), p_coeffs[j].cast<py::array_t<std::complex<double>>>()));
            }
            mtf_vectors.push_back(vec);
        }
        return mtf_vectors;
    };

    std::vector<MtfVector> source_points = to_mtf_vector(source_points_exps, source_points_coeffs);
    std::vector<MtfVector> dl_vectors = to_mtf_vector(dl_vectors_exps, dl_vectors_coeffs);
    std::vector<MtfVector> field_points = to_mtf_vector(field_points_exps, field_points_coeffs);

    std::vector<MtfVector> b_fields = biot_savart_core_cpp(source_points, dl_vectors, field_points);

    py::list result;
    for (const auto& b_vec : b_fields) {
        py::list b_vec_py;
        for (const auto& mtf : b_vec) {
            b_vec_py.append(mtf.to_dict());
        }
        result.append(b_vec_py);
    }

    return result;
}
