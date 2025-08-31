#include "biot_savart_ops.hpp"
#include "mtf_data.hpp"
#include <stdexcept>
#include <cmath>
#include <omp.h>

// Helper for vector subtraction
void subtract_vectors_inplace(MtfVector& result, const MtfVector& a, const MtfVector& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::runtime_error("Vectors must have 3 components.");
    }
    result[0] = a[0];
    result[0].add_inplace(b[0].negate());
    result[1] = a[1];
    result[1].add_inplace(b[1].negate());
    result[2] = a[2];
    result[2].add_inplace(b[2].negate());
}

// Helper for vector cross product
void cross_product_inplace(MtfVector& result, const MtfVector& a, const MtfVector& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::runtime_error("Vectors must have 3 components.");
    }
    result[0] = a[1].multiply(b[2]);
    result[0].add_inplace(a[2].multiply(b[1]).negate());
    result[1] = a[2].multiply(b[0]);
    result[1].add_inplace(a[0].multiply(b[2]).negate());
    result[2] = a[0].multiply(b[1]);
    result[2].add_inplace(a[1].multiply(b[0]).negate());
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
    std::vector<MtfVector> B_fields(field_points.size());

    double mu_0_4pi = 1e-7;
    MtfData scale_factor(dim);
    std::vector<int32_t> zero_exp(dim, 0);
    scale_factor.exponents.insert(scale_factor.exponents.end(), zero_exp.begin(), zero_exp.end());
    scale_factor.coeffs.push_back({mu_0_4pi, 0.0});
    scale_factor.n_terms = 1;


    #pragma omp parallel for
    for (long i = 0; i < field_points.size(); ++i) {
        const auto& field_point = field_points[i];
        MtfVector B_field_total = {MtfData(dim), MtfData(dim), MtfData(dim)};
        for (long j = 0; j < source_points.size(); ++j) {
            const auto& source_point = source_points[j];
            const auto& dl_vector = dl_vectors[j];

            MtfVector r_vector(3, MtfData(dim));
            subtract_vectors_inplace(r_vector, field_point, source_point);

            MtfData r_squared = r_vector[0];
            r_squared.multiply_inplace(r_vector[0]);
            MtfData r_squared_y = r_vector[1];
            r_squared_y.multiply_inplace(r_vector[1]);
            r_squared.add_inplace(r_squared_y);
            MtfData r_squared_z = r_vector[2];
            r_squared_z.multiply_inplace(r_vector[2]);
            r_squared.add_inplace(r_squared_z);

            r_squared.power_inplace(-0.5); // r_inv
            MtfData inv_r_cubed = r_squared;
            inv_r_cubed.multiply_inplace(r_squared);
            inv_r_cubed.multiply_inplace(r_squared);

            inv_r_cubed.multiply_inplace(scale_factor);

            MtfVector cross_prod(3, MtfData(dim));
            cross_product_inplace(cross_prod, dl_vector, r_vector);

            cross_prod[0].multiply_inplace(inv_r_cubed);
            B_field_total[0].add_inplace(cross_prod[0]);

            cross_prod[1].multiply_inplace(inv_r_cubed);
            B_field_total[1].add_inplace(cross_prod[1]);

            cross_prod[2].multiply_inplace(inv_r_cubed);
            B_field_total[2].add_inplace(cross_prod[2]);
        }
        B_fields[i] = B_field_total;
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
                MtfData mtf;
                mtf.from_numpy(p_exps[j].cast<py::array_t<int32_t>>(), p_coeffs[j].cast<py::array_t<std::complex<double>>>());
                vec.push_back(mtf);
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

py::list biot_savart_from_flat_numpy(
    py::array_t<int32_t> all_exponents,
    py::array_t<std::complex<double>> all_coeffs,
    py::array_t<int32_t> shapes
) {
    auto exps_ptr = all_exponents.unchecked<2>();
    auto coeffs_ptr = all_coeffs.unchecked<1>();
    auto shapes_ptr = shapes.unchecked<1>();

    int n_source_points = shapes_ptr(0);
    int n_dl_vectors = shapes_ptr(1);
    int n_field_points = shapes_ptr(2);
    int dimension = shapes_ptr(3);
    int n_sp_terms_offset = 4;
    int n_dl_terms_offset = n_sp_terms_offset + n_source_points * 3;
    int n_fp_terms_offset = n_dl_terms_offset + n_dl_vectors * 3;

    auto to_mtf_vector = [&](int n_points, int terms_offset, int& current_offset) {
        std::vector<MtfVector> mtf_vectors;
        mtf_vectors.reserve(n_points);
        for (int i = 0; i < n_points; ++i) {
            MtfVector vec;
            vec.reserve(3);
            for (int j = 0; j < 3; ++j) {
                int n_terms = shapes_ptr(terms_offset + i*3 + j);
                MtfData mtf(dimension);
                mtf.n_terms = n_terms;
                mtf.exponents.resize(n_terms * dimension);
                mtf.coeffs.resize(n_terms);

                for (int k = 0; k < n_terms; ++k) {
                    for (int l = 0; l < dimension; ++l) {
                        mtf.exponents[k * dimension + l] = exps_ptr(current_offset + k, l);
                    }
                    mtf.coeffs[k] = coeffs_ptr(current_offset + k);
                }
                vec.push_back(mtf);
                current_offset += n_terms;
            }
            mtf_vectors.push_back(vec);
        }
        return mtf_vectors;
    };

    int current_offset = 0;
    std::vector<MtfVector> source_points = to_mtf_vector(n_source_points, n_sp_terms_offset, current_offset);
    std::vector<MtfVector> dl_vectors = to_mtf_vector(n_dl_vectors, n_dl_terms_offset, current_offset);
    std::vector<MtfVector> field_points = to_mtf_vector(n_field_points, n_fp_terms_offset, current_offset);

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
