#include "mtf_data.hpp"
#include <vector>
#include <unordered_map>
#include "precomputed_coefficients.hpp"
#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <cmath>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

MtfData::MtfData() : dimension(0) {}

MtfData::MtfData(py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps_arr,
                 py::array_t<double, py::array::c_style | py::array::forcecast> coeffs_arr) {
    auto exps_buf = exps_arr.request();
    auto coeffs_buf = coeffs_arr.request();

    if (exps_buf.ndim != 2 || coeffs_buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 2 for exponents and 1 for coefficients.");
    }
    if (exps_buf.shape[0] != coeffs_buf.shape[0]) {
        throw std::runtime_error("Input shapes must match.");
    }

    dimension = exps_buf.shape[1];
    ssize_t n_terms = exps_buf.shape[0];
    auto exps_ptr = static_cast<int32_t*>(exps_buf.ptr);
    auto coeffs_ptr = static_cast<double*>(coeffs_buf.ptr);

    exponents.resize(n_terms, std::vector<int32_t>(dimension));
    coeffs.resize(n_terms);

    for (ssize_t i = 0; i < n_terms; ++i) {
        for (ssize_t j = 0; j < dimension; ++j) {
            exponents[i][j] = exps_ptr[i * dimension + j];
        }
        coeffs[i] = coeffs_ptr[i];
    }
}

MtfData::MtfData(std::vector<std::vector<int32_t>> exps, std::vector<double> cs, int dim)
    : exponents(exps), coeffs(cs), dimension(dim) {}


py::dict MtfData::to_dict() const {
    py::dict d;

    ssize_t n_terms = exponents.size();
    py::array_t<int32_t> exps_arr({n_terms, (ssize_t)dimension});
    py::array_t<double> coeffs_arr(n_terms);

    auto exps_ptr = exps_arr.mutable_unchecked<2>();
    auto coeffs_ptr = coeffs_arr.mutable_unchecked<1>();

    for (ssize_t i = 0; i < n_terms; ++i) {
        for (ssize_t j = 0; j < dimension; ++j) {
            exps_ptr(i, j) = exponents[i][j];
        }
        coeffs_ptr(i) = coeffs[i];
    }

    d["exponents"] = exps_arr;
    d["coeffs"] = coeffs_arr;
    return d;
}

MtfData MtfData::add(const MtfData& other) const {
    if (dimension != other.dimension) {
        throw std::runtime_error("Dimensions must match for addition.");
    }

    std::unordered_map<std::vector<int32_t>, double, VectorHasher> combined_coeffs_map;

    for (size_t i = 0; i < coeffs.size(); ++i) {
        combined_coeffs_map[exponents[i]] = coeffs[i];
    }

    for (size_t i = 0; i < other.coeffs.size(); ++i) {
        combined_coeffs_map[other.exponents[i]] += other.coeffs[i];
    }

    std::vector<std::vector<int32_t>> new_exponents;
    std::vector<double> new_coeffs;
    for (auto const& [exp, coeff] : combined_coeffs_map) {
        new_exponents.push_back(exp);
        new_coeffs.push_back(coeff);
    }

    return MtfData(new_exponents, new_coeffs, dimension);
}


MtfData MtfData::multiply(const MtfData& other) const {
    if (dimension != other.dimension) {
        throw std::runtime_error("Dimensions must match for multiplication.");
    }

    std::unordered_map<std::vector<int32_t>, double, VectorHasher> combined_coeffs_map;

    for (size_t i = 0; i < coeffs.size(); ++i) {
        for (size_t j = 0; j < other.coeffs.size(); ++j) {
            std::vector<int32_t> new_exp(dimension);
            for (int k = 0; k < dimension; ++k) {
                new_exp[k] = exponents[i][k] + other.exponents[j][k];
            }
            combined_coeffs_map[new_exp] += coeffs[i] * other.coeffs[j];
        }
    }

    std::vector<std::vector<int32_t>> new_exponents;
    std::vector<double> new_coeffs;
    for (auto const& [exp, coeff] : combined_coeffs_map) {
        new_exponents.push_back(exp);
        new_coeffs.push_back(coeff);
    }

    return MtfData(new_exponents, new_coeffs, dimension);
}

MtfData MtfData::compose(const std::map<int, MtfData>& other_function_dict) const {
    if (other_function_dict.empty()) {
        return *this;
    }

    int result_dim = -1;
    for (auto const& [var_index, g] : other_function_dict) {
        if (result_dim == -1) {
            result_dim = g.dimension;
        } else if (result_dim != g.dimension) {
            throw std::runtime_error("All inner functions must have the same dimension.");
        }
    }

    // Create a zero MTF for the result
    MtfData final_mtf({}, {}, result_dim);

    for (size_t i = 0; i < coeffs.size(); ++i) {
        double coeff = coeffs[i];
        const auto& exp = exponents[i];

        MtfData term_result({{std::vector<int32_t>(result_dim, 0)}}, {coeff}, result_dim);

        for (int j = 0; j < dimension; ++j) {
            int power = exp[j];
            if (power > 0) {
                MtfData g_j = other_function_dict.at(j + 1);
                for (int p = 0; p < power; ++p) {
                    term_result = term_result.multiply(g_j);
                }
            }
        }
        final_mtf = final_mtf.add(term_result);
    }

    return final_mtf;
}

MtfData MtfData::negate() const {
    std::vector<double> new_coeffs = coeffs;
    for (double& c : new_coeffs) {
        c = -c;
    }
    return MtfData(exponents, new_coeffs, dimension);
}

std::pair<double, MtfData> _split_constant_polynomial_part(const MtfData& mtf) {
    double constant_term = 0.0;
    std::vector<std::vector<int32_t>> poly_exponents;
    std::vector<double> poly_coeffs;

    for (size_t i = 0; i < mtf.coeffs.size(); ++i) {
        bool is_const = true;
        for (int32_t exp_val : mtf.exponents[i]) {
            if (exp_val != 0) {
                is_const = false;
                break;
            }
        }
        if (is_const) {
            constant_term = mtf.coeffs[i];
        } else {
            poly_exponents.push_back(mtf.exponents[i]);
            poly_coeffs.push_back(mtf.coeffs[i]);
        }
    }
    return {constant_term, MtfData(poly_exponents, poly_coeffs, mtf.dimension)};
}

MtfData MtfData::power(double exponent) const {
    if (exponent == 0) {
        return MtfData({{std::vector<int32_t>(dimension, 0)}}, {1.0}, dimension);
    }
    if (exponent == 1) {
        return *this;
    }

    if (abs(exponent - static_cast<int>(exponent)) < 1e-9) { // integer power
        int power_int = static_cast<int>(exponent);
        if (power_int > 1) {
            MtfData result = *this;
            for (int i = 1; i < power_int; ++i) {
                result = result.multiply(*this);
            }
            return result;
        } else if (power_int == -1) {
            auto [const_term, poly_part] = _split_constant_polynomial_part(*this);
            if (abs(const_term) < 1e-16) {
                throw std::runtime_error("Cannot invert MTF with zero constant term.");
            }
            MtfData rescaled_mtf = this->multiply(MtfData({{std::vector<int32_t>(dimension, 0)}}, {1.0/const_term}, dimension));
            MtfData one_mtf({{std::vector<int32_t>(dimension, 0)}}, {1.0}, dimension);
            MtfData composed_mtf = inverse_taylor_1D_expansion(rescaled_mtf.add(one_mtf.negate()), 10);
            return composed_mtf.multiply(MtfData({{std::vector<int32_t>(dimension, 0)}}, {1.0/const_term}, dimension));
        }
    } else if (exponent == 0.5) {
        auto [const_term, poly_part] = _split_constant_polynomial_part(*this);
        if (const_term < 0) {
             throw std::runtime_error("Cannot take sqrt of MTF with negative constant term.");
        }
        double const_factor_sqrt = sqrt(const_term);
        MtfData poly_part_x = poly_part.multiply(MtfData({{std::vector<int32_t>(dimension, 0)}}, {1.0/const_term}, dimension));
        MtfData sqrt_1_plus_x = sqrt_taylor_1D_expansion(poly_part_x, 10);
        return sqrt_1_plus_x.multiply(MtfData({{std::vector<int32_t>(dimension, 0)}}, {const_factor_sqrt}, dimension));
    } else if (exponent == -0.5) {
        auto [const_term, poly_part] = _split_constant_polynomial_part(*this);
        if (const_term <= 0) {
             throw std::runtime_error("Cannot take isqrt of MTF with non-positive constant term.");
        }
        double const_factor_isqrt = 1.0/sqrt(const_term);
        MtfData poly_part_x = poly_part.multiply(MtfData({{std::vector<int32_t>(dimension, 0)}}, {1.0/const_term}, dimension));
        MtfData isqrt_1_plus_x = isqrt_taylor_1D_expansion(poly_part_x, 10);
        return isqrt_1_plus_x.multiply(MtfData({{std::vector<int32_t>(dimension, 0)}}, {const_factor_isqrt}, dimension));
    } else if (exponent == -1.5) {
        MtfData r_inv = this->power(-1.0);
        MtfData r_inv_sqrt = this->power(-0.5);
        return r_inv.multiply(r_inv_sqrt);
    }

    throw std::runtime_error("Power function not implemented for this exponent.");
}

const std::vector<double>& get_precomputed_coeffs(const std::string& name) {
    auto it = mtf_coeffs::precomputed_coefficients.find(name);
    if (it == mtf_coeffs::precomputed_coefficients.end()) {
        throw std::runtime_error("Precomputed coefficients not found for " + name);
    }
    return it->second;
}

MtfData sqrt_taylor_1D_expansion(const MtfData& variable, int order) {
    const auto& coeffs = get_precomputed_coeffs("sqrt");
    std::vector<std::vector<int32_t>> exps;
    std::vector<double> selected_coeffs;
    for (int i = 0; i <= order; ++i) {
        exps.push_back({i});
        selected_coeffs.push_back(coeffs[i]);
    }
    MtfData mtf_1d(exps, selected_coeffs, 1);
    std::map<int, MtfData> compose_map;
    compose_map[1] = variable;
    return mtf_1d.compose(compose_map);
}

MtfData isqrt_taylor_1D_expansion(const MtfData& variable, int order) {
    const auto& coeffs = get_precomputed_coeffs("isqrt");
    std::vector<std::vector<int32_t>> exps;
    std::vector<double> selected_coeffs;
    for (int i = 0; i <= order; ++i) {
        exps.push_back({i});
        selected_coeffs.push_back(coeffs[i]);
    }
    MtfData mtf_1d(exps, selected_coeffs, 1);
    std::map<int, MtfData> compose_map;
    compose_map[1] = variable;
    return mtf_1d.compose(compose_map);
}

MtfData inverse_taylor_1D_expansion(const MtfData& variable, int order) {
    const auto& coeffs = get_precomputed_coeffs("inverse");
    std::vector<std::vector<int32_t>> exps;
    std::vector<double> selected_coeffs;
    for (int i = 0; i <= order; ++i) {
        exps.push_back({i});
        selected_coeffs.push_back(coeffs[i]);
    }
    MtfData mtf_1d(exps, selected_coeffs, 1);
    std::map<int, MtfData> compose_map;
    compose_map[1] = variable;
    return mtf_1d.compose(compose_map);
}


std::pair<py::array_t<int32_t>, py::array_t<double>> add_mtf_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps1,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs1,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps2,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs2) {

    MtfData mtf1(exps1, coeffs1);
    MtfData mtf2(exps2, coeffs2);
    MtfData result = mtf1.add(mtf2);
    py::dict d = result.to_dict();
    return std::make_pair(d["exponents"].cast<py::array_t<int32_t>>(), d["coeffs"].cast<py::array_t<double>>());
}

std::pair<py::array_t<int32_t>, py::array_t<double>> multiply_mtf_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps1,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs1,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps2,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs2) {

    MtfData mtf1(exps1, coeffs1);
    MtfData mtf2(exps2, coeffs2);
    MtfData result = mtf1.multiply(mtf2);
    py::dict d = result.to_dict();
    return std::make_pair(d["exponents"].cast<py::array_t<int32_t>>(), d["coeffs"].cast<py::array_t<double>>());
}

double extract_coefficient_cpp(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> exps,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs,
    std::vector<int32_t> exp_to_find) {

    auto e = exps.unchecked<2>();
    auto c = coeffs.unchecked<1>();

    ssize_t n_terms = e.shape(0);
    ssize_t dimension = e.shape(1);

    if (exp_to_find.size() != (size_t)dimension) {
        throw std::runtime_error("Dimension of exponent to find does not match dimension of MTF.");
    }

    std::unordered_map<std::vector<int32_t>, double, VectorHasher> exp_map;
    for (ssize_t i = 0; i < n_terms; ++i) {
        std::vector<int32_t> exp_vec(dimension);
        for (ssize_t j = 0; j < dimension; ++j) {
            exp_vec[j] = e(i, j);
        }
        exp_map[exp_vec] = c(i);
    }

    auto it = exp_map.find(exp_to_find);
    if (it != exp_map.end()) {
        return it->second;
    }

    return 0.0;
}
