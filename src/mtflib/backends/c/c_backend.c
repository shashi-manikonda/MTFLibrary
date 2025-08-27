#include "c_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- Helper Functions for MtfDataC ---

MtfDataC create_mtf_data(int dimension, size_t n_terms) {
    MtfDataC mtf;
    mtf.dimension = dimension;
    mtf.n_terms = n_terms;
    if (n_terms > 0) {
        mtf.exponents = (int32_t*)malloc(n_terms * dimension * sizeof(int32_t));
        mtf.coeffs = (complex_t*)malloc(n_terms * sizeof(complex_t));
    } else {
        mtf.exponents = NULL;
        mtf.coeffs = NULL;
    }
    return mtf;
}

void free_mtf_data(MtfDataC* mtf) {
    if (mtf) {
        free(mtf->exponents);
        free(mtf->coeffs);
    }
}

MtfDataC clone_mtf_data(const MtfDataC* src) {
    MtfDataC dst = create_mtf_data(src->dimension, src->n_terms);
    if (src->n_terms > 0) {
        memcpy(dst.exponents, src->exponents, src->n_terms * src->dimension * sizeof(int32_t));
        memcpy(dst.coeffs, src->coeffs, src->n_terms * sizeof(complex_t));
    }
    return dst;
}

// --- Math Operations ---

MtfDataC mtf_negate(const MtfDataC* a) {
    MtfDataC result = clone_mtf_data(a);
    for (size_t i = 0; i < result.n_terms; ++i) {
        result.coeffs[i] = -result.coeffs[i];
    }
    return result;
}

int exponent_compare(const int32_t* exp1, const int32_t* exp2, int dimension) {
    return memcmp(exp1, exp2, dimension * sizeof(int32_t));
}

MtfDataC mtf_add(const MtfDataC* a, const MtfDataC* b) {
    if (!a || !b) return create_mtf_data(0, 0);
    if (a->n_terms == 0) return clone_mtf_data(b);
    if (b->n_terms == 0) return clone_mtf_data(a);

    size_t new_capacity = a->n_terms + b->n_terms;
    MtfDataC result = create_mtf_data(a->dimension, 0);
    result.exponents = (int32_t*)realloc(result.exponents, new_capacity * a->dimension * sizeof(int32_t));
    result.coeffs = (complex_t*)realloc(result.coeffs, new_capacity * sizeof(complex_t));

    size_t i = 0, j = 0;
    while (i < a->n_terms && j < b->n_terms) {
        const int32_t* exp_a = a->exponents + i * a->dimension;
        const int32_t* exp_b = b->exponents + j * b->dimension;
        int cmp = exponent_compare(exp_a, exp_b, a->dimension);

        if (cmp < 0) {
            memcpy(result.exponents + result.n_terms * a->dimension, exp_a, a->dimension * sizeof(int32_t));
            result.coeffs[result.n_terms] = a->coeffs[i];
            i++;
        } else if (cmp > 0) {
            memcpy(result.exponents + result.n_terms * a->dimension, exp_b, a->dimension * sizeof(int32_t));
            result.coeffs[result.n_terms] = b->coeffs[j];
            j++;
        } else {
            complex_t sum = a->coeffs[i] + b->coeffs[j];
            if (cabs(sum) > 1e-16) {
                memcpy(result.exponents + result.n_terms * a->dimension, exp_a, a->dimension * sizeof(int32_t));
                result.coeffs[result.n_terms] = sum;
                result.n_terms++;
            }
            i++;
            j++;
            continue;
        }
        result.n_terms++;
    }

    while (i < a->n_terms) {
        memcpy(result.exponents + result.n_terms * a->dimension, a->exponents + i * a->dimension, a->dimension * sizeof(int32_t));
        result.coeffs[result.n_terms] = a->coeffs[i];
        i++;
        result.n_terms++;
    }
    while (j < b->n_terms) {
        memcpy(result.exponents + result.n_terms * a->dimension, b->exponents + j * b->dimension, b->dimension * sizeof(int32_t));
        result.coeffs[result.n_terms] = b->coeffs[j];
        j++;
        result.n_terms++;
    }
    return result;
}

typedef struct {
    int32_t* exponents;
    complex_t coeff;
} Term;

int compare_terms_r(const void* a, const void* b, void* arg) {
    int dimension = *(int*)arg;
    const Term* term_a = (const Term*)a;
    const Term* term_b = (const Term*)b;
    return memcmp(term_a->exponents, term_b->exponents, dimension * sizeof(int32_t));
}

MtfDataC mtf_multiply(const MtfDataC* a, const MtfDataC* b) {
    if (!a || !b || a->n_terms == 0 || b->n_terms == 0) {
        return create_mtf_data(a ? a->dimension : (b ? b->dimension : 0), 0);
    }

    size_t new_capacity = a->n_terms * b->n_terms;
    Term* new_terms = (Term*)malloc(new_capacity * sizeof(Term));
    int32_t* new_exponents_data = (int32_t*)malloc(new_capacity * a->dimension * sizeof(int32_t));

    size_t current_term = 0;
    for (size_t i = 0; i < a->n_terms; ++i) {
        for (size_t j = 0; j < b->n_terms; ++j) {
            new_terms[current_term].exponents = new_exponents_data + current_term * a->dimension;
            for (int k = 0; k < a->dimension; ++k) {
                new_terms[current_term].exponents[k] = a->exponents[i * a->dimension + k] + b->exponents[j * b->dimension + k];
            }
            new_terms[current_term].coeff = a->coeffs[i] * b->coeffs[j];
            current_term++;
        }
    }

    qsort_r(new_terms, new_capacity, sizeof(Term), compare_terms_r, &a->dimension);

    size_t merged_n_terms = 0;
    if (new_capacity > 0) {
        merged_n_terms = 1;
        for (size_t i = 1; i < new_capacity; ++i) {
            if (exponent_compare(new_terms[i].exponents, new_terms[merged_n_terms - 1].exponents, a->dimension) == 0) {
                new_terms[merged_n_terms - 1].coeff += new_terms[i].coeff;
            } else {
                if (new_terms[merged_n_terms].exponents != new_terms[i].exponents) {
                    memcpy(new_terms[merged_n_terms].exponents, new_terms[i].exponents, a->dimension * sizeof(int32_t));
                }
                new_terms[merged_n_terms].coeff = new_terms[i].coeff;
                merged_n_terms++;
            }
        }
    }

    size_t final_n_terms = 0;
    for (size_t i = 0; i < merged_n_terms; ++i) {
        if (cabs(new_terms[i].coeff) > 1e-16) {
            final_n_terms++;
        }
    }

    MtfDataC result = create_mtf_data(a->dimension, final_n_terms);
    size_t current_final_term = 0;
    for (size_t i = 0; i < merged_n_terms; ++i) {
        if (cabs(new_terms[i].coeff) > 1e-16) {
            memcpy(result.exponents + current_final_term * a->dimension, new_terms[i].exponents, a->dimension * sizeof(int32_t));
            result.coeffs[current_final_term] = new_terms[i].coeff;
            current_final_term++;
        }
    }

    free(new_terms);
    free(new_exponents_data);
    return result;
}

static const double isqrt_coeffs[] = {
    1.0, -0.5, 0.375, -0.3125, 0.2734375, -0.24609375, 0.2255859375, -0.20947265625,
    0.196380615234375, -0.1854705810546875, 0.17619705200195312, -0.16818809509277344,
    0.1611802577972412, -0.15498101711273193, 0.14944598078727722, -0.14446444809436798
};
static const int num_isqrt_coeffs = sizeof(isqrt_coeffs) / sizeof(double);

MtfDataC mtf_power(const MtfDataC* a, double exponent) {
    if (exponent != -0.5) {
        int exp_int = (int)exponent;
        if (exp_int == exponent && exp_int >= 0) { // integer power
            if (exp_int == 0) {
                MtfDataC result = create_mtf_data(a->dimension, 1);
                memset(result.exponents, 0, result.dimension * sizeof(int32_t));
                result.coeffs[0] = 1.0;
                return result;
            }
            MtfDataC result = clone_mtf_data(a);
            for (int i = 1; i < exp_int; ++i) {
                MtfDataC temp = mtf_multiply(&result, a);
                free_mtf_data(&result);
                result = temp;
            }
            return result;
        }
        return create_mtf_data(a->dimension, 0); // Not implemented for other exponents
    }

    complex_t const_term = 0.0;
    int const_term_idx = -1;
    int32_t* zero_exp = (int32_t*)calloc(a->dimension, sizeof(int32_t));
    for (size_t i = 0; i < a->n_terms; ++i) {
        if (memcmp(a->exponents + i * a->dimension, zero_exp, a->dimension * sizeof(int32_t)) == 0) {
            const_term = a->coeffs[i];
            const_term_idx = i;
            break;
        }
    }
    free(zero_exp);

    if (const_term_idx == -1 || cabs(const_term) < 1e-16) {
        return create_mtf_data(a->dimension, 0);
    }

    MtfDataC temp = clone_mtf_data(a);
    complex_t const_factor_isqrt = 1.0 / csqrt(const_term);
    for (size_t i = 0; i < temp.n_terms; ++i) {
        temp.coeffs[i] /= const_term;
    }

    MtfDataC x = create_mtf_data(temp.dimension, temp.n_terms > 0 ? temp.n_terms - 1 : 0);
    size_t current_x_term = 0;
    for (size_t i = 0; i < temp.n_terms; ++i) {
        if (i == (size_t)const_term_idx) continue;
        memcpy(x.exponents + current_x_term * x.dimension, temp.exponents + i * temp.dimension, x.dimension * sizeof(int32_t));
        x.coeffs[current_x_term] = temp.coeffs[i];
        current_x_term++;
    }
    free_mtf_data(&temp);

    MtfDataC result = create_mtf_data(a->dimension, 1);
    memset(result.exponents, 0, result.dimension * sizeof(int32_t));
    result.coeffs[0] = isqrt_coeffs[0];

    MtfDataC x_power_n = clone_mtf_data(&x);

    for (int n = 1; n < num_isqrt_coeffs; ++n) {
        MtfDataC term = clone_mtf_data(&x_power_n);
        for (size_t i = 0; i < term.n_terms; ++i) {
            term.coeffs[i] *= isqrt_coeffs[n];
        }
        MtfDataC new_result = mtf_add(&result, &term);
        free_mtf_data(&result);
        result = new_result;
        free_mtf_data(&term);

        if (n < num_isqrt_coeffs - 1) {
            MtfDataC new_x_power_n = mtf_multiply(&x_power_n, &x);
            free_mtf_data(&x_power_n);
            x_power_n = new_x_power_n;
        }
    }

    free_mtf_data(&x);
    free_mtf_data(&x_power_n);

    for (size_t i = 0; i < result.n_terms; ++i) {
        result.coeffs[i] *= const_factor_isqrt;
    }

    return result;
}

// --- Main Biot-Savart Function ---

void biot_savart_c(
    const int32_t* all_exponents, const complex_t* all_coeffs, const int32_t* shapes,
    int32_t** result_exps, complex_t** result_coeffs, int32_t** result_shapes, size_t* total_result_terms_out
) {
    int n_source_points = shapes[0];
    int n_dl_vectors = shapes[1];
    int n_field_points = shapes[2];
    int dimension = shapes[3];
    const int32_t* sp_shapes = shapes + 4;
    const int32_t* dl_shapes = shapes + 4 + n_source_points * 3;
    const int32_t* fp_shapes = shapes + 4 + n_source_points * 3 + n_dl_vectors * 3;

    MtfDataC* sp = (MtfDataC*)malloc(n_source_points * 3 * sizeof(MtfDataC));
    MtfDataC* dl = (MtfDataC*)malloc(n_dl_vectors * 3 * sizeof(MtfDataC));
    MtfDataC* fp = (MtfDataC*)malloc(n_field_points * 3 * sizeof(MtfDataC));

    int current_term_offset = 0;
    for(int i=0; i<n_source_points*3; ++i) {
        sp[i] = create_mtf_data(dimension, sp_shapes[i]);
        memcpy(sp[i].exponents, all_exponents + (size_t)current_term_offset * dimension, (size_t)sp_shapes[i] * dimension * sizeof(int32_t));
        memcpy(sp[i].coeffs, all_coeffs + current_term_offset, (size_t)sp_shapes[i] * sizeof(complex_t));
        current_term_offset += sp_shapes[i];
    }
    // Note: The python wrapper must send all_coeffs and all_exponents concatenated in the right order.
    // Here we assume it is sp, then dl, then fp.
    for(int i=0; i<n_dl_vectors*3; ++i) {
        dl[i] = create_mtf_data(dimension, dl_shapes[i]);
        memcpy(dl[i].exponents, all_exponents + (size_t)current_term_offset * dimension, (size_t)dl_shapes[i] * dimension * sizeof(int32_t));
        memcpy(dl[i].coeffs, all_coeffs + current_term_offset, (size_t)dl_shapes[i] * sizeof(complex_t));
        current_term_offset += dl_shapes[i];
    }
    for(int i=0; i<n_field_points*3; ++i) {
        fp[i] = create_mtf_data(dimension, fp_shapes[i]);
        memcpy(fp[i].exponents, all_exponents + (size_t)current_term_offset * dimension, (size_t)fp_shapes[i] * dimension * sizeof(int32_t));
        memcpy(fp[i].coeffs, all_coeffs + current_term_offset, (size_t)fp_shapes[i] * sizeof(complex_t));
        current_term_offset += fp_shapes[i];
    }

    MtfDataC* B_fields = (MtfDataC*)malloc(n_field_points * 3 * sizeof(MtfDataC));
    double mu_0_4pi = 1e-7;

    #pragma omp parallel for
    for (int i = 0; i < n_field_points; ++i) {
        MtfDataC B_field_total[3];
        for(int k=0; k<3; ++k) B_field_total[k] = create_mtf_data(dimension, 0);

        for (int j = 0; j < n_source_points; ++j) {
            MtfDataC r_vec[3];
            for(int k=0; k<3; ++k) {
                MtfDataC neg_sp = mtf_negate(&sp[j*3+k]);
                r_vec[k] = mtf_add(&fp[i*3+k], &neg_sp);
                free_mtf_data(&neg_sp);
            }

            MtfDataC r2 = mtf_multiply(&r_vec[0], &r_vec[0]);
            MtfDataC r2y = mtf_multiply(&r_vec[1], &r_vec[1]);
            MtfDataC r2z = mtf_multiply(&r_vec[2], &r_vec[2]);
            MtfDataC r2_sum1 = mtf_add(&r2, &r2y);
            MtfDataC r2_sum2 = mtf_add(&r2_sum1, &r2z);

            MtfDataC r_inv = mtf_power(&r2_sum2, -0.5);
            MtfDataC r_inv2 = mtf_multiply(&r_inv, &r_inv);
            MtfDataC inv_r3 = mtf_multiply(&r_inv2, &r_inv);

            MtfDataC cross_prod[3];
            MtfDataC dl_x_r_x = mtf_multiply(&dl[j*3+1], &r_vec[2]);
            MtfDataC dl_y_r_x = mtf_multiply(&dl[j*3+2], &r_vec[1]);
            MtfDataC neg_dl_y_r_x = mtf_negate(&dl_y_r_x);
            cross_prod[0] = mtf_add(&dl_x_r_x, &neg_dl_y_r_x);

            MtfDataC dl_z_r_x = mtf_multiply(&dl[j*3+2], &r_vec[0]);
            MtfDataC dl_x_r_z = mtf_multiply(&dl[j*3+0], &r_vec[2]);
            MtfDataC neg_dl_x_r_z = mtf_negate(&dl_x_r_z);
            cross_prod[1] = mtf_add(&dl_z_r_x, &neg_dl_x_r_z);

            MtfDataC dl_x_r_y = mtf_multiply(&dl[j*3+0], &r_vec[1]);
            MtfDataC dl_y_r_x_2 = mtf_multiply(&dl[j*3+1], &r_vec[0]);
            MtfDataC neg_dl_y_r_x_2 = mtf_negate(&dl_y_r_x_2);
            cross_prod[2] = mtf_add(&dl_x_r_y, &neg_dl_y_r_x_2);

            for(int k=0; k<3; ++k) {
                MtfDataC term = mtf_multiply(&cross_prod[k], &inv_r3);
                for(size_t l=0; l<term.n_terms; ++l) term.coeffs[l] *= mu_0_4pi;
                MtfDataC new_total = mtf_add(&B_field_total[k], &term);
                free_mtf_data(&B_field_total[k]);
                B_field_total[k] = new_total;
                free_mtf_data(&term);
            }
        }
        for(int k=0; k<3; ++k) B_fields[i*3+k] = B_field_total[k];
    }

    size_t total_terms = 0;
    for(int i=0; i<n_field_points*3; ++i) total_terms += B_fields[i].n_terms;
    *total_result_terms_out = total_terms;

    *result_exps = (int32_t*)malloc(total_terms * dimension * sizeof(int32_t));
    *result_coeffs = (complex_t*)malloc(total_terms * sizeof(complex_t));
    *result_shapes = (int32_t*)malloc((n_field_points * 3 + 1) * sizeof(int32_t));
    (*result_shapes)[0] = n_field_points;

    size_t current_res_offset = 0;
    for(int i=0; i<n_field_points*3; ++i) {
        (*result_shapes)[i+1] = B_fields[i].n_terms;
        if (B_fields[i].n_terms > 0) {
            memcpy(*result_exps + current_res_offset * dimension, B_fields[i].exponents, B_fields[i].n_terms * dimension * sizeof(int32_t));
            memcpy(*result_coeffs + current_res_offset, B_fields[i].coeffs, B_fields[i].n_terms * sizeof(complex_t));
            current_res_offset += B_fields[i].n_terms;
        }
    }

    for(int i=0; i<n_source_points*3; ++i) free_mtf_data(&sp[i]);
    for(int i=0; i<n_dl_vectors*3; ++i) free_mtf_data(&dl[i]);
    for(int i=0; i<n_field_points*3; ++i) free_mtf_data(&fp[i]);
    for(int i=0; i<n_field_points*3; ++i) free_mtf_data(&B_fields[i]);
    free(sp);
    free(dl);
    free(fp);
    free(B_fields);
}
