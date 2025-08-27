#ifndef C_BACKEND_H
#define C_BACKEND_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include <complex>
typedef std::complex<double> complex_t;
#else
#include <complex.h>
typedef double complex complex_t;
#endif

typedef struct {
    int32_t* exponents;
    complex_t* coeffs;
    size_t n_terms;
    int dimension;
} MtfDataC;

#ifdef __cplusplus
extern "C" {
#endif

void biot_savart_c(
    const int32_t* all_exponents, const complex_t* all_coeffs, const int32_t* shapes,
    int32_t** result_exps, complex_t** result_coeffs, int32_t** result_shapes, size_t* total_result_terms
);

#ifdef __cplusplus
}
#endif

#endif // C_BACKEND_H
