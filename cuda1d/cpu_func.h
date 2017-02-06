#ifndef H_FUNC_CPU_FFT_KYANG
#define H_FUNC_CPU_FFT_KYANG
#ifdef __cplusplus
extern "C" {
#endif
#include <complex.h>
#include <fftw3.h>

void fftw_fft1d_plan(fftwf_plan *p, int nr);
void fftw_fft1d_r2c(fftwf_plan p, float *pr, int rdist, int nmemb, complex float *po, int cdist, int repeat);
void fftw_fft1d_c2r(fftwf_plan p, complex float *po, int cdist, int nmemb, float *pr, int rdist, int repeat);
void fftw_fft1d_destroy(fftwf_plan *p);

#ifdef __cplusplus
}
#endif
#endif
