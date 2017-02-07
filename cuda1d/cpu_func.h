#ifndef H_FUNC_CPU_FFT_KYANG
#define H_FUNC_CPU_FFT_KYANG
#ifdef __cplusplus
extern "C" {
#endif
#include <complex.h>
#include <fftw3.h>

void fftw1d_base_plan(fftwf_plan *p, int nr);
void fftw1d_base_r2c(fftwf_plan p, float *pr, int rdist, int nmemb, complex float *po, int cdist, int repeat);
void fftw1d_base_c2r(fftwf_plan p, complex float *po, int cdist, int nmemb, float *pr, int rdist, int repeat);
void fftw1d_base_destroy(fftwf_plan *p);

void fftw1d_many_plan(fftwf_plan *p, int nr, int howmany);
void fftw1d_many_r2c(fftwf_plan p, float *pr, complex float *po, int repeat);
void fftw1d_many_c2r(fftwf_plan p, complex float *po, float *pr, int repeat);
void fftw1d_many_destroy(fftwf_plan *p);

#ifdef __cplusplus
}
#endif
#endif
