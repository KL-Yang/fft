#include <sys/time.h>
#include <sys/resource.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>

/**
 * Valid if the FFT gives correct result by compare with fftw!
 * */
void fftw_fft1d_plan(fftwf_plan *p, int nr)
{
    float *pr; complex float *pc; int nc=nr/2+1;
    pr = calloc(nr, sizeof(float));
    pc = calloc(nc, sizeof(complex float));
    p[0] = fftwf_plan_dft_r2c_1d(nr, pr, (fftwf_complex*)pc, FFTW_PATIENT);
    p[1] = fftwf_plan_dft_c2r_1d(nr, (fftwf_complex*)pc, pr, FFTW_PATIENT);
    free(pr); free(pc);
}

void fftw_fft1d_r2c(fftwf_plan p, float *pr, int nr, int nmemb, 
        complex float *po, int repeat)
{
    int nc = nr/2+1;
    for(int i=0; i<repeat; i++) 
        for(int j=0; j<nmemb; j++) 
            fftwf_execute_dft_r2c(p, pr+j*nr, (fftwf_complex*)(po+j*nc));
}

void fftw_fft1d_destroy(fftwf_plan *p)
{
    fftwf_destroy_plan(p[0]);
    fftwf_destroy_plan(p[1]);
}
