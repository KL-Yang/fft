#include <sys/time.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include "cpu_func.h"

void fftw1d_base_plan(fftwf_plan *p, int nr)
{
    float *pr; complex float *pc; int nc=nr/2+1;
    pr = calloc(nr, sizeof(float));
    pc = calloc(nc, sizeof(complex float));
    p[0] = fftwf_plan_dft_r2c_1d(nr, pr, (fftwf_complex*)pc, FFTW_PATIENT);
    p[1] = fftwf_plan_dft_c2r_1d(nr, (fftwf_complex*)pc, pr, FFTW_PATIENT);
    free(pr); free(pc);
}

void fftw1d_base_r2c(fftwf_plan p, float *pr, int rdist, int nmemb, 
        complex float *po, int cdist, int repeat)
{
    for(int i=0; i<repeat; i++) 
        for(int j=0; j<nmemb; j++) 
            fftwf_execute_dft_r2c(p, pr+j*rdist, (fftwf_complex*)(po+j*cdist));
}

void fftw1d_base_c2r(fftwf_plan p, complex float *po, int cdist, int nmemb,
        float *pr, int rdist, int repeat)
{
    for(int i=0; i<repeat; i++) 
        for(int j=0; j<nmemb; j++) 
            fftwf_execute_dft_c2r(p, (fftwf_complex*)(po+j*cdist), pr+j*rdist);
}

void fftw1d_base_destroy(fftwf_plan *p)
{
    fftwf_destroy_plan(p[0]);
    fftwf_destroy_plan(p[1]);
}
