#include <sys/time.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include "common.h"
#include "cpu_func.h"

/**
 * Note the buffer is properly aligned to AVX 8*float
 * */
void fftw1d_many_plan(fftwf_plan *p, int nr, int howmany)
{
    float *pr; complex float *pc; 
    int nc=nr/2+1, anr=ALIGN8(nr), anc=ALIGN4(nc);

    pr = calloc(howmany, anr*sizeof(float));
    pc = calloc(howmany, anc*sizeof(complex float));
    p[0] = fftwf_plan_many_dft_r2c(1, &nr, howmany, pr, NULL, 1, anr, (fftwf_complex*)pc, 
            NULL, 1, anc, FFTW_PATIENT);
    p[1] = fftwf_plan_many_dft_c2r(1, &nr, howmany, (fftwf_complex*)pc, NULL, 1, anc, pr,
            NULL, 1, anr, FFTW_PATIENT);
    free(pr); free(pc);
}

void fftw1d_many_r2c(fftwf_plan p, float *pr, complex float *po, int repeat)
{
    for(int i=0; i<repeat; i++) 
        fftwf_execute_dft_r2c(p, pr, (fftwf_complex*)po);
}

void fftw1d_many_c2r(fftwf_plan p, complex float *po, float *pr, int repeat)
{
    for(int i=0; i<repeat; i++) 
        fftwf_execute_dft_c2r(p, (fftwf_complex*)po, pr);
}

void fftw1d_many_destroy(fftwf_plan *p)
{
    fftwf_destroy_plan(p[0]);
    fftwf_destroy_plan(p[1]);
}
