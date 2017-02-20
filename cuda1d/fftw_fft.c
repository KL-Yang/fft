#include <fftw3.h>
#include "common.h"

typedef struct {
    fftwf_plan      fwd;
    fftwf_plan      rvs;
} fftwplan_t;

void fftw1d_base_plan(fftwplan_h *h, int nr)
{
    fftwplan_t *tt;
    float *pr; complex float *pc; int nc=nr/2+1;
    tt = calloc(1, sizeof(fftwplan_t));
    pr = calloc(nr, sizeof(float));
    pc = calloc(nc, sizeof(complex float));
    tt->fwd = fftwf_plan_dft_r2c_1d(nr, pr, (fftwf_complex*)pc, FFTW_PATIENT);
    tt->rvs = fftwf_plan_dft_c2r_1d(nr, (fftwf_complex*)pc, pr, FFTW_PATIENT);
    free(pr); 
    free(pc);
    *h = (fftwplan_h)tt;
}

void fftw1d_base_r2c(fftwplan_h h, float *pr, int rdist, int nmemb, 
        complex float *po, int cdist, int repeat)
{
    fftwplan_t *t = (fftwplan_t*)h;
    for(int i=0; i<repeat; i++) 
        for(int j=0; j<nmemb; j++) 
            fftwf_execute_dft_r2c(t->fwd, pr+j*rdist, (fftwf_complex*)(po+j*cdist));
}

void fftw1d_base_c2r(fftwplan_h h, complex float *po, int cdist, int nmemb,
        float *pr, int rdist, int repeat)
{
    fftwplan_t *t = (fftwplan_t*)h;
    for(int i=0; i<repeat; i++) 
        for(int j=0; j<nmemb; j++) 
            fftwf_execute_dft_c2r(t->rvs, (fftwf_complex*)(po+j*cdist), pr+j*rdist);
}

void fftw1d_base_destroy(fftwplan_h h)
{
    fftwplan_t *t = (fftwplan_t*)h;
    fftwf_destroy_plan(t->fwd);
    fftwf_destroy_plan(t->rvs);
    free(t);
}
