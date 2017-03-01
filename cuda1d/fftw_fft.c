#include <fftw3.h>
#include "common.h"

typedef struct {
    fftwf_plan      r2c, c2r;   //real/complex
    fftwf_plan      fwd, rvs;   //complex transform
} fftwplan_t;

void fftw1d_plan(fftwplan_h *h, int n)
{
    fftwplan_t *tt;
    void *pi; void *po; int nr, nc;
    tt = calloc(1, sizeof(fftwplan_t));

    //1. real/complex
    nr = n; nc=nr/2+1;
    pi = calloc(nr, sizeof(float));
    po = calloc(nc, sizeof(complex float));
    tt->r2c = fftwf_plan_dft_r2c_1d(nr, pi, po, FFTW_PATIENT);
    tt->c2r = fftwf_plan_dft_c2r_1d(nr, po, pi, FFTW_PATIENT);
    free(pi); free(po);

    //2. complex/complex
    pi = calloc(n, sizeof(complex float));
    po = calloc(n, sizeof(complex float));
    tt->fwd = fftwf_plan_dft_1d(n, pi, po, FFTW_FORWARD, FFTW_PATIENT);
    tt->rvs = fftwf_plan_dft_1d(n, pi, po, FFTW_BACKWARD, FFTW_PATIENT);
    free(pi); free(po);

    *h = (fftwplan_h)tt;
}

void fftw1d_r2c(fftwplan_h h, float *pr, int rdist, int nmemb, 
        complex float *po, int cdist, int repeat)
{
    fftwplan_t *t = (fftwplan_t*)h;
    for(int i=0; i<repeat; i++) 
        for(int j=0; j<nmemb; j++) 
            fftwf_execute_dft_r2c(t->r2c, pr+j*rdist, (fftwf_complex*)(po+j*cdist));
}

void fftw1d_c2r(fftwplan_h h, complex float *po, int cdist, int nmemb,
        float *pr, int rdist, int repeat)
{
    fftwplan_t *t = (fftwplan_t*)h;
    for(int i=0; i<repeat; i++) 
        for(int j=0; j<nmemb; j++) 
            fftwf_execute_dft_c2r(t->c2r, (fftwf_complex*)(po+j*cdist), pr+j*rdist);
}

void fftw1d_c2c(fftwplan_h h, complex float *pi, int idist, int nmemb, 
        complex float *po, int odist, int repeat, int flag)
{
    fftwplan_t *t = (fftwplan_t*)h;
    if(flag>0) {
        for(int i=0; i<repeat; i++) 
            for(int j=0; j<nmemb; j++) 
                fftwf_execute_dft(t->fwd, (fftwf_complex*)(pi+j*idist), 
                        (fftwf_complex*)(po+j*odist));
    } else {
        for(int i=0; i<repeat; i++) 
            for(int j=0; j<nmemb; j++) 
                fftwf_execute_dft(t->rvs, (fftwf_complex*)(pi+j*idist), 
                        (fftwf_complex*)(po+j*odist));
    }
}

void fftw1d_destroy(fftwplan_h h)
{
    fftwplan_t *t = (fftwplan_t*)h;
    fftwf_destroy_plan(t->r2c);
    fftwf_destroy_plan(t->c2r);
    free(t);
}
