#include "common.h"
#include <cufft.h>

typedef struct {
    cufftHandle     plan;
    int             fftnr;
    int             rskip;      //
    int             cskip;
    int             howmany;
} cudaplan_t;

#define CCK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/**
 * repeat is used to control doing something or just measure overhead
 */
void cuda1d_r2c(cudaplan_h h, float *pi, complex float *po, int repeat)
{
    float *d_inp; cudaplan_t *t = (cudaplan_t*)h;
    cufftComplex *d_out; int nc = t->fftnr/2+1;

    CCK(cudaMalloc((void**)&d_inp, sizeof(float)*t->fftnr*t->howmany));
    CCK(cudaMalloc((void**)&d_out, sizeof(cufftComplex)*t->fftnr*t->howmany));
    CCK(cudaMemcpy(d_inp, pi, sizeof(float)*t->fftnr*t->howmany, cudaMemcpyHostToDevice));
    CCK(cudaMemset(d_out, 0, sizeof(float)*t->fftnr*t->howmany));

    for(int i=0; i<repeat; i++) {
        cufftResult_t code = cufftExecR2C(t->plan, d_inp, d_out);
        assert(code==CUFFT_SUCCESS);
    }

    CCK(cudaMemcpy(po, d_out, sizeof(cufftComplex)*nc*t->howmany, cudaMemcpyDeviceToHost));
    CCK(cudaFree(d_inp));
    CCK(cudaFree(d_out));
}

void cuda1d_plan(cudaplan_h *h, int fftnr, int rskip, int cskip, int howmany)
{
    cudaplan_t *t; int nc=fftnr/2+1;
    t = (cudaplan_t*)calloc(1, sizeof(cudaplan_t));
    t->fftnr = fftnr;
    t->rskip = rskip;
    t->cskip = cskip;
    t->howmany = howmany;
    cufftResult_t code = cufftPlanMany(&t->plan, 1, &fftnr, &fftnr, 1, rskip, &nc, 1, cskip, CUFFT_R2C, howmany);
    assert(code==CUFFT_SUCCESS);

    *h = (cudaplan_h)(t);
}

void cuda1d_destroy(cudaplan_h h) 
{
    cudaplan_t *t = (cudaplan_t*)h;
    cufftResult_t code;
    code = cufftDestroy(t->plan);
    assert(code==CUFFT_SUCCESS);
}
