#include "common.h"
#include <cufft.h>

typedef struct {
    cufftHandle     plan, r2c, c2r;
    int             n;          //fft logical length
    int             skip;       //at least 2*(n/2+1)
    int             howmany;
} cudaplan_t;

#define CCK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/**
 * repeat is used to control doing something or just measure overhead
 */
void cuda1d_r2c(cudaplan_h h, const float *pi, complex float *po, int repeat)
{
    cudaplan_t *t = (cudaplan_t*)h;
    cufftComplex *d_out; float *d_inp; 

    CCK(cudaMalloc((void**)&d_inp, sizeof(float)*t->skip*t->howmany));
    CCK(cudaMalloc((void**)&d_out, sizeof(float)*t->skip*t->howmany));
    CCK(cudaMemcpy(d_inp, pi, sizeof(float)*t->skip*t->howmany, cudaMemcpyHostToDevice));
    CCK(cudaMemset(d_out, 0, sizeof(float)*t->skip*t->howmany));

    for(int i=0; i<repeat; i++) {
        cufftResult_t code = cufftExecR2C(t->r2c, d_inp, d_out);
        assert(code==CUFFT_SUCCESS);
    }

    CCK(cudaMemcpy(po, d_out, sizeof(float)*t->skip*t->howmany, cudaMemcpyDeviceToHost));
    CCK(cudaFree(d_inp));
    CCK(cudaFree(d_out));
}

/**
 * repeat is used to control doing something or just measure overhead
 */
void cuda1d_c2r(cudaplan_h h, const complex float *pi, float *po, int repeat)
{
    cudaplan_t *t = (cudaplan_t*)h;
    cufftComplex *d_inp; float *d_out; 

    CCK(cudaMalloc((void**)&d_inp, sizeof(float)*t->skip*t->howmany));
    CCK(cudaMalloc((void**)&d_out, sizeof(float)*t->skip*t->howmany));
    CCK(cudaMemcpy(d_inp, pi, sizeof(float)*t->skip*t->howmany, cudaMemcpyHostToDevice));
    CCK(cudaMemset(d_out, 0, sizeof(float)*t->skip*t->howmany));

    for(int i=0; i<repeat; i++) {
        cufftResult_t code = cufftExecC2R(t->c2r, d_inp, d_out);
        assert(code==CUFFT_SUCCESS);
    }

    CCK(cudaMemcpy(po, d_out, sizeof(float)*t->skip*t->howmany, cudaMemcpyDeviceToHost));
    CCK(cudaFree(d_inp));
    CCK(cudaFree(d_out));
}

/**
 * repeat is used to control doing something or just measure overhead
 */
void cuda1d_c2c(cudaplan_h h, const complex float *pi, complex float *po, int repeat, int flag)
{
    cufftComplex *d_inp, *d_out;
    cudaplan_t *t = (cudaplan_t*)h;
    int direction = (flag>0)?(CUFFT_FORWARD):(CUFFT_INVERSE);

    CCK(cudaMalloc((void**)&d_inp, sizeof(float)*t->skip*t->howmany));
    CCK(cudaMalloc((void**)&d_out, sizeof(float)*t->skip*t->howmany));
    CCK(cudaMemcpy(d_inp, pi, sizeof(float)*t->skip*t->howmany, cudaMemcpyHostToDevice));
    CCK(cudaMemset(d_out, 0, sizeof(float)*t->skip*t->howmany));

    for(int i=0; i<repeat; i++) {
        cufftResult_t code = cufftExecC2C(t->plan, d_inp, d_out, direction);
        assert(code==CUFFT_SUCCESS);
    }

    CCK(cudaMemcpy(po, d_out, sizeof(float)*t->skip*t->howmany, cudaMemcpyDeviceToHost));
    CCK(cudaFree(d_inp));
    CCK(cudaFree(d_out));
}


/**
 * Note, for r2c and c2r, input/output need padding to the 2*nc float length
 */
void cuda1d_plan(cudaplan_h *h, int n, int skip, int howmany)
{
    cudaplan_t *t = (cudaplan_t*)calloc(1, sizeof(cudaplan_t));
    t->n = n; 
    t->skip = skip; 
    t->howmany = howmany;

    int nr=n, nc=n/2+1; cufftResult_t code;
    code = cufftPlanMany(&t->r2c, 1, &nr, &nr, 1, skip, &nc, 1, skip/2, CUFFT_R2C, howmany);
    assert(code==CUFFT_SUCCESS);
    code = cufftPlanMany(&t->c2r, 1, &nr, &nc, 1, skip/2, &nr, 1, skip, CUFFT_C2R, howmany);
    assert(code==CUFFT_SUCCESS);
    code = cufftPlanMany(&t->plan, 1, &n, &n, 1, skip/2, &n, 1, skip/2, CUFFT_C2C, howmany);
    *h = (cudaplan_h)(t);
}

void cuda1d_destroy(cudaplan_h h) 
{
    cufftResult_t code;
    cudaplan_t *t = (cudaplan_t*)h;
    code = cufftDestroy(t->r2c);  assert(code==CUFFT_SUCCESS);
    code = cufftDestroy(t->c2r);  assert(code==CUFFT_SUCCESS);
    code = cufftDestroy(t->plan); assert(code==CUFFT_SUCCESS);
}
