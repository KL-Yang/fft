#include "common.h"
#include <cufft.h>

typedef struct {
    cufftHandle     plan;
    int             n;          //fft logical length
    int             rskip;       //at least 2*(n/2+1)
    int             howmany;
    char            type[4];
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
    assert(strcasecmp(t->type, "r2c")==0);

    CCK(cudaMalloc((void**)&d_inp, sizeof(float)*t->rskip*t->howmany));
    CCK(cudaMalloc((void**)&d_out, sizeof(float)*t->rskip*t->howmany));
    CCK(cudaMemcpy(d_inp, pi, sizeof(float)*t->rskip*t->howmany, cudaMemcpyHostToDevice));
    CCK(cudaMemset(d_out, 0, sizeof(float)*t->rskip*t->howmany));

    for(int i=0; i<repeat; i++) {
        cufftResult_t code = cufftExecR2C(t->plan, d_inp, d_out);
        assert(code==CUFFT_SUCCESS);
    }

    CCK(cudaMemcpy(po, d_out, sizeof(float)*t->rskip*t->howmany, cudaMemcpyDeviceToHost));
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
    assert(strcasecmp(t->type, "c2r")==0);

    CCK(cudaMalloc((void**)&d_inp, sizeof(float)*t->rskip*t->howmany));
    CCK(cudaMalloc((void**)&d_out, sizeof(float)*t->rskip*t->howmany));
    CCK(cudaMemcpy(d_inp, pi, sizeof(float)*t->rskip*t->howmany, cudaMemcpyHostToDevice));
    CCK(cudaMemset(d_out, 0, sizeof(float)*t->rskip*t->howmany));

    for(int i=0; i<repeat; i++) {
        cufftResult_t code = cufftExecC2R(t->plan, d_inp, d_out);
        assert(code==CUFFT_SUCCESS);
    }

    CCK(cudaMemcpy(po, d_out, sizeof(float)*t->rskip*t->howmany, cudaMemcpyDeviceToHost));
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

    CCK(cudaMalloc((void**)&d_inp, sizeof(float)*t->rskip*t->howmany));
    CCK(cudaMalloc((void**)&d_out, sizeof(float)*t->rskip*t->howmany));
    CCK(cudaMemcpy(d_inp, pi, sizeof(float)*t->rskip*t->howmany, cudaMemcpyHostToDevice));
    CCK(cudaMemset(d_out, 0, sizeof(float)*t->rskip*t->howmany));

    for(int i=0; i<repeat; i++) {
        cufftResult_t code = cufftExecC2C(t->plan, d_inp, d_out, direction);
        assert(code==CUFFT_SUCCESS);
    }

    CCK(cudaMemcpy(po, d_out, sizeof(float)*t->rskip*t->howmany, cudaMemcpyDeviceToHost));
    CCK(cudaFree(d_inp));
    CCK(cudaFree(d_out));
}

/**
 * Note, for r2c and c2r, input/output need padding to the 2*nc float length
 * For r2c or c2r, skip count float!
 * For c2c skip count complex float!
 */
void cuda1d_plan(cudaplan_h *h, int n, int rskip, int howmany, const char *type)
{
    cudaplan_t *t = (cudaplan_t*)calloc(1, sizeof(cudaplan_t));
    t->n = n; t->rskip = rskip; 
    t->howmany = howmany;
    strncpy(t->type, type, 3);

    int nr, nc, cskip=rskip/2; cufftResult_t code;
    if(strcasecmp(type, "r2c")==0) {
        nr = n; nc = nr/2+1;
        code = cufftPlanMany(&t->plan, 1, &nr, &nr, 1, rskip, &nc, 1, cskip, CUFFT_R2C, howmany);
        assert(code==CUFFT_SUCCESS);
    } else
    if(strcasecmp(type, "c2r")==0) {
        nr = n; nc = nr/2+1;
        code = cufftPlanMany(&t->plan, 1, &nr, &nc, 1, cskip, &nr, 1, rskip, CUFFT_C2R, howmany);
        assert(code==CUFFT_SUCCESS);
    } else
    if(strcasecmp(type, "c2c")==0) {
        nc = n; 
        code = cufftPlanMany(&t->plan, 1, &nc, &nc, 1, cskip, &nc, 1, cskip, CUFFT_C2C, howmany);
        assert(code==CUFFT_SUCCESS);
    } else abort();
    *h = (cudaplan_h)(t);
}

void cuda1d_destroy(cudaplan_h h) 
{
    cufftResult_t code;
    cudaplan_t *t = (cudaplan_t*)h;
    code = cufftDestroy(t->plan); assert(code==CUFFT_SUCCESS);
}
