#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <assert.h>
#include <cufft.h>

#define CCK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define ID_LEN  16

typedef struct gputimer_struct * gputimer_h;

/**
 * repeat is used to control doing something or just measure overhead
 */
void cuda_fft1d_r2c(cufftHandle plan, float *pi, int nr, int nmemb, complex float *po, int repeat)
{
    float *d_inp; 
    cufftComplex *d_out; int nc = nr/2+1;

    CCK(cudaMalloc((void**)&d_inp, sizeof(float)*nr*nmemb));
    CCK(cudaMalloc((void**)&d_out, sizeof(cufftComplex)*nr*nmemb));
    CCK(cudaMemcpy(d_inp, pi, sizeof(float)*nr*nmemb, cudaMemcpyHostToDevice));

    for(int i=0; i<repeat; i++) {
        cufftResult_t code = cufftExecR2C(plan, d_inp, d_out);
        assert(code==CUFFT_SUCCESS);
    }

    CCK(cudaMemcpy(po, d_out, sizeof(cufftComplex)*nc*nmemb, cudaMemcpyDeviceToHost));
    CCK(cudaFree(d_inp));
    CCK(cudaFree(d_out));
}

cufftHandle cuda_fft1d_plan(int nr, int howmany)
{
    float elapsedTime; cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventRecord(start,0);

    cufftHandle plan; int nc=nr/2+1;

    cufftResult_t code = cufftPlanMany(&plan, 1, &nr, &nr, 1, nr, &nc, 1, nc, CUFFT_R2C, howmany);
    assert(code==CUFFT_SUCCESS);

    cudaEventCreate(&stop); cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Elapsed time : %f ms\n" ,elapsedTime);

    return plan;
}

void cuda_fft1d_destroy(cufftHandle plan) 
{
    cufftResult_t code;
    code = cufftDestroy(plan);
    assert(code==CUFFT_SUCCESS);
}