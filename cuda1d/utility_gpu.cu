#include "cuda_common.h"

#define CCK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

typedef struct {
    float           sum;
    cudaStream_t    stream;
    cudaEvent_t     mark;
    char            id[ID_LEN];
} gputimer_t;

void gputimer_start(gputimer_h h)
{
    gputimer_t *t = (gputimer_t*)h;
    CCK(cudaEventRecord(t->mark, t->stream));
}

void gputimer_init(gputimer_h *h, cudaStream_t stream, const char *id)
{
    gputimer_t *t = (gputimer_t*)calloc(1, sizeof(gputimer_t));
    t->stream = stream;
    strncpy(t->id, id, ID_LEN*sizeof(char));
    CCK(cudaEventCreate(&t->mark));
    *h = (gputimer_h)t;
}

void gputimer_pause(gputimer_h h)
{
    gputimer_t *t = (gputimer_t*)h;
    float this_time; cudaEvent_t mark; 
    CCK(cudaEventCreate(&mark));
    CCK(cudaEventRecord(mark, t->stream));
    CCK(cudaEventSynchronize(mark));
    CCK(cudaEventElapsedTime(&this_time, t->mark, mark));
    t->sum += this_time;
}

void gputimer_done(gputimer_h h)
{
    gputimer_t *t = (gputimer_t*)h;
    printf(" S[%p]<%16s>: Elapsed time: %9.1f ms\n", t->stream, t->id, t->sum);
    CCK(cudaEventDestroy(t->mark));
    free(t);
}
