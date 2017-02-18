#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <complex.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "common.h"

typedef struct {
    struct rusage   u0;
    float           utime;
    float           stime;
    char            id[ID_LEN];
} cputimer_t;

void cputimer_start(cputimer_h h)
{
    cputimer_t *t = (cputimer_t*)h;
    getrusage(RUSAGE_SELF, &t->u0);
}

void cputimer_init(cputimer_h *h, const char *id)
{
    cputimer_t *t = (cputimer_t*)calloc(1, sizeof(cputimer_t));
    strncpy(t->id, id, ID_LEN*sizeof(char));
    *h = (cputimer_h)t;
}

void cputimer_pause(cputimer_h h)
{
    struct rusage u1;
    getrusage(RUSAGE_SELF, &u1);

    float tu, ts;
    cputimer_t *t = (cputimer_t*)h;
    tu = (u1.ru_utime.tv_sec-t->u0.ru_utime.tv_sec)*1E3
        + (u1.ru_utime.tv_usec - t->u0.ru_utime.tv_usec)*1E-3;    //in ms
    ts = (u1.ru_stime.tv_sec-t->u0.ru_stime.tv_sec)*1E3
        + (u1.ru_stime.tv_usec - t->u0.ru_stime.tv_usec)*1E-3;    //in ms
    t->utime += tu;
    t->stime += ts;
}

float cputimer_done(cputimer_h h)
{
    float total;
    cputimer_t *t = (cputimer_t*)h;
    printf(" CPU<%16s>: stime: %9.1f ms, utime: %9.1f\n", t->id, t->stime, t->utime);
    total = t->stime+t->utime;
    free(t);
    return total;
}

int isfftsize(int size, int *base, int n)
{
    int i;
    while(size!=1) {
        for(i=0; i<n; i++)
          if(size%base[i]==0) {
              size = size/base[i];
              break;
          }
        if(i==n)
          return 0;
    };
    return 1;
}


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
    cudaEvent_t     mark;
    char            id[ID_LEN];
} gputimer_t;

void gputimer_start(gputimer_h h)
{
    gputimer_t *t = (gputimer_t*)h;
    CCK(cudaEventRecord(t->mark, 0));
}

void gputimer_init(gputimer_h *h, const char *id)
{
    gputimer_t *t = (gputimer_t*)calloc(1, sizeof(gputimer_t));
    strncpy(t->id, id, ID_LEN*sizeof(char));
    CCK(cudaEventCreate(&t->mark));
    *h = (gputimer_h)t;
}

void gputimer_pause(gputimer_h h)
{
    gputimer_t *t = (gputimer_t*)h;
    float this_time; cudaEvent_t mark; 
    CCK(cudaEventCreate(&mark));
    CCK(cudaEventRecord(mark, 0));
    CCK(cudaEventSynchronize(mark));
    CCK(cudaEventElapsedTime(&this_time, t->mark, mark));
    t->sum += this_time;
}

float gputimer_done(gputimer_h h)
{
    float total;
    gputimer_t *t = (gputimer_t*)h;
    printf(" <%16s>: Elapsed time: %9.1f ms\n", t->id, t->sum);
    total = t->sum;
    CCK(cudaEventDestroy(t->mark));
    free(t);
    return total;
}

#ifdef __cplusplus
}
#endif
