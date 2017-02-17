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
    cputimer_t *t = calloc(1, sizeof(cputimer_t));
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

float cputimer_utime(cputimer_h h)
{
    cputimer_t *t = (cputimer_t*)h;
    return (t->utime);
}

float cputimer_stime(cputimer_h h)
{
    cputimer_t *t = (cputimer_t*)h;
    return (t->stime);
}

void cputimer_done(cputimer_h h)
{
    cputimer_t *t = (cputimer_t*)h;
    printf(" CPU<%16s>: stime: %9.1f ms, utime: %9.1f\n", t->id,
            t->stime, t->utime);
    free(t);
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


