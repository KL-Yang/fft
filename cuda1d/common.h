#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>

#define ID_LEN      16
#define ALIGN8(n)   ((n%8)?(n+8-n%8):(n))
#define ALIGN4(n)   ((n%4)?(n+4-n%4):(n))

typedef struct cputimer_struct * cputimer_h;
typedef struct gputimer_struct * gputimer_h;
typedef struct cudaplan_struct * cudaplan_h;
typedef struct fftwplan_struct * fftwplan_h;

int isfftsize(int size, int *base, int n);

void cputimer_init(cputimer_h *h, const char *id);
void cputimer_start(cputimer_h h);
void cputimer_pause(cputimer_h h);
float cputimer_done(cputimer_h h);

void gputimer_init(gputimer_h *h, const char *id);
void gputimer_start(gputimer_h h);
void gputimer_pause(gputimer_h h);
float gputimer_done(gputimer_h h);

void fftw1d_base_plan(fftwplan_h *h, int nr);
void fftw1d_base_r2c(fftwplan_h h, float *pr, int rdist, int nmemb, complex float *po, int cdist, int repeat);
void fftw1d_base_c2r(fftwplan_h h, complex float *po, int cdist, int nmemb, float *pr, int rdist, int repeat);
void fftw1d_base_destroy(fftwplan_h h);

void cuda1d_plan(cudaplan_h *h, int nr, int howmany);
void cuda1d_destroy(cudaplan_h h);
void cuda1d_r2c(cudaplan_h h, float *pi, int nr, int nmemb, complex float *po, int repeat);

#ifdef __cplusplus
}
#endif
