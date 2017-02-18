#include <math.h>
#include "common.h"

static void run_valid(int nr)
{
    int nc = nr/2+1;
    fftwf_plan plan[2];
    fftw1d_base_plan(plan, nr);

    float *pr; complex float *pc;
    pr = calloc(nr, sizeof(float));
    pc = calloc(nc, sizeof(complex float));

    float freq = 10.0f;
    printf("#frequency=%f\n", freq);
    for(int i=0; i<nr; i++) {
        pr[i] = cosf(2*M_PI*freq*i/nr);
        //printf(" %5d  %f\n", i, pr[i]);
    }

    fftw1d_base_r2c(plan[0], pr, nr, 1, pc, nc, 1);
    for(int i=0; i<nc; i++) 
        printf(" %5d  %f\n", i, cabsf(pc[i]));

    fftw1d_base_destroy(plan);
    free(pr); free(pc);
}

int main()
{
    run_valid(1024);
    return 0;
}
