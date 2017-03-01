#include "common.h"

/**
 * @param n      : FFT logical length
 * @param skip   : length between two transform
 * @param repeat : repeat times, 0 to measure overhead
 * */
static void run_test_r2c(int n, int skip, int howmany, int repeat)
{
    cudaplan_h plan;
    float *pr; complex float *pc;

    pr = calloc(howmany, skip*sizeof(float));
    pc = calloc(howmany, skip*sizeof(float));

    cuda1d_plan(&plan, n, skip, howmany, "r2c");
    cuda1d_r2c(plan, pr, pc, repeat);
    cuda1d_destroy(plan);

    free(pr); free(pc);
}

static void run_test_c2c(int n, int skip, int howmany, int repeat)
{
    cudaplan_h plan;
    complex float *pi, *po;

    pi = calloc(howmany, skip*sizeof(float));
    po = calloc(howmany, skip*sizeof(float));

    cuda1d_plan(&plan, n, skip, howmany, "c2c");
    cuda1d_c2c(plan, pi, po, repeat, 1);
    cuda1d_destroy(plan);

    free(pi); free(po);
}

float benchmark_tps(int n, int skip, int howmany, int repeat,
        void(*run_test)(int,int,int,int))
{
    gputimer_h t0, t1;
    float tps, time0, time1;

    run_test(n, skip, howmany, 0);   //warm up the plan!

    gputimer_init(&t0, "Overhead");
    gputimer_start(t0);
    run_test(n, skip, howmany, 0);
    gputimer_pause(t0);
    time0 = gputimer_done(t0);

    gputimer_init(&t1, "Measure");
    gputimer_start(t1);
    run_test(n, skip, howmany, repeat);
    gputimer_pause(t1);
    time1 = gputimer_done(t1);

    tps = (howmany*repeat*1000.0f)/(time1-time0);
    return tps;
}

/**
 * After first FFTW plan of some size, overhead of subsequent plan of 
 *      the same size is very small!
 * Usually should run several times and use the minimum one.
 * */
int main(int argc, char * argv[])
{
    if(argc!=4) {
        printf("Usage: %s n howmany repeat\n"
                "     n, logical length of Fast Fourier Transform\n"
                "     howmany, number of transform in batch\n"
                "     repeat, times to repeat the transform\n"
                "repeat=0 to measure the system overhead! compile with -O0!\n", argv[0]);
        exit(1);
    }
    int n, skip, howmany, repeat; float tps_r2c, tps_c2c;
    n = atoi(argv[1]); 
    howmany = atoi(argv[2]);
    repeat = atoi(argv[3]);

    printf("#%s: len=%8d, howmany=%8d, repeat=%8d\n", argv[0], n, howmany, repeat);
    fflush(stdout);

    skip = ALIGN8(2*(n/2+1));
    tps_r2c = benchmark_tps(n, skip, howmany, repeat, &run_test_r2c);

    skip = ALIGN8(2*n);
    tps_c2c = benchmark_tps(n, skip, howmany, repeat, &run_test_c2c);

    printf("#%8s  %12s  %12s\n", "FFTsize", "R2C_TPS", "C2C_TPS");
    printf("%9d  %12.1f  %12.1f\n", n, tps_r2c, tps_c2c);

    return 0;
}
