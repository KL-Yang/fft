#include "common.h"

/**
 * @param nr     : FFT logical length
 * @param len    : length between two transform
 * @param repeat : repeat times, 0 to measure overhead
 * */
static void run_test(int nr, int len, int howmany, int repeat)
{
    cudaplan_h plan;
    float *pr; complex float *pc;

    pr = calloc(howmany, len*sizeof(float));
    pc = calloc(howmany, len*sizeof(float));

    cuda1d_plan(&plan, nr, len, howmany);
    cuda1d_r2c(plan, pr, pc, repeat);
    cuda1d_destroy(plan);

    free(pr); free(pc);
}

/**
 * After first FFTW plan of some size, overhead of subsequent plan of 
 *      the same size is very small!
 * Usually should run several times and use the minimum one.
 * */
int main(int argc, char * argv[])
{
    if(argc!=4) {
        printf("Usage: %s nr howmany repeat\n"
                "     nr, logical length of Fast Fourier Transform\n"
                "     howmany, number of transform in batch\n"
                "     repeat, times to repeat the transform\n"
                "repeat=0 to measure the system overhead! compile with -O0!\n", argv[0]);
        exit(1);
    }
    int nr, len, howmany, repeat, verbose=0;
    nr = atoi(argv[1]); len = ALIGN8(2*(nr/2+1));
    howmany = atoi(argv[2]);
    repeat = atoi(argv[3]);

    gputimer_h t0, t1;
    float time0, time1;

    run_test(nr, len, howmany, 0);   //warm up the plan!

    gputimer_init(&t0, "Overhead");
    gputimer_start(t0);
    run_test(nr, len, howmany, 0);
    gputimer_pause(t0);
    time0 = gputimer_done(t0);

    gputimer_init(&t1, "Measure");
    gputimer_start(t1);
    run_test(nr, len, howmany, repeat);
    gputimer_pause(t1);
    time1 = gputimer_done(t1);

    if(verbose) {
        printf(">CMD: %s nr=%d howmany=%d repeat=%d\n", argv[0], nr, howmany, repeat);
        printf(">INF: aligned len=%d time=%9.1fms\n", len, time1-time0);
    }

    float tps = (howmany*repeat*1000.0f)/(time1-time0);
    printf("%8d, %8d, %8d, %14.1f\n", nr, howmany, repeat, tps);

    return 0;
}
