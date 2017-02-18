#include "common.h"

/**
 * @param pool_size : buffer size in MB for fourier transform
 * @param fftnr     : FFT logical length
 * @param repeat    : repeat times, 0 to measure overhead
 * */
static void run_test(int fftnr, int howmany, int repeat)
{
    int i, j, anr, anc; 
    float *pr, *xr; complex float *pc, *xc;

    anr = ALIGN8(fftnr);
    anc = ALIGN4((fftnr/2+1));

    fftwf_plan plan[2];
    fftw1d_base_plan(plan, fftnr);

    //allocate and warm up the buffer!
    pr = calloc(howmany, anr*sizeof(float));
    pc = calloc(howmany, anc*sizeof(complex float));
    for(i=0, xr=pr; i<howmany; i++)
        for(j=0; j<anr; j++, xr++)
            *xr = 1.0f;
    for(i=0, xc=pc; i<howmany; i++)
        for(j=0; j<anc; j++, xc++)
            *xc = 1.0f;

    fftw1d_base_r2c(plan[0], pr, anr, howmany, pc, anc, repeat);

    fftw1d_base_destroy(plan);
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
    int nr, nc, anr, anc, howmany, repeat, verbose=0;
    nr = atoi(argv[1]); anr = ALIGN8(nr);
    nc = nr/2+1; anc = ALIGN4(nc);
    howmany = atoi(argv[2]);
    repeat = atoi(argv[3]);

    cputimer_h t0, t1;
    float u0, s0, u1, s1;

    run_test(nr, howmany, 0);   //warm up the plan!

    cputimer_init(&t0, "Overhead");
    cputimer_start(t0);
    run_test(nr, howmany, 0);
    cputimer_pause(t0);
    u0 = cputimer_utime(t0);
    s0 = cputimer_stime(t0);
    cputimer_done(t0);

    cputimer_init(&t1, "Measure");
    cputimer_start(t1);
    run_test(nr, howmany, repeat);
    cputimer_pause(t1);
    u1 = cputimer_utime(t1);
    s1 = cputimer_stime(t1);
    cputimer_done(t1);

    if(verbose) {
        printf(">CMD: %s nr=%d howmany=%d repeat=%d\n", argv[0], nr, howmany, repeat);
        printf(">INF: aligned nr=%d(%d) nc=%d(%d)\n", anr, nr, anc, nc);
        printf("utime=%9.1f stime=%9.1f\n", u1-u0, s1-s0);
    }

    float tps = (howmany*repeat*1000.0f)/(u1+s1-u0-s0);
    printf("%8d, %8d, %8d, %9.1f\n", nr, howmany, repeat, tps);

    return 0;
}
