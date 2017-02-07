#include "common.h"
#include "cpu_func.h"

static void run_test(int nr, int howmany, int repeat)
{
    int anr = ALIGN8(nr);
    int anc = ALIGN4((nr/2+1));

    fftwf_plan plan[2];
    fftw1d_base_plan(plan, nr);

    float *pr; complex float *pc;
    pr = calloc(howmany, anr*sizeof(float));
    pc = calloc(howmany, anc*sizeof(complex float));
    fftw1d_base_r2c(plan[0], pr, anr, howmany, pc, anc, repeat);
    free(pr); free(pc);

    fftw1d_base_destroy(plan);
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
    int nr, nc, anr, anc, howmany, repeat;
    nr = atoi(argv[1]); anr = ALIGN8(nr);
    nc = nr/2+1; anc = ALIGN4(nc);
    howmany = atoi(argv[2]);
    repeat = atoi(argv[3]);
    printf(">CMD: %s nr=%d howmany=%d repeat=%d\n", argv[0], nr, howmany, repeat);
    printf(">INF: aligned nr=%d(%d) nc=%d(%d)\n", anr, nr, anc, nc);

    cputimer_h t0;
    cputimer_init(&t0, "Basic");
    cputimer_start(t0);
    run_test(nr, howmany, repeat);
    cputimer_pause(t0);
    cputimer_done(t0);
    return 0;
}
