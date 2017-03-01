#include "common.h"

static void run_valid(int nr, int howmany)
{
    fftwplan_h fftw; cudaplan_h cuda;

    int skip = ALIGN8(2*(nr/2+1)); 
    float *pr; complex float *c1, *c2;

    pr = (float*)calloc(howmany*skip, sizeof(float));
    c1 = (complex float*)calloc(howmany*skip, sizeof(float));
    c2 = (complex float*)calloc(howmany*skip, sizeof(float));

    unsigned int seed=123;
    for(int i=0; i<howmany; i++) 
        for(int j=0; j<nr; j++)
            pr[i*skip+j] = rand_r(&seed)*100.0f/RAND_MAX;

    fftw1d_plan(&fftw, nr);
    fftw1d_r2c(fftw, pr, skip, howmany, c1, skip/2, 1);
    fftw1d_destroy(fftw);

    cuda1d_plan(&cuda, nr, skip, howmany, "r2c");
    cuda1d_r2c(cuda, pr, c2, 1);
    cuda1d_destroy(cuda);

    for(int j=0; j<howmany; j++) {
        int anc = skip/2;
        for(int i=0; i<anc; i++) {
            printf("[%4d][%4d] <%9.1f,%9.1f>-<%9.1f,%9.1f>=<%9.1f,%9.1f>\n",
                    j, i, crealf(c1[j*anc+i]), cimagf(c1[j*anc+i]),
                    crealf(c2[j*anc+i]), cimagf(c2[j*anc+i]),
                    crealf(c1[j*anc+i]-c2[j*anc+i]), 
                    cimagf(c1[j*anc+i]-c2[j*anc+i]));
        }
    }
    fflush(stdout);
    free(pr); free(c1); free(c2);
}

/**
 * Test against cpu fftw
 */
int main(int argc, char * argv[])
{
    int nr=1024;
    run_valid(nr, 10);
    return 0;
}
