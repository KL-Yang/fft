#include "common.h"

static void run_valid(int nr, int howmany)
{
    fftwplan_h fftw; cudaplan_h cuda;

    float *pr; complex float *c1, *c2;
    int nc = nr/2+1; int anr = ALIGN8(nr); int anc = ALIGN4(nc);

    pr = (float*)calloc(howmany*nr, sizeof(float));
    c1 = (complex float*)calloc(howmany*nc, sizeof(complex float));
    c2 = (complex float*)calloc(howmany*nc, sizeof(complex float));

    unsigned int seed=123;
    for(int i=0; i<anr*howmany; i++) 
        pr[i] = rand_r(&seed)*100.0f/RAND_MAX;

    fftw1d_plan(&fftw, nr);
    fftw1d_r2c(fftw, pr, anr, howmany, c1, anc, 1);
    fftw1d_destroy(fftw);

    cuda1d_plan(&cuda, nr, anr, anc, howmany);
    cuda1d_r2c(cuda, pr, c2, 1);
    cuda1d_destroy(cuda);

    for(int j=0; j<howmany; j++) {
        for(int i=0; i<anc; i++) {
            printf("[%4d][%4d] <%9.1f,%9.1f>-<%9.1f,%9.1f>=<%9.1f,%9.1f>\n",
                    j, i, crealf(c1[j*anc+i]), cimagf(c1[j*anc+i]),
                    crealf(c2[j*anc+i]), cimagf(c2[j*anc+i]),
                    crealf(c1[j*anc+i]-c2[j*anc+i]), 
                    cimagf(c1[j*anc+i]-c2[j*anc+i]));
        }
        if(j==1)
            break;
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
