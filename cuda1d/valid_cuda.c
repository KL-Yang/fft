#include "common.h"

static void run_valid_r2c(int nr, int howmany)
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

    float max_r=0, max_i=0;
    for(int j=0, nc=skip/2; j<howmany; j++) {
        complex float *x1 = c1+j*nc;
        complex float *x2 = c2+j*nc;
        for(int i=0; i<nc; i++) {
            max_r = MAX(max_r, fabsf(crealf(x2[i])-crealf(x1[i])));
            max_i = MAX(max_i, fabsf(cimagf(x2[i])-cimagf(x1[i])));
            //printf("[%4d][%4d] <%9.1f,%9.1f>-<%9.1f,%9.1f>=<%9.1f,%9.1f>\n",
            //        j, i, crealf(x1[i]), cimagf(x1[i]), crealf(x2[i]), cimagf(x2[i]),
            //        crealf(x1[i]-x2[i]), cimagf(x1[i]-x2[i]));
        }
    }
    printf("Maximum difference %s(%9.2f, %9.2f)\n", __func__, max_r, max_i);
    fflush(stdout);
    free(pr); free(c1); free(c2);
}

static void run_valid_c2c(int n, int howmany)
{
    fftwplan_h fftw; cudaplan_h cuda;

    int skip = ALIGN4(n); 
    complex float *pi, *c1, *c2;

    pi = (complex float*)calloc(howmany*skip, sizeof(complex float));
    c1 = (complex float*)calloc(howmany*skip, sizeof(complex float));
    c2 = (complex float*)calloc(howmany*skip, sizeof(complex float));

    unsigned int seed=123;
    for(int i=0; i<howmany; i++) 
        for(int j=0; j<n; j++)
            pi[i*skip+j] = rand_r(&seed)*100.0f/RAND_MAX
                +I*rand_r(&seed)*100.0f/RAND_MAX;

    fftw1d_plan(&fftw, n);
    fftw1d_c2c(fftw, pi, skip, howmany, c1, skip, 1, 1);
    fftw1d_destroy(fftw);

    cuda1d_plan(&cuda, n, 2*skip, howmany, "c2c");
    cuda1d_c2c(cuda, pi, c2, 1, 1);
    cuda1d_destroy(cuda);

    float max_r=0, max_i=0;
    for(int j=0, nc=skip/2; j<howmany; j++) {
        complex float *x1 = c1+j*nc;
        complex float *x2 = c2+j*nc;
        for(int i=0; i<nc; i++) {
            max_r = MAX(max_r, fabsf(crealf(x2[i])-crealf(x1[i])));
            max_i = MAX(max_i, fabsf(cimagf(x2[i])-cimagf(x1[i])));
            //printf("[%4d][%4d] <%9.1f,%9.1f>-<%9.1f,%9.1f>=<%9.1f,%9.1f>\n",
            //        j, i, crealf(x1[i]), cimagf(x1[i]), crealf(x2[i]), cimagf(x2[i]),
            //        crealf(x1[i]-x2[i]), cimagf(x1[i]-x2[i]));
        }
    }
    printf("Maximum difference %s(%9.2f, %9.2f)\n", __func__, max_r, max_i);
    fflush(stdout);
    free(pi); free(c1); free(c2);

}

/**
 * Test against cpu fftw
 */
int main(int argc, char * argv[])
{
    int n=1024;
    run_valid_r2c(n, 10);
    run_valid_c2c(n, 10);
    return 0;
}
