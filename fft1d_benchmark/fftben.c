#include <sys/time.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <string.h>

float get_ms_rusage(struct rusage *r0, struct rusage *r1)
{
    float tu, ts;
    tu = (r1->ru_utime.tv_sec  - r0->ru_utime.tv_sec)*1E3
       + (r1->ru_utime.tv_usec - r0->ru_utime.tv_usec)*1E-3;    //in ms
    ts = (r1->ru_stime.tv_sec  - r0->ru_stime.tv_sec)*1E3
       + (r1->ru_stime.tv_usec - r0->ru_stime.tv_usec)*1E-3;    //in ms
    return (tu+ts);
}

int rand_range(int max)
{
    return ((int)nearbyint((max-1)*((rand()*1.0)/(RAND_MAX*1.0))));
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

float benchmark_r2c_tps(int pool_size, int fftsize)
{
    fftwf_plan p;
    float *pool, *work;
    struct rusage u0, u1;
    int i, k, nr, nx, nn;

    nr = fftsize; (nr%4==0)?(nx=nr):(nx=nr+(4-nr%4));
    nn = (1024L*1024L*pool_size)/nx/sizeof(float);

    pool = calloc(pool_size, 1024*1024);
    work = calloc(nr/2+1, sizeof(complex float));
    p = fftwf_plan_dft_r2c_1d(nr, pool, (fftwf_complex*)work, FFTW_PATIENT);

    memset(pool, 0, 1024L*1024L*pool_size);     //avoid page fault
    getrusage(RUSAGE_SELF, &u0);
    for(i=0; i<nn*10; i++) {
        k = rand_range(nn);
        fftwf_execute_dft_r2c(p, (float*)(pool+k*nx), (fftwf_complex*)work);
    }
    getrusage(RUSAGE_SELF, &u1);

    fftwf_destroy_plan(p);
    free(pool);
    free(work);

    return ((nn*10*1E3)/get_ms_rusage(&u0, &u1));
}

float benchmark_c2c_tps(int pool_size, int fftsize)
{
    fftwf_plan p;
    float *pool, *work;
    struct rusage u0, u1;
    int i, k, nr, nx, nn;

    nr = fftsize; nx=nr+nr%2;
    nn = (1024L*1024L*pool_size)/nx/sizeof(complex float);

    pool = calloc(pool_size, 1024*1024);
    work = calloc(nr, sizeof(complex float));
    p = fftwf_plan_dft_1d(nr, (fftwf_complex*)pool, (fftwf_complex*)work, FFTW_FORWARD, FFTW_PATIENT);

    memset(pool, 0, 1024L*1024L*pool_size);     //avoid page fault
    getrusage(RUSAGE_SELF, &u0);
    for(i=0; i<nn*10; i++) {
        k = rand_range(nn);
        fftwf_execute_dft(p, (fftwf_complex*)(pool+2*k*nx), (fftwf_complex*)work);
    }
    getrusage(RUSAGE_SELF, &u1);

    fftwf_destroy_plan(p);
    free(pool);
    free(work);

    return ((nn*10*1E3)/get_ms_rusage(&u0, &u1));
}

/**
 * Randomly choosen transfer from a 100MB buffer
 * */
int main(int argc, char * argv[])
{
    if(argc<3) {
        printf("Usage: %s poolsize_MB fftsize\n"
               "  FFT benchmark by randomly and repeatly choose a buffer from the pool and perform FFT"
               "then give how many transform can be done per seconds for r2c and c2c transfrom.\n",
               argv[0]);
        return 0;
    }

    float r2c_tps, c2c_tps;
    int i, nr, pz,  base[]={2,3,5,7};

    pz = atoi(argv[1]);     //pool size in MB
    nr = atoi(argv[2]);     //fftsize

    printf("#%s Result:\n"
           "#%8s  %12s  %12s\n", argv[0], "FFTsize", "R2C_TPS", "C2C_TPS");

    for(i=16; i<=nr; i++) {
        if(isfftsize(i, base, 4)) {
            r2c_tps = benchmark_r2c_tps(pz, i);
            c2c_tps = benchmark_c2c_tps(pz, i);
            printf("%8d  %12.1f  %12.1f\n", i, r2c_tps, c2c_tps);
            fflush(stdout);
        }
    }
    return 0;
}
