#include <math.h>
#include <fftw3.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pmmintrin.h>

float get_ms_rusage(struct rusage *r0, struct rusage *r1)
{
    float tu, ts;
    tu = (r1->ru_utime.tv_sec  - r0->ru_utime.tv_sec)*1E3
       + (r1->ru_utime.tv_usec - r0->ru_utime.tv_usec)*1E-3;    //in ms
    ts = (r1->ru_stime.tv_sec  - r0->ru_stime.tv_sec)*1E3
       + (r1->ru_stime.tv_usec - r0->ru_stime.tv_usec)*1E-3;    //in ms
    return (tu+ts);
}


int main(int argc, char *argv[])
{

    fftwf_plan p1, p4;
    float *inp4, *out4;
    int i, nr, nc, xr, xc;

    nr = 2048;
    nc = nr/2+1;
    xc = nc+nc%2;
    xr = (nr%4==0)?(nr):(nr+4-nr%4);

    inp4 = calloc(4*xr, sizeof(float));
    out4 = calloc(4*xc, sizeof(fftwf_complex));

    p1 = fftwf_plan_dft_r2c_1d(nr, inp4, (fftwf_complex *)out4, FFTW_PATIENT);
    p4 = fftwf_plan_many_dft_r2c(1, &nr, 4, inp4, NULL, 1, xr, 
                            (fftwf_complex*)out4, NULL, 1, xc, FFTW_PATIENT);

    struct rusage u0, u1, u2;

    getrusage(RUSAGE_SELF, &u0);
    for(i=0; i<28800*10; i++) {
        fftwf_execute_dft_r2c(p1, inp4+0*xr, (fftwf_complex*)(out4+2*0*xc));
        fftwf_execute_dft_r2c(p1, inp4+1*xr, (fftwf_complex*)(out4+2*1*xc));
        fftwf_execute_dft_r2c(p1, inp4+2*xr, (fftwf_complex*)(out4+2*2*xc));
        fftwf_execute_dft_r2c(p1, inp4+3*xr, (fftwf_complex*)(out4+2*3*xc));
    }
    getrusage(RUSAGE_SELF, &u1);
    for(i=0; i<28800*10; i++) {
        fftwf_execute_dft_r2c(p4, inp4, (fftwf_complex*)out4);
    }
    getrusage(RUSAGE_SELF, &u2);

    float time_fftw, time_nfft;
    time_fftw = get_ms_rusage(&u0, &u1);
    time_nfft = get_ms_rusage(&u1, &u2);

    printf("  p1 : %f ms\n", time_fftw);
    printf("  p4 : %f ms\n", time_nfft);

    free(inp4);
    free(out4);
    fftwf_destroy_plan(p1);
    fftwf_destroy_plan(p4);

    return 0;
}
