#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <string.h>
#include <pmmintrin.h>

unsigned int seed = 100;

float rand_float(float range)
{
    return ((rand_r(&seed)*range)/((float)RAND_MAX));
}

void x86_split(const complex float *C, int nr, complex float *AC,
               complex float *BC)
{
    int i, nc = nr/2+1;
    AC[0] = crealf(C[0]);
    BC[0] = cimagf(C[0]);
    for(i=1; i<nc; i++) {
        AC[i] =   (conjf(C[nr-i])+C[i])/2.0f;
        BC[i] = I*(conjf(C[nr-i])-C[i])/2.0f;
    }
}

inline static __m128 cmul_sse3(__m128 a, __m128 b)
{
    __m128 c;
     c = _mm_moveldup_ps(a);
     a = _mm_movehdup_ps(a);
     c = _mm_mul_ps(c, b);
     b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2,3,0,1));
     a = _mm_mul_ps(a, b);
    return _mm_addsub_ps(c, a);
}
/**
 * output the convolve result in frequency domain
 * */
void two_way_conv(const complex float *C, int nr, complex float *D)
{
    int nc = nr/2+1;
    int nn = (nc%2)?(nc):(nc-1);
    D[0] = ((float*)C)[0]*((float*)C)[1]; //DC component

    for(int i=1; i<nn; i=i+2) {
        __m128 ci = _mm_loadu_ps((float*)(C+i));
        __m128 cj = _mm_mul_ps(_mm_loadu_ps((float*)(C+nr-1-i)), 
               /*conjugate*/   _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f)); 
        cj = _mm_shuffle_ps(cj, cj, _MM_SHUFFLE(1,0,3,2));
        __m128 pl = _mm_add_ps(cj, ci);
        __m128 mi = _mm_sub_ps(cj, ci);
        __m128 xx = cmul_sse3(pl, mi); //multiple (a+b)(a-b)
        xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,3,0,1));
        xx = _mm_mul_ps(xx, _mm_set_ps(0.25f, -0.25f, 0.25f, -0.25f));
        _mm_storeu_ps((float*)(D+i), xx); //multiple by I/4
    }
    if(nn!=nc) {
        complex float A =   (conjf(C[nr-nn])+C[nn])/2.0f;
        complex float B = I*(conjf(C[nr-nn])-C[nn])/2.0f;
        D[nn] = A*B;
    }
}

int main()
{
    int i, nr, nc;
    fftwf_plan cp, fp, rp;

    nr = 4;
    nc = nr/2+1;
    float a[nr], b[nr];
    complex float c[nr], A[nc], B[nc], C[nr], AC[nc], BC[nc], D0[nc], D1[nc];

    memset(a, 0, nr*sizeof(float));
    memset(c, 0, nr*sizeof(complex float));
    memset(C, 0, nr*sizeof(complex float));
    memset(A, 0, nc*sizeof(complex float));

    fp = fftwf_plan_dft_r2c_1d(nr, a, (fftwf_complex*)A, FFTW_ESTIMATE);
    cp = fftwf_plan_dft_1d(nr, (fftwf_complex*)c, (fftwf_complex*)C, FFTW_FORWARD, FFTW_ESTIMATE);

    for(i=0; i<nr; i++) {
        a[i] = rand_float(10.0f);
        b[i] = rand_float(10.0f);
        c[i] = a[i]+I*b[i];
    }

    fftwf_execute_dft(cp, (fftwf_complex*)c, (fftwf_complex*)C);
    //Here is how to rebuild the r2c from c2c!
    x86_split(C, nr, AC, BC);

    fftwf_execute_dft_r2c(fp, a, (fftwf_complex*)A);
    fftwf_execute_dft_r2c(fp, b, (fftwf_complex*)B);

    printf("%10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n", "A-r-DIR", "A-i-DIR", "A-r-CPX", "A-i-CPX",
                                                       "B-r-DIR", "B-i-DIR", "B-r-CPX", "B-i-CPX");
    for(i=0; i<nc; i++) {
        printf("%10f  %10f  %10f  %10f  ", crealf(A[i]), cimagf(A[i]), crealf(AC[i]), cimagf(AC[i]));
        printf("%10f  %10f  %10f  %10f\n", crealf(B[i]), cimagf(B[i]), crealf(BC[i]), cimagf(BC[i]));
    }

    for(i=0; i<nc; i++)
      D0[i] = A[i]*B[i];
    two_way_conv(C, nr, D1);
    printf("Two-Way convolution:\n");
    printf("%12s  %12s  %12s  %12s\n", "D-r-DIR", "D-i-DIR", "D-r-2way", "D-i-2way");
    for(i=0; i<nc; i++) 
      printf("%12f  %12f  %12f  %12f\n", crealf(D0[i]), cimagf(D0[i]), crealf(D1[i]), cimagf(D1[i]));

    fftwf_destroy_plan(fp);
    fftwf_destroy_plan(cp);

    return 0;
}
