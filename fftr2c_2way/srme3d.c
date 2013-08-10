#include <math.h>
#include <fftw3.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pmmintrin.h>
#include <immintrin.h>

#define SSE_MM_LOAD_PS     _mm_loadu_ps
#define SSE_MM_STORE_PS    _mm_storeu_ps

inline static __m128 cmul_sse3_i(__m128 a, __m128 b)
{
    __m128 c;
     c = _mm_moveldup_ps(a);
     a = _mm_movehdup_ps(a);
     c = _mm_mul_ps(c, b);
     b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2,3,0,1));
     a = _mm_mul_ps(a, b);
    return _mm_addsub_ps(c, a);
}
/*
inline static __m256 avx_cmul_i(__m256 a, __m256 b)
{
    __m256 c;
    c = _mm256_mul_ps(_mm256_moveldup_ps(a), b);
    b = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2,3,0,1));
    a = _mm256_mul_ps(_mm256_movehdup_ps(a), b);
    return _mm256_addsub_ps(c, a);
}*/

inline static void sse3_cmul(complex float *a, complex float *b, 
                             complex float *c, int num)
{
    int i, n;
    __m128 xmm0, xmm1, xmm2, xmmc;
    n = num-num%2;

    for(i=0; i<n; i=i+2) {
        xmm0 = SSE_MM_LOAD_PS((float*)(a+i));
        xmm1 = SSE_MM_LOAD_PS((float*)(b+i));
        xmm2 = cmul_sse3_i(xmm0, xmm1);
        xmmc = SSE_MM_LOAD_PS((float*)(c+i));
        SSE_MM_STORE_PS((float*)(c+i), _mm_add_ps(xmmc, xmm2));
    }
    if(num%2)
      c[n] += a[n]*b[n];
}
/*
inline static void avx_cmul(complex float *a, complex float *b, 
                            complex float *c, int num)
{
    int i, n;
    __m256 xmm0, xmm1, xmm2, xmmc;
    n = num-num%4;

    for(i=0; i<n; i=i+4) {
        xmm0 = _mm256_loadu_ps((float*)(a+i));
        xmm1 = _mm256_loadu_ps((float*)(b+i));
        xmm2 = avx_cmul_i(xmm0, xmm1);
        xmmc = _mm256_loadu_ps((float*)(c+i));
        _mm256_storeu_ps((float*)(c+i), _mm256_add_ps(xmmc, xmm2));
    }
    for(i=n; i<num; i++)
      c[i] += a[i]*b[i];
}
*/

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
    int x;
    x = (int)nearbyint((max-1)*((rand()*1.0)/(RAND_MAX*1.0)));
    return x;
}

/**
 * output the convolve result in frequency domain
 * */
void two_way_conv(const complex float *C, int nr, complex float *D)
{
    int nc = nr/2+1;
    int nn = (nc%2)?(nc):(nc-1);
    D[0] += ((float*)C)[0]*((float*)C)[1]; //DC component

    for(int i=1; i<nn; i=i+2) {
        __m128 ci = _mm_loadu_ps((float*)(C+i));
        __m128 cj = _mm_mul_ps(_mm_loadu_ps((float*)(C+nr-1-i)), 
               /*conjugate*/   _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f)); 
        cj = _mm_shuffle_ps(cj, cj, _MM_SHUFFLE(1,0,3,2));
        __m128 pl = _mm_add_ps(cj, ci);
        __m128 mi = _mm_sub_ps(cj, ci);
        __m128 xx = cmul_sse3_i(pl, mi); //multiple (a+b)(a-b)
        xx = _mm_shuffle_ps(xx, xx, _MM_SHUFFLE(2,3,0,1));
        xx = _mm_mul_ps(xx, _mm_set_ps(0.25f, -0.25f, 0.25f, -0.25f));
        xx = _mm_add_ps(xx, _mm_loadu_ps((float*)(D+i)));   //+=
        _mm_storeu_ps((float*)(D+i), xx); //multiple by I/4
    }
    if(nn!=nc) {
        complex float A =   (conjf(C[nr-nn])+C[nn])/2.0f;
        complex float B = I*(conjf(C[nr-nn])-C[nn])/2.0f;
        D[nn] += A*B;
    }
}


int main(int argc, char *argv[])
{

    fftwf_plan plan[2], pm;
    int i, j, k, nr, nc, nx;
    float *data, *vbuf, *temp, *work, *ctmp, *ftmp;
    float complex *freq, *snow, *rosy;

    if(argc!=5) {
        printf("If you want benchmark the speed, usage:\n"
               "%s length ntransfer repeat_times sse_align?\n"
               "sse_align=1 or sse_align=0\n", argv[0]);
        return 0;
    }
    printf("\nstart testing speed....\n");

    int repeat, ntrans, length;
    length = atoi(argv[1]);
    ntrans = atoi(argv[2]);
    repeat = atoi(argv[3]);
    nr = length;
    nc = nr/2+1;
    nx = nr;
    if(nr%4!=0)
      nx = (nr+4)-(nr+4)%4;
    printf("nr=%d, nc=%d step=%d\n", nr, nc, nx);

    data = calloc(nx*ntrans, sizeof(float));
    vbuf = calloc(nx, sizeof(float));
    temp = calloc(nx, sizeof(float));
    work = calloc(nx, 2*sizeof(float));
    freq = calloc(nc, sizeof(fftwf_complex));
    snow = calloc(nc, sizeof(fftwf_complex));
    rosy = calloc(nc, sizeof(fftwf_complex));
    ctmp = calloc(nr, sizeof(fftwf_complex));
    ftmp = calloc(nr, sizeof(fftwf_complex));

    for(i=0; i<nx; i++)
      vbuf[i] = 1500.0f+1000.0f*i/nx;

    plan[0] = fftwf_plan_dft_r2c_1d(nr, (float*)data, (fftwf_complex *)freq, FFTW_PATIENT);
    plan[1] = fftwf_plan_dft_1d(nr, (fftwf_complex*)ctmp, (fftwf_complex*)ftmp, FFTW_FORWARD, FFTW_PATIENT);

    int a, b, off1, off2;
    struct rusage u0, u1, u2;

    memset(data, 0, ntrans*nx*sizeof(float));
    getrusage(RUSAGE_SELF, &u0);
    for(i=0; i<repeat; i++) {
        memset(freq, 0, nc*sizeof(complex float));
        for(j=0; j<ntrans; j++) {
            a = rand_range(ntrans); b = rand_range(ntrans);
            fftwf_execute_dft_r2c(plan[0], data+a*nx, (fftwf_complex*)snow);
            fftwf_execute_dft_r2c(plan[0], data+b*nx, (fftwf_complex*)rosy);
            sse3_cmul(snow, rosy, freq, nc);
            //avx_cmul(snow, rosy, freq, nc);
        }
    }
    getrusage(RUSAGE_SELF, &u1);
    for(i=0; i<repeat; i++) {
        memset(freq, 0, nc*sizeof(complex float));
        for(j=0; j<ntrans; j++) {
            a = rand_range(ntrans); b = rand_range(ntrans);
            for(k=0; k<nr; k++) {
                ctmp[2*k+0] = data[a*nx+k];
                ctmp[2*k+1] = data[b*nx+k];
            } 
            fftwf_execute_dft(plan[1], (fftwf_complex*)ctmp, (fftwf_complex*)ftmp);
            two_way_conv((complex float*)ftmp, nr, (complex float*)freq);
        }
    }
    getrusage(RUSAGE_SELF, &u2);

    float time_fftw, time_nfft;
    time_fftw = get_ms_rusage(&u0, &u1);
    time_nfft = get_ms_rusage(&u1, &u2);

    printf("FFT of %dx%d repeat %d times:\n", nr, ntrans, repeat);
    printf("  fftw : %f ms\n", time_fftw);
    printf("  newf : %f ms\n", time_nfft);

    free(freq);
    free(data);
    free(snow);
    free(rosy);
    free(work);
    free(temp);
    fftwf_destroy_plan(plan[0]);
    fftwf_destroy_plan(plan[1]);

    return 0;
}
