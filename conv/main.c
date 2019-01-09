#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emmintrin.h>

void conv_basic(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int i=0; i<n; i++)
        for(int j=0; j<m && j<=i; j++)
            b[i] += f[j]*a[i-j];
}

void conv_decimate(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int i=0; i<n; i+=2)
        for(int j=0; j<m && j<=i; j++)
            b[i/2] += f[j]*a[i-j];
}

void data_gen1(float *a, int n, float *f, int m)
{
    for(int i=0; i<n; i++)
        a[i] = i+1;
    for(int i=0; i<m; i++)
        f[i] = 2*(i+1);
}

void conv_optimize(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int j=0; j<m; j++)
        for(int i=j; i<n; i++)
            b[i] += f[j]*a[i-j];
}

void conv_opsse(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    //j=0,4,8
    //j=0 -> for(int i=0; i<n; i++)
    //  b[i] += f[0]*a[i]
    //j=4 -> for(int i=4; i<n; i++)
    //  b[i] += f[4]*a[i-4];
    //j=8 -> for(int i=8; i<n; i++)
    //  b[i] += f[8]*a[i-8];
    //
    //j=1 -> for(int i=1; i<n; i++)
    //  b[i] += f[1]*a[i-1];
    //
    //note must pre-know m, here we know m is 12!
    __m128 fi[3], ai, bi;
    for(int j=0; j<4; j++) {
        for(int k=0; k<3 /*m/4*/; k++) {
            fi[k] = _mm_load_ps1(f+4*k+j);
            for(int i=k*4; i<n; i+=4) {
                bi = _mm_loadu_ps(b+i);
                ai = _mm_loadu_ps(a+i-4*k-j);
                bi = _mm_add_ps(bi, _mm_mul_ps(ai, fi[k]));
                _mm_storeu_ps(b+i, bi);
            }
        }
    }
}

void conv_opsse2(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    __m128 fi[3], ai, bi; //close to GCC compiler implementation!
    for(int j=0; j<4; j++) {
            fi[0] = _mm_load_ps1(f+4*0+j);
            fi[1] = _mm_load_ps1(f+4*1+j);
            fi[2] = _mm_load_ps1(f+4*2+j);
            for(int i=0*4; i<2*4; i+=4) {
                bi = _mm_loadu_ps(b+i);
                ai = _mm_loadu_ps(a+i-4*0-j);
                bi = _mm_add_ps(bi, _mm_mul_ps(ai, fi[0]));
                _mm_storeu_ps(b+i, bi);
            }
            for(int i=1*4; i<2*4; i+=4) {
                bi = _mm_loadu_ps(b+i);
                ai = _mm_loadu_ps(a+i-4*1-j);
                bi = _mm_add_ps(bi, _mm_mul_ps(ai, fi[1]));
                _mm_storeu_ps(b+i, bi);
            }
            for(int i=2*4; i<n; i+=4) {
                bi = _mm_loadu_ps(b+i);
                ai = _mm_loadu_ps(a+i-4*0-j);
                bi = _mm_add_ps(bi, _mm_mul_ps(ai, fi[0]));
                ai = _mm_loadu_ps(a+i-4*1-j);
                bi = _mm_add_ps(bi, _mm_mul_ps(ai, fi[1]));
                ai = _mm_loadu_ps(a+i-4*2-j);
                bi = _mm_add_ps(bi, _mm_mul_ps(ai, fi[2]));
                _mm_storeu_ps(b+i, bi);
            }
    }
}

void conv_opdeci(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int j=0; j<m; j++)
        for(int i=j-j%2; i<n; i+=2)
            b[i/2] += f[j]*a[i-j];
}


int main()
{
    int repeat=100000;
    //int repeat=1;
    int n=4096, m=12;
    float a[n], b[n], c[n], d[n], f[m];

    data_gen1(a, n, f, m);
    memset(b, 0, n*sizeof(float));
    memset(c, 0, n*sizeof(float));

    for(int i=0; i<repeat; i++) {
        if(i%1000==0)
            printf("run@%d\n", i);
        //conv_basic(a, n, f, m, b);
        //conv_decimate(a, n, f, m, b);
        //conv_optimize(a, n, f, m, c);
        conv_opsse2(a, n, f, m, c);
        //conv_opdeci(a, n, f, m, c);
    }
    for(int i=0; i<n; i++) {
        if(fabs(b[i]-c[i])>0.1)
            printf("%4d b=%e c=%e\n", i, b[i], c[i]);
    }

    return 0;
}
