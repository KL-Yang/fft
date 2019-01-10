#include <emmintrin.h>
#include "common.h"

/**
 * @brief Intresting method, just re-arrange the filters seems speed up a lot
 * even faster than the manual coded SSE on AMD K-8 (Athlon II X4 630)
 * */
void conv_alg2(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int j=0; j<m; j+=4) {
        //for(int i=j+0; i<n; i++)  //split!
        for(int i=j+0; i<j+3; i++)
            b[i] += f[j+0]*a[i-j-0];

        //for(int i=j+1; i<n; i++)  //split!
        for(int i=j+1; i<j+3; i++)
            b[i] += f[j+1]*a[i-j-1];

        //for(int i=j+2; i<n; i++)  //split!
        for(int i=j+2; i<j+3; i++)
            b[i] += f[j+2]*a[i-j-2];

        for(int i=j+3; i<n; i++) {
            b[i] += f[j+0]*a[i-j-0];
            b[i] += f[j+1]*a[i-j-1];
            b[i] += f[j+2]*a[i-j-2];
            b[i] += f[j+3]*a[i-j-3];
        }
    }
}

void conv_opsse(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{   //TODO: remove this partially wrong function code, and the next one!
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
            for(int i=0*4; i<2*4; i+=4) {   //this is not correct!
                //valgrind will not pass!!!!
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


