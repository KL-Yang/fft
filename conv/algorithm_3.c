#include <emmintrin.h>
#include "common.h"

/**
 * Derived from alg2
 * */ 
void conv_alg3(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    //for(int j=0; j<m; j++) {
    for(int j=0; j<4; j++) {
        for(int k=0; k<m/4; k++) {
            for(int i=j+k*4; i<n; i++)
                b[i] += f[j+k*4]*a[i-j-k*4];
        }
    }
}

/**
 * @brief
 * @param xa:
 * @param a :
 * */
void stream0(__m128 *xa, const float *a, const float *f, int m, float *b)
{
    __m128 xx, xb;
    xa[1] = _mm_loadu_ps(a+4);
    xx = _mm_shuffle_ps(xa[0], xa[1], _MM_SHUFFLE(1,0,3,2));
    for(int i=0; i<m; i+=4) {
        //note, each move 2 i+=2, shuffle to get intermedia ones
        xb  = _mm_loadu_ps(b+i);
        fi  = _mm_load_ps1(f+i);
        xb += xa[0]*fi;
        _mm_storeu_ps(b+i, xb);
    }
    xa[0] = xa[1];
    //reload xa[1] outside!
}

void conv_alg3a(const float *a, int n, const float *f, int m, float *b)
{
    const float *pa = a;
    __m128 xa[6];

    for(int i=0; i<n; i=i+8) { //divided in n/m block!
        //first stream
        xa[0] = _mm_loadu_ps(pa+0);
        xa[4] = _mm_loadu_ps(pa+4);
        xa[2] = _mm_shuffle_ps(xa[0], xa[4], _MM_SHUFFLE(1,0,3,2));
        for(int j=0; j<m; j++) {
        }
        //loop through the f!

        //second stream
        a1 = _mm_loadu_ps(pa+1);
        a5 = _mm_loadu_ps(pa+5);
        a3 = _mm_shuffle_ps(a1, a5, _MM_SHUFFLE(1,0,3,2));
        //loop through the f!
    }
}

/**
 * This is more like m stream of input and output
 * */ /*
         void conv_alg3(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int j=0; j<m; j++) {
        int i; __m128 bi, fj, ax;
        for(i=j; i<n-4; i+=4) {
            bi = _mm_loadu_ps(b+i);
            fj = _mm_load_ps1(f+j);
            ax = _mm_loadu_ps(a+i-j);
            bi = _mm_add_ps(bi, _mm_mul_ps(ax, fj));
            _mm_storeu_ps(b+i, bi);
        }
        for(; i<n; i++)
            b[i] += f[j]*a[i-j];
    }
} */
