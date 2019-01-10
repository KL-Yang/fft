#include <emmintrin.h>
#include "common.h"

/**
 * Derived from alg2
 * */
void conv_alg3(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int j=0; j<m; j+=4) {
        for(int i=j+0; i<j+3; i++)
            b[i] += f[j+0]*a[i-j-0];
        for(int i=j+1; i<j+3; i++)
            b[i] += f[j+1]*a[i-j-1];
        for(int i=j+2; i<j+3; i++)
            b[i] += f[j+2]*a[i-j-2];
    }
    for(int j=0; j<m; j+=4) { //its' very trick to swap this loop
        for(int x=3; x<n-m; x++) { //i=x+j; replace variable!
            b[x+j] += f[j+0]*a[x-0];    //note, this is still horizontal sum
            b[x+j] += f[j+1]*a[x-1];    //unroll x might bring some benefits
            b[x+j] += f[j+2]*a[x-2];    //x=x+4 steps assert(n-m-3)%4==0
            b[x+j] += f[j+3]*a[x-3];
        }
    }
    for(int j=0; j<m; j+=4) {   //its' very trick to swap this loop
        for(int x=n-m; x<n-j; x++) { //i=x+j; replace variable!
            b[x+j] += f[j+0]*a[x-0];
            b[x+j] += f[j+1]*a[x-1];
            b[x+j] += f[j+2]*a[x-2];
            b[x+j] += f[j+3]*a[x-3];
        }
    }
}


