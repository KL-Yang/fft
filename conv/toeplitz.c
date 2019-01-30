#include <stdio.h>
#include <xmmintrin.h>

/**
 * Toeplitz_mat4
 * f : in reverse order!
 * */
void Toeplitz_mat4(const float *a, const float *f, float *b)
{
    __m128 ax, fx, bx = _mm_setzero_ps();
    for(int x=0; x<4; x++) {
        ax = _mm_loadu_ps(a+3-x);
        fx = _mm_load_ps1(f+x);
        bx = bx+ax*fx;
    }
    _mm_storeu_ps(b, bx);
}

/**
 * TODO: add mat04/08/12/16
 * 16 floats (64byte) has reached the cache line size!
 * May be f is better 64byte cache line size aligned.
 *
 * Use python unittest+numpy+scipy to validate result
 * and use C function to measure speed!
 * */
void Toeplitz_mat04(const float *a, const float *f, float *b)
{
    __m128 ax, fx, bx = _mm_setzero_ps();
    ax  = _mm_loadu_ps(a+0);
    fx  = _mm_load_ps1(f+3);
    bx += ax*fx;
    ax  = _mm_loadu_ps(a+1);
    fx  = _mm_load_ps1(f+2);
    bx += ax*fx;
    ax  = _mm_loadu_ps(a+2);
    fx  = _mm_load_ps1(f+1);
    bx += ax*fx;
    ax  = _mm_loadu_ps(a+3);
    fx  = _mm_load_ps1(f+0);
    bx += ax*fx;
    _mm_storeu_ps(b, bx);
}

void Toeplitz_mat8(const float *a, const float *f, float *b)
{
    __m128 b1 = _mm_setzero_ps();
    __m128 b2 = _mm_setzero_ps();

    //8th colume, 8-7-group
    __m128 a0 = _mm_loadu_ps(a+0);
    __m128 a4 = _mm_loadu_ps(a+4);
    __m128 f7 = _mm_load_ps1(f+7);
    b1 += a0*f7;
    b2 += a4*f7;

    //7th colume
    __m128 a1 = _mm_loadu_ps(a+1);
    __m128 a5 = _mm_loadu_ps(a+5);
    __m128 f6 = _mm_load_ps1(f+6);
    b1 += a1*f6;
    b2 += a5*f6;

    //4th colume, 4-3-group
    __m128 a8 = _mm_loadu_ps(a+8);
    __m128 f3 = _mm_load_ps1(f+3);
    b1 += a4*f3;
    b2 += a8*f3;
 
    //3th colume
    __m128 a9 = _mm_loadu_ps(a+9);
    __m128 f2 = _mm_load_ps1(f+2);
    b1 += a5*f2;
    b2 += a9*f2;

    //6th colume, 6-5-group
    __m128 a2 = _mm_shuffle_ps(a0, a4, _MM_SHUFFLE(1,0,3,2));
    __m128 a6 = _mm_shuffle_ps(a4, a8, _MM_SHUFFLE(1,0,3,2));
    __m128 f5 = _mm_load_ps1(f+5);
    b1 += a2*f5;
    b2 += a6*f5;

    //5th colume
    __m128 a3 = _mm_shuffle_ps(a1, a5, _MM_SHUFFLE(1,0,3,2));
    __m128 a7 = _mm_shuffle_ps(a5, a9, _MM_SHUFFLE(1,0,3,2));
    __m128 f4 = _mm_load_ps1(f+4);
    b1 += a3*f4;
    b2 += a7*f4;

    //2th colume, 2-1-group
    __m128 aa = _mm_loadu_ps(a+10);
    __m128 f1 = _mm_load_ps1(f+1);
    b1 += a6*f1;
    b2 += aa*f1;

    //1st colume
    __m128 ab = _mm_loadu_ps(a+11);
    __m128 f0 = _mm_load_ps1(f+0);
    b1 += a7*f0;
    b2 += ab*f0;

    //output
    _mm_storeu_ps(b+0, b1);
    _mm_storeu_ps(b+4, b2);
}
/*
void Toeplitz_matx(const float *a, const float *f, int m, float *b)
{
    for(int i=0; i<n; i=i+m) {
        for(int j=0; j<m; j+=4) {
            a0 = _mm_loadu_ps(a+0);
        }
    }
} */
/*
//12x12 matrix
void Toeplitz_matb(const float *a, const float *f, float *b)
{
    __m128 bx[m]; //blocksize m*4
    for(int i=0; i<m; i++)
        bx[i] = _mm_loadu_ps(b+4*i);

    //last-0 [even/odd] colume
    for(int i=0; i<m; i++) {
        ax0[i] = _mm_loadu_ps(a+4*i);
        fx0[i] = _mm_load_ps1(f+4*i);
        bx[i] += ax0[i]*fx0[i];
        ax1[i] = _mm_loadu_ps(a+4*i+1);
        fx1[i] = _mm_load_ps1(f+4*i-1);
        bx[i] += ax1[i]*fx1[i];
    }
    //last-4 [even/odd] colume
    for()
    {
    }
    //last-2 [even/odd] colume
    for()
    {
    }

    

    _mm_storeu_ps(b+0, b1);
    _mm_storeu_ps(b+4, b2);
    _mm_storeu_ps(b+8, b3);
}
*/
//16x16 matrix


//

/**
 * a : length of 2*m-1
 * */
/*
void Toeplitz_matv(const float *a, const float *f, int m, float *b)
{
    __m128 ax, fx, bx;
    for(int y=0; y<m; y+=4) {
        bx = _mm_loadu_ps(b+y);
        for(int x=0; x<m; x++) {
            ax = _mm_loadu_ps(a+m-1-x+y); //this one repeats
            fx = _mm_load_ps1(f+x);
            bx = bx+ax*fx;
        }
        _mm_storeu_ps(b+y, bx);
    }
} */

/**
 import numpy
 from scipy.linalg import toeplitz
#test1
 a = toeplitz([4,5,6,7], [4,3,2,1])
 f = numpy.array([1,2,3,4]).reshape((4,1))
 b = numpy.dot(a,f)
 print(a)
 print(f)
 print(b)
#test2
 * */
void test_4(void)
{
    float a[] = {1,2,3,4,5,6,7};
    float f[] = {1,2,3,4};  //in reverse order!
    float b[4];
    Toeplitz_mat4(a, f, b);
    for(int i=0; i<4; i++) 
        printf("%2d  %f\n", i, b[i]);
}

void test_n(int n)
{
    float a[2*n-1], f[n], b[n];
    for(int i=0; i<2*n-1; i++)
        a[i] = i+1;
    for(int i=0; i<n; i++) {
        f[i] = i+1;
        b[i] = 0;
    }
    //Toeplitz_matv(a, f, n, b);
    Toeplitz_mat8(a, f, b);
    for(int i=0; i<n; i++) 
        printf("%4d  %f\n", i, b[i]);
}
/*
int main()
{
    test_n(8);
    return 0;
} */
