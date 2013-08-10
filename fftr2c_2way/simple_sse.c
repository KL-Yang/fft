#include <pmmintrin.h>
#include <complex.h>
#include <stdio.h>

inline static __m128 i_sse3_1_cmul(__m128 a, __m128 b)
{
    __m128 c;
    c = _mm_mul_ps(_mm_moveldup_ps(a), b);
    b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2,3,0,1));
    a = _mm_mul_ps(_mm_movehdup_ps(a), b);
    return _mm_addsub_ps(c, a);
}

int main()
{
    complex float a[2], b[2], c[2], x[2];

    __m128 xc, xa, xb;

    /**
     * Tesing of complex conjugate SSE code
     * */
    a[0] = 1+2*I;
    a[1] = 3+4*I;
    xc = _mm_loadu_ps((float*)a);
    xc = _mm_mul_ps(xc, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f));
    _mm_storeu_ps((float*)c, xc);
    printf("complex conjugate: \n"
           "  (%5.1f, %5.1f) -> (%5.1f, %5.1f)\n"
           "  (%5.1f, %5.1f) -> (%5.1f, %5.1f)\n", 
           crealf(a[0]), cimagf(a[0]), crealf(c[0]), cimagf(c[0]), 
           crealf(a[1]), cimagf(a[1]), crealf(c[1]), cimagf(c[1]));

    /**
     * swap {a,b,c,d} -> {c, d, a, b}
     * */
    a[0] = 1+2*I;
    a[1] = 3+4*I;
    xa = _mm_loadu_ps((float*)a);
    xa = _mm_shuffle_ps(xa, xa, _MM_SHUFFLE(1, 0, 3, 2));
    _mm_storeu_ps((float*)c, xa);
    printf("complex sequence swap: \n"
           "  {%5.1f, %5.1f, %5.1f, %5.1f} -> {%5.1f, %5.1f, %5.1f, %5.1f}\n", 
           crealf(a[0]), cimagf(a[0]), crealf(a[1]), cimagf(a[1]),
           crealf(c[0]), cimagf(c[0]), crealf(c[1]), cimagf(c[1]));

    /**
     * complex multiply by I/2
     * */
    xa = _mm_loadu_ps((float*)a);
    xa = _mm_shuffle_ps(xa, xa, _MM_SHUFFLE(2,3,0,1));
    xa = _mm_mul_ps(xa, _mm_set_ps(0.5f, -0.5f, 0.5f, -0.5f));
    _mm_storeu_ps((float*)c, xa);
    printf("complex multiply by I/2: \n"
           "  (%5.1f, %5.1f) -> (%5.1f, %5.1f)\n"
           "  (%5.1f, %5.1f) -> (%5.1f, %5.1f)\n", 
           crealf(a[0]), cimagf(a[0]), crealf(c[0]), cimagf(c[0]), 
           crealf(a[1]), cimagf(a[1]), crealf(c[1]), cimagf(c[1]));

    /**
     * complex multiply of two number
     * */
    a[0] = 1+2*I;
    a[1] = 3+4*I;
    b[0] = 5+6*I;
    b[1] = 7+8*I;
    xa = _mm_loadu_ps((float*)a);
    xb = _mm_loadu_ps((float*)b);
    xc = i_sse3_1_cmul(xa, xb);
    _mm_storeu_ps((float*)c, xc);
    x[0] = a[0]*b[0];
    x[1] = a[1]*b[1];
    printf("complex multiply of two number: \n");
    printf("  X87: (%5.1f, %5.1f)  (%5.1f, %5.1f)\n", crealf(x[0]), cimagf(x[0]), crealf(x[1]), cimagf(x[1]));
    printf("  SSE: (%5.1f, %5.1f)  (%5.1f, %5.1f)\n", crealf(c[0]), cimagf(c[0]), crealf(c[1]), cimagf(c[1]));

    return 0;
}
