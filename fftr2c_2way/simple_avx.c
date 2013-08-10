#include <immintrin.h>
#include <complex.h>
#include <stdio.h>
/*
inline static __m128 i_sse3_1_cmul(__m128 a, __m128 b)
{
    __m128 c;
    c = _mm_mul_ps(_mm_moveldup_ps(a), b);
    b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2,3,0,1));
    a = _mm_mul_ps(_mm_movehdup_ps(a), b);
    return _mm_addsub_ps(c, a);
}
*/
inline static __m256 avx_cmul_i(__m256 a, __m256 b)
{
    __m256 c;
    c = _mm256_mul_ps(_mm256_moveldup_ps(a), b);
    b = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2,3,0,1));
    a = _mm256_mul_ps(_mm256_movehdup_ps(a), b);
    return _mm256_addsub_ps(c, a);
}

int main()
{
    complex float a[4], b[4], c[4], x[4];

    __m256 xc, xa, xb;

    /**
     * Tesing of complex conjugate SSE code
     * */
    a[0] = 1+2*I; a[1] = 3+4*I;
    a[2] = 5+6*I; a[3] = 7+8*I;
    xc = _mm256_loadu_ps((float*)a);
    xc = _mm256_mul_ps(xc, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f));
    _mm256_storeu_ps((float*)c, xc);
    printf("complex conjugate: \n"
           "  a[0] = (%5.1f, %5.1f) -> c[0] = (%5.1f, %5.1f)\n"
           "  a[1] = (%5.1f, %5.1f) -> c[1] = (%5.1f, %5.1f)\n"
           "  a[2] = (%5.1f, %5.1f) -> c[2] = (%5.1f, %5.1f)\n"
           "  a[3] = (%5.1f, %5.1f) -> c[3] = (%5.1f, %5.1f)\n", 
           crealf(a[0]), cimagf(a[0]), crealf(c[0]), cimagf(c[0]), 
           crealf(a[1]), cimagf(a[1]), crealf(c[1]), cimagf(c[1]),
           crealf(a[2]), cimagf(a[2]), crealf(c[2]), cimagf(c[2]), 
           crealf(a[3]), cimagf(a[3]), crealf(c[3]), cimagf(c[3]));
    /**
     * swap {a,b,c,d} -> {c, d, a, b}
     * */
    a[0] = 1+2*I; a[1] = 3+4*I;
    a[2] = 5+6*I; a[3] = 7+8*I;
    xa = _mm256_loadu2_m128((float*)a, (float*)a+4);
    xa = _mm256_permute_ps(xa, _MM_SHUFFLE(1, 0, 3, 2));
    _mm256_storeu_ps((float*)c, xa);
    printf("complex sequence swap: \n"
           "  OLD = {%5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f}\n"
           "  NEW = {%5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f}\n", 
           crealf(a[0]), cimagf(a[0]), crealf(a[1]), cimagf(a[1]),
           crealf(a[2]), cimagf(a[2]), crealf(a[3]), cimagf(a[3]),
           crealf(c[0]), cimagf(c[0]), crealf(c[1]), cimagf(c[1]),
           crealf(c[2]), cimagf(c[2]), crealf(c[3]), cimagf(c[3]));

    /**
     * complex multiply by I/2
     * */
    xa = _mm256_loadu_ps((float*)a);
    xa = _mm256_permute_ps(xa, _MM_SHUFFLE(2,3,0,1));
    xa = _mm256_mul_ps(xa, _mm256_set_ps(0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f));
    _mm256_storeu_ps((float*)c, xa);
    printf("complex multiply by I/2: \n"
           "  a[0] = (%5.1f, %5.1f) -> c[0] = (%5.1f, %5.1f)\n"
           "  a[1] = (%5.1f, %5.1f) -> c[1] = (%5.1f, %5.1f)\n"
           "  a[2] = (%5.1f, %5.1f) -> c[2] = (%5.1f, %5.1f)\n"
           "  a[3] = (%5.1f, %5.1f) -> c[3] = (%5.1f, %5.1f)\n", 
           crealf(a[0]), cimagf(a[0]), crealf(c[0]), cimagf(c[0]), 
           crealf(a[1]), cimagf(a[1]), crealf(c[1]), cimagf(c[1]),
           crealf(a[2]), cimagf(a[2]), crealf(c[2]), cimagf(c[2]), 
           crealf(a[3]), cimagf(a[3]), crealf(c[3]), cimagf(c[3]));

    /**
     * complex multiply of two number
     * */
    a[0] = 1+2*I; a[1] = 3+4*I;
    a[2] = 5+6*I; a[3] = 7+8*I;
    b[0] = 2+3*I, b[1] = 4+5*I;
    b[2] = 6+7*I, b[3] = 8+9*I;
    xa = _mm256_loadu_ps((float*)a);
    xb = _mm256_loadu_ps((float*)b);
    xc = avx_cmul_i(xa, xb);
    _mm256_storeu_ps((float*)c, xc);
    x[0] = a[0]*b[0];
    x[1] = a[1]*b[1];
    x[2] = a[2]*b[2];
    x[3] = a[3]*b[3];
    printf("complex multiply of two number: \n");
    printf("  X87: (%5.1f, %5.1f)  (%5.1f, %5.1f)  (%5.1f, %5.1f)  (%5.1f, %5.1f)\n", 
           crealf(x[0]), cimagf(x[0]), crealf(x[1]), cimagf(x[1]),
           crealf(x[2]), cimagf(x[2]), crealf(x[3]), cimagf(x[3]));
    printf("  AVX: (%5.1f, %5.1f)  (%5.1f, %5.1f)  (%5.1f, %5.1f)  (%5.1f, %5.1f)\n", 
           crealf(c[0]), cimagf(c[0]), crealf(c[1]), cimagf(c[1]),
           crealf(c[2]), cimagf(c[2]), crealf(c[3]), cimagf(c[3]));

    return 0;
}
