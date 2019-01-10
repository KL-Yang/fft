#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <emmintrin.h>
#include "common.h"

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

/**
 * Generate random data in range [-10, 10]
 * */
void data_gen_rand(float *a, int n)
{
    for(int i=0; i<n; i++)
        a[i] = 10.0f*(0.5f-random()*1.0f/RAND_MAX);
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

void conv_opdeci(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int j=0; j<m; j++)
        for(int i=j-j%2; i<n; i+=2)
            b[i/2] += f[j]*a[i-j];
}

void (*algorithm_func[])(const float*, int, const float*, int, float*)
    = { &conv_basic, 
        &conv_optimize
    };

int fir_valid(int n, int m, int algorithm_id, int verbose)
{
    float *a, *f, *b, *c;
    a = calloc(n, sizeof(float));
    f = calloc(m, sizeof(float));
    b = calloc(n, sizeof(float));
    c = calloc(n, sizeof(float));
    data_gen_rand(a, n);
    data_gen_rand(f, m);
    conv_basic(a, n, f, m, b);
    algorithm_func[algorithm_id](a, n, f, m, c);

    int bad_points=0;
    for(int i=0; i<n; i++)
        if(fabs(b[i]-c[i])>0.01f) {
            bad_points++;
            if(verbose)
                printf("%4d b=%e c=%e\n", i, b[i], c[i]);
        }
    free(a);
    free(f);
    free(b);
    free(c);
    return bad_points;
}

void fir_test(int n, int m, int repeat, int algorithm_id, int check, int verbose)
{
    float *a, *f, *b;
    a = calloc(n, sizeof(float));
    f = calloc(m, sizeof(float));
    b = calloc(n, sizeof(float));

    clock_t start = clock();
    for(int i=0; i<repeat; i++) {
        algorithm_func[algorithm_id](a, n, f, m, b);
        asm("");
    }
    clock_t current=clock();
    free(a);
    free(f);
    free(b);

    float tcost = (current-start)*1.0f/CLOCKS_PER_SEC;
    printf("%2d %8d %4d %8d %9.4f\n", algorithm_id, n, m, repeat, tcost);
}
/**
 *  -a algrithm id
 *  -m 12
 *  -n 4096
 *  -r 100000
 *  -c check correctness
 *  -v vorbose
 * */
int main(int argc, char **argv)
{
    char ch; int debug=0;
    int m=0, n=0, repeat=0, check=0, verbose=0, algorithm_id=0;
    while((ch=getopt(argc, argv, "a:m:n:r:cv"))!=-1)
        switch(ch) {
            case 'a':
                if(debug) { printf("-a %s\n", optarg);}
                algorithm_id = atoi(optarg);
                break;
            case 'm':
                if(debug) { printf("-m %s\n", optarg);}
                m = atoi(optarg);
                break;
            case 'n':
                if(debug) { printf("-n %s\n", optarg);}
                n = atoi(optarg);
                break;
            case 'r':
                printf("-r %s\n", optarg);
                repeat = atoi(optarg);
                break;
            case 'c':
                printf("-c\n");
                check = 1;
                break;
            case 'v':
                printf("-v\n");
                verbose = 1;
                break;
            case '?':
                printf("Usage: %s -n -m -r -a [-c -v]\n"
                        "\t -n : input series length\n"
                        "\t -m : short filter length\n"
                        "\t -r : repeat times\n"
                        "\t -a : algorithm id\n"
                        "\t -c : check algorithm against basic one\n"
                        "\t -v : verbose print\n", argv[0]);
                break;
            default:
                abort();
        }

    if(fir_valid(n,m,algorithm_id, verbose)) {
        printf("fir_valid!\n");
        if(verbose)
            printf("Failed algorithm=%d, result is wrong!\n", algorithm_id);
        exit(1);
    }
    if(verbose)
        printf("algorithm_id=%d successful!\n", algorithm_id);

    fir_test(n, m, repeat, algorithm_id, check, verbose);
    return 0;
}
