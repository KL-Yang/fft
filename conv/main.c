#include <getopt.h>
#include "common.h"

void conv_basic(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int i=0; i<n; i++)
        for(int j=0; j<m && j<=i; j++)
            b[i] += f[j]*a[i-j];
}

void conv_optimize(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int j=0; j<m; j++)
        for(int i=j; i<n; i++)
            b[i] += f[j]*a[i-j];
}

void conv_decimate(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int i=0; i<n; i+=2)
        for(int j=0; j<m && j<=i; j++)
            b[i/2] += f[j]*a[i-j];
}

/**
 * Generate random data in range [-10, 10]
 * */
void data_gen_rand(float *a, int n)
{
    for(int i=0; i<n; i++)
        a[i] = 10.0f*(0.5f-random()*1.0f/RAND_MAX);
}

void conv_opdeci(const float * restrict a, int n, const float * restrict f, int m, float * restrict b)
{
    for(int j=0; j<m; j++)
        for(int i=j-j%2; i<n; i+=2)
            b[i/2] += f[j]*a[i-j];
}

void (*algorithm_func[])(const float*, int, const float*, int, float*)
    = { &conv_basic, 
        &conv_optimize,
        &conv_opsse
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

void fir_test(int n, int m, int repeat, int algorithm_id, int verbose)
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
    if(verbose)
        printf("ID %8s %4s %8s %9s\n", "n", "m", "repeat", "time(s)");
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
                if(debug) { printf("-r %s\n", optarg);}
                repeat = atoi(optarg);
                break;
            case 'c':
                if(debug) { printf("-c\n");}
                check = 1;
                break;
            case 'v':
                if(debug) { printf("-v\n");}
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

    if(check && fir_valid(n,m,algorithm_id, verbose)) {
        printf("fir_valid!\n");
        if(verbose)
            printf("Failed algorithm=%d, result is wrong!\n", algorithm_id);
        exit(1);
    }
    fir_test(n, m, repeat, algorithm_id, verbose);
    return 0;
}
