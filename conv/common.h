#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void conv_opsse(const float * restrict a, int n, const float * restrict f, int m, float * restrict b);
void conv_alg2(const float * restrict a, int n, const float * restrict f, int m, float * restrict b);
void conv_alg3(const float * restrict a, int n, const float * restrict f, int m, float * restrict b);
void conv_opsse2(const float * restrict a, int n, const float * restrict f, int m, float * restrict b);
