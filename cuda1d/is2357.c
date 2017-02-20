#include <stdio.h>
#include <stdlib.h>

int isfftsize(int size, int *base, int n)
{
    int i;
    while(size!=1) {
        for(i=0; i<n; i++)
          if(size%base[i]==0) {
              size = size/base[i];
              break;
          }
        if(i==n)
          return 0;
    };
    return 1;
}

/**
 * is2357 min max
 *   fftsize : logical length
 * If it is 2357 size, return 0, otherwise return 1.
 * */
int main(int argc, char * argv[])
{
    if(argc<3) {
        printf("Usage: %s nr_min nr_max\n", argv[0]);
        return 1;
    }
    int base[] = {2,3,5,7};
    int nr_min = atoi(argv[1]);
    int nr_max = atoi(argv[2]);
    for(int i=nr_min; i<=nr_max; i++) 
        if(isfftsize(i, base, 4)) 
            printf("%d\n", i);
    return 0;
}
