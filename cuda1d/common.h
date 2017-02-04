#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ID_LEN  16
typedef struct cputimer_struct * cputimer_h;

void cputimer_init(cputimer_h *h, const char *id);
void cputimer_start(cputimer_h h);
void cputimer_pause(cputimer_h h);
void cputimer_done(cputimer_h h);

#ifdef __cplusplus
}
#endif
