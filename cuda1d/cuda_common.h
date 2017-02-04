#include "common.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct gputimer_struct * gputimer_h;

void gputimer_init(gputimer_h *h, cudaStream_t stream, const char *id);
void gputimer_start(gputimer_h h);
void gputimer_pause(gputimer_h h);
void gputimer_done(gputimer_h h);

#ifdef __cplusplus
}
#endif
