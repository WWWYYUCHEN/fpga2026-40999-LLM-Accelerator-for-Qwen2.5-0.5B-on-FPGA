#include <stdint.h>
typedef uint64_t XTime;
#define COUNTS_PER_SECOND 1000000ULL
void XTime_GetTime(XTime *);
