#ifndef OS
#define OS

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

void to_ply(const float *points, size_t num_points, const char *filename);

#ifdef __cplusplus
}
#endif

#endif

