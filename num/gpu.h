/*
**	Optional GPU mat_mul backend (CUDA when ANN_CUDA is defined)
*/
#ifndef GPU_H__
#define GPU_H__

#include "mat.h"

/* 1 if a GPU backend is linked; 0 for CPU-only builds. */
int	gpu_available(void);
const char	*gpu_backend_name(void);
/*
** Try GPU mat_mul; returns 0 on success, -1 to fall back to CPU mat_mul.
** CPU builds always return -1.
*/
int	gpu_mat_mul(mat *c, const mat *a, const mat *b);

#endif
