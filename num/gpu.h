/*
**	Optional GPU backend (CUDA via make cuda; stub otherwise)
*/
#ifndef GPU_H__
#define GPU_H__

#include "mat.h"
#include <stddef.h>

int	gpu_available(void);
const char	*gpu_backend_name(void);

void	*gpu_malloc(size_t nbytes);
void	gpu_free(void *p);
int	gpu_memcpy_h2d(void *dst, const void *src, size_t nbytes);
int	gpu_memcpy_d2h(void *dst, const void *src, size_t nbytes);
int	gpu_memcpy_d2d(void *dst, const void *src, size_t nbytes);
int	gpu_memset(void *dst, int value, size_t nbytes);

/*
** Try GPU mat_mul; returns 0 on success, -1 to fall back to CPU.
** Uses mat.d_data when both inputs/outputs are on device.
*/
int	gpu_mat_mul(mat *c, const mat *a, const mat *b);
/* dA = dC * B^T, dB = A^T * dC (NULL skips that output). */
int	gpu_mat_mul_backward(mat *da, mat *db, const mat *dc,
		const mat *a, const mat *b);
int	gpu_mat_mul_bt(mat *c, const mat *a, const mat *b);
int	gpu_mat_outer_add(mat *dw, const mat *a, const mat *b);

#endif
