/*
**	SIMD-friendly kernels (auto-vectorization helpers + optional intrinsics)
*/
#ifndef SIMD_H__
#define SIMD_H__

#include "real.h"

#ifdef __GNUC__
#define ANN_RESTRICT __restrict__
#else
#define ANN_RESTRICT
#endif

ann_real	ann_dot(const ann_real *ANN_RESTRICT a,
		const ann_real *ANN_RESTRICT b, int n);
void	ann_axpy(ann_real *ANN_RESTRICT y, ann_real alpha,
		const ann_real *ANN_RESTRICT x, int n);
void	ann_scale(ann_real *ANN_RESTRICT v, ann_real s, int n);
const char	*ann_simd_backend(void);

#endif
