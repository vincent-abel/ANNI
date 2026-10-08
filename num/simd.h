/*
**	SIMD-friendly kernels (auto-vectorization helpers; no intrinsics yet)
*/
#ifndef SIMD_H__
#define SIMD_H__

#ifdef __GNUC__
#define ANN_RESTRICT __restrict__
#else
#define ANN_RESTRICT
#endif

double	ann_dot(const double *ANN_RESTRICT a, const double *ANN_RESTRICT b, int n);
void	ann_axpy(double *ANN_RESTRICT y, double alpha,
		const double *ANN_RESTRICT x, int n);
void	ann_scale(double *ANN_RESTRICT v, double s, int n);

#endif
