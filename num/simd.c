/*
**	SIMD-friendly contiguous kernels for auto-vectorization
*/
#include "simd.h"

double ann_dot(const double *ANN_RESTRICT a, const double *ANN_RESTRICT b, int n)
{
	int i;
	double sum;

	sum = 0.0;
	if (!a || !b || n <= 0)
		return 0.0;
	for (i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

void ann_axpy(double *ANN_RESTRICT y, double alpha,
	const double *ANN_RESTRICT x, int n)
{
	int i;

	if (!y || !x || n <= 0)
		return;
	for (i = 0; i < n; i++)
		y[i] += alpha * x[i];
}

void ann_scale(double *ANN_RESTRICT v, double s, int n)
{
	int i;

	if (!v || n <= 0)
		return;
	for (i = 0; i < n; i++)
		v[i] *= s;
}
