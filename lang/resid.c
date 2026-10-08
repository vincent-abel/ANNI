/*
**	Residual stream helpers (pre-norm wiring)
*/
#include <string.h>
#include "resid.h"

int resid_add(mat *dst, const mat *src)
{
	int i;
	int n;

	if (!dst || !src || !dst->data || !src->data)
		return -1;
	if (dst->rows != src->rows || dst->cols != src->cols)
		return -1;
	n = dst->rows * dst->cols;
	for (i = 0; i < n; i++)
		dst->data[i] += src->data[i];
	return 0;
}

int resid_copy(mat *dst, const mat *src)
{
	int n;

	if (!dst || !src || !dst->data || !src->data)
		return -1;
	if (dst->rows != src->rows || dst->cols != src->cols)
		return -1;
	n = dst->rows * dst->cols;
	memcpy(dst->data, src->data, (size_t)n * sizeof(double));
	return 0;
}
