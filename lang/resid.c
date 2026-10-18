/*
**	Residual stream helpers (pre-norm wiring)
*/
#include <string.h>
#include "gpu.h"
#include "resid.h"

int resid_add(mat *dst, const mat *src)
{
	int i;
	int n;

	if (!dst || !src)
		return -1;
	if (dst->rows != src->rows || dst->cols != src->cols)
		return -1;
	if (gpu_mat_add(dst, src) == 0)
		return 0;
	if (mat_ensure_host(dst) != 0 || mat_ensure_host((mat *)src) != 0)
		return -1;
	mat_host_dirty(dst);
	n = dst->rows * dst->cols;
	for (i = 0; i < n; i++)
		dst->data[i] += src->data[i];
	return 0;
}

int resid_sum(mat *dst, const mat *a, const mat *b)
{
	if (!dst || !a || !b)
		return -1;
	if (dst->rows != a->rows || dst->cols != a->cols
	    || b->rows != a->rows || b->cols != a->cols)
		return -1;
	if (gpu_mat_add2(dst, a, b) == 0)
		return 0;
	if (resid_copy(dst, a) != 0)
		return -1;
	return resid_add(dst, b);
}

int resid_copy(mat *dst, const mat *src)
{
	int n;

	if (!dst || !src)
		return -1;
	if (dst->rows != src->rows || dst->cols != src->cols)
		return -1;
	if (gpu_mat_copy(dst, src) == 0)
		return 0;
	if (mat_ensure_host((mat *)src) != 0 || mat_ensure_host(dst) != 0)
		return -1;
	n = dst->rows * dst->cols;
	memcpy(dst->data, src->data, (size_t)n * sizeof(ann_real));
	mat_host_dirty(dst);
	return 0;
}
