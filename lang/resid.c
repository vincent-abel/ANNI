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

	if (!dst || !src || !dst->data || !src->data)
		return -1;
	if (dst->rows != src->rows || dst->cols != src->cols)
		return -1;
	if (gpu_mat_add(dst, src) == 0)
		return 0;
	mat_ensure_host(dst);
	mat_ensure_host((mat *)src);
	mat_host_dirty(dst);
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
	if (gpu_mat_copy(dst, src) == 0)
		return 0;
	mat_ensure_host((mat *)src);
	n = dst->rows * dst->cols;
	memcpy(dst->data, src->data, (size_t)n * sizeof(ann_real));
	mat_host_dirty(dst);
	return 0;
}
