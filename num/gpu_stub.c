/*
**	CPU stub for optional GPU backend (default build)
*/
#include <stdlib.h>
#include "gpu.h"

int gpu_available(void)
{
	return 0;
}

const char *gpu_backend_name(void)
{
	return "cpu";
}

void *gpu_malloc(size_t nbytes)
{
	(void)nbytes;
	return NULL;
}

void gpu_free(void *p)
{
	(void)p;
}

int gpu_memcpy_h2d(void *dst, const void *src, size_t nbytes)
{
	(void)dst;
	(void)src;
	(void)nbytes;
	return -1;
}

int gpu_memcpy_d2h(void *dst, const void *src, size_t nbytes)
{
	(void)dst;
	(void)src;
	(void)nbytes;
	return -1;
}

int gpu_memcpy_d2d(void *dst, const void *src, size_t nbytes)
{
	(void)dst;
	(void)src;
	(void)nbytes;
	return -1;
}

int gpu_memset(void *dst, int value, size_t nbytes)
{
	(void)dst;
	(void)value;
	(void)nbytes;
	return -1;
}

int gpu_mat_mul(mat *c, const mat *a, const mat *b)
{
	(void)c;
	(void)a;
	(void)b;
	return -1;
}

int gpu_mat_mul_backward(mat *da, mat *db, const mat *dc,
	const mat *a, const mat *b)
{
	(void)da;
	(void)db;
	(void)dc;
	(void)a;
	(void)b;
	return -1;
}

int gpu_mat_mul_bt(mat *c, const mat *a, const mat *b)
{
	(void)c;
	(void)a;
	(void)b;
	return -1;
}

int gpu_mat_outer_add(mat *dw, const mat *a, const mat *b)
{
	(void)dw;
	(void)a;
	(void)b;
	return -1;
}

int gpu_layernorm_backward(mat *dx, const mat *dy, const double *gamma,
	const double *x_hat, const double *inv_std, int T, int d,
	double *dgamma, double *dbeta)
{
	(void)dx;
	(void)dy;
	(void)gamma;
	(void)x_hat;
	(void)inv_std;
	(void)T;
	(void)d;
	(void)dgamma;
	(void)dbeta;
	return -1;
}
