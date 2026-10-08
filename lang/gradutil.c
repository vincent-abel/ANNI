/*
**	Small helpers for LM gradient buffers
*/
#include <string.h>
#include "gpu.h"
#include "gradutil.h"

void mat_zero(mat *m)
{
	if (m && m->data)
		memset(m->data, 0, (size_t)m->rows * (size_t)m->cols * sizeof(double));
}

void vec_zero(vec *v)
{
	if (v && v->data)
		memset(v->data, 0, (size_t)v->n * sizeof(double));
}

void mat_axpy(mat *y, double a, const mat *x)
{
	int i;
	int n;

	if (!y || !x || !y->data || !x->data)
		return;
	if (y->rows != x->rows || y->cols != x->cols)
		return;
	n = y->rows * y->cols;
	for (i = 0; i < n; i++)
		y->data[i] += a * x->data[i];
}

/* dw += a^T b  with a: (T x m), b: (T x n) -> dw (m x n) */
int mat_outer_add(mat *dw, const mat *a, const mat *b)
{
	int t;
	int i;
	int j;
	double s;
	const double *ar;
	const double *br;

	if (!dw || !a || !b || !dw->data || !a->data || !b->data)
		return -1;
	if (a->rows != b->rows || dw->rows != a->cols || dw->cols != b->cols)
		return -1;
	if (gpu_mat_outer_add(dw, a, b) == 0)
		return 0;
	for (t = 0; t < a->rows; t++) {
		ar = &a->data[t * a->cols];
		br = &b->data[t * b->cols];
		for (i = 0; i < a->cols; i++) {
			s = ar[i];
			for (j = 0; j < b->cols; j++)
				dw->data[i * dw->cols + j] += s * br[j];
		}
	}
	return 0;
}

/* c = a^T * b ; a (m x k), b (m x n) -> c (k x n) */
int mat_mul_at(mat *c, const mat *a, const mat *b)
{
	int i;
	int j;
	int t;
	double sum;

	if (!c || !a || !b || !c->data || !a->data || !b->data)
		return -1;
	if (a->rows != b->rows || c->rows != a->cols || c->cols != b->cols)
		return -1;
	for (i = 0; i < a->cols; i++) {
		for (j = 0; j < b->cols; j++) {
			sum = 0.0;
			for (t = 0; t < a->rows; t++)
				sum += a->data[t * a->cols + i] * b->data[t * b->cols + j];
			c->data[i * c->cols + j] = sum;
		}
	}
	return 0;
}
