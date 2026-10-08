/*
**	Small helpers for LM gradient buffers
*/
#include <string.h>
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

	if (!dw || !a || !b)
		return -1;
	if (a->rows != b->rows || dw->rows != a->cols || dw->cols != b->cols)
		return -1;
	for (t = 0; t < a->rows; t++) {
		for (i = 0; i < a->cols; i++) {
			s = mat_get(a, t, i);
			for (j = 0; j < b->cols; j++)
				mat_set(dw, i, j, mat_get(dw, i, j) + s * mat_get(b, t, j));
		}
	}
	return 0;
}

/* c = a^T * b ; a (m x k), b (m x n) -> c (k x n)  OR use rows of a as columns */
int mat_mul_at(mat *c, const mat *a, const mat *b)
{
	int i;
	int j;
	int t;
	double sum;

	if (!c || !a || !b)
		return -1;
	if (a->rows != b->rows || c->rows != a->cols || c->cols != b->cols)
		return -1;
	for (i = 0; i < a->cols; i++) {
		for (j = 0; j < b->cols; j++) {
			sum = 0.0;
			for (t = 0; t < a->rows; t++)
				sum += mat_get(a, t, i) * mat_get(b, t, j);
			mat_set(c, i, j, sum);
		}
	}
	return 0;
}

/* c = a * b^T ; a (m x k), b (n x k) -> c (m x n) */
int mat_mul_bt(mat *c, const mat *a, const mat *b)
{
	int i;
	int j;
	int t;
	double sum;

	if (!c || !a || !b)
		return -1;
	if (a->cols != b->cols || c->rows != a->rows || c->cols != b->rows)
		return -1;
	for (i = 0; i < a->rows; i++) {
		for (j = 0; j < b->rows; j++) {
			sum = 0.0;
			for (t = 0; t < a->cols; t++)
				sum += mat_get(a, i, t) * mat_get(b, j, t);
			mat_set(c, i, j, sum);
		}
	}
	return 0;
}
