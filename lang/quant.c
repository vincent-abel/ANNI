/*
**	Int8 weight quantization for inference
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "quant.h"

qmat *qmat_from_mat(const mat *m)
{
	qmat *q;
	int n;
	int i;
	double amax;
	double v;

	if (!m || !m->data || m->rows <= 0 || m->cols <= 0)
		return NULL;
	q = malloc(sizeof(qmat));
	if (!q)
		return NULL;
	n = m->rows * m->cols;
	q->data = malloc((size_t)n * sizeof(int8_t));
	if (!q->data) {
		free(q);
		return NULL;
	}
	q->rows = m->rows;
	q->cols = m->cols;
	amax = 0.0;
	for (i = 0; i < n; i++) {
		v = fabs(m->data[i]);
		if (v > amax)
			amax = v;
	}
	q->scale = (amax > 0.0) ? (amax / 127.0) : 1.0;
	for (i = 0; i < n; i++) {
		v = m->data[i] / q->scale;
		if (v > 127.0)
			v = 127.0;
		if (v < -127.0)
			v = -127.0;
		q->data[i] = (int8_t)lround(v);
	}
	return q;
}

void qmat_free(qmat **q)
{
	if (!q || !*q)
		return;
	free((*q)->data);
	free(*q);
	*q = NULL;
}

int qmat_gemv_t(vec *y, const qmat *q, const double *x)
{
	int i;
	int j;
	double sum;

	if (!y || !q || !x || !y->data || !q->data)
		return -1;
	if (y->n != q->cols)
		return -1;
	for (j = 0; j < q->cols; j++) {
		sum = 0.0;
		for (i = 0; i < q->rows; i++)
			sum += x[i] * (double)q->data[i * q->cols + j];
		y->data[j] = sum * q->scale;
	}
	return 0;
}
