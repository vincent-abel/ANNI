/*
**	Matrix Code
*/
#include "main.h"

mat *mat_create(int rows, int cols)
{
	mat *m;

	if (rows <= 0 || cols <= 0)
		return NULL;
	m = malloc(sizeof(mat));
	if (!m)
		return NULL;
	m->data = calloc((size_t)rows * (size_t)cols, sizeof(double));
	if (!m->data) {
		free(m);
		return NULL;
	}
	m->rows = rows;
	m->cols = cols;
	return m;
}

void mat_free(mat **m)
{
	if (!m || !*m)
		return;
	free((*m)->data);
	free(*m);
	*m = NULL;
}

double mat_get(const mat *m, int row, int col)
{
	if (!m || !m->data || row < 0 || col < 0 || row >= m->rows || col >= m->cols)
		return 0.0;
	return m->data[row * m->cols + col];
}

void mat_set(mat *m, int row, int col, double value)
{
	if (!m || !m->data || row < 0 || col < 0 || row >= m->rows || col >= m->cols)
		return;
	m->data[row * m->cols + col] = value;
}

void mat_fill(mat *m, double value)
{
	int i;
	int n;

	if (!m || !m->data)
		return;
	n = m->rows * m->cols;
	for (i = 0; i < n; i++)
		m->data[i] = value;
}

int mat_vec_mul(vec *y, const mat *a, const vec *x)
{
	int i;
	int j;
	double sum;

	if (!y || !a || !x || !y->data || !a->data || !x->data)
		return -1;
	if (a->cols != x->n || y->n != a->rows)
		return -1;
	for (i = 0; i < a->rows; i++) {
		sum = 0.0;
		for (j = 0; j < a->cols; j++)
			sum += a->data[i * a->cols + j] * x->data[j];
		y->data[i] = sum;
	}
	return 0;
}

int mat_mul_naive(mat *c, const mat *a, const mat *b)
{
	int i;
	int j;
	int k;
	double sum;

	if (!c || !a || !b || !c->data || !a->data || !b->data)
		return -1;
	if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols)
		return -1;
	for (i = 0; i < a->rows; i++) {
		for (j = 0; j < b->cols; j++) {
			sum = 0.0;
			for (k = 0; k < a->cols; k++)
				sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
			c->data[i * c->cols + j] = sum;
		}
	}
	return 0;
}

#define MAT_BLOCK 32

int mat_mul(mat *c, const mat *a, const mat *b)
{
	int i0;
	int j0;
	int k0;
	int i;
	int j;
	int k;
	int i_max;
	int j_max;
	int k_max;
	double sum;

	if (!c || !a || !b || !c->data || !a->data || !b->data)
		return -1;
	if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols)
		return -1;

	mat_fill(c, 0.0);
	for (i0 = 0; i0 < a->rows; i0 += MAT_BLOCK) {
		i_max = i0 + MAT_BLOCK;
		if (i_max > a->rows)
			i_max = a->rows;
		for (k0 = 0; k0 < a->cols; k0 += MAT_BLOCK) {
			k_max = k0 + MAT_BLOCK;
			if (k_max > a->cols)
				k_max = a->cols;
			for (j0 = 0; j0 < b->cols; j0 += MAT_BLOCK) {
				j_max = j0 + MAT_BLOCK;
				if (j_max > b->cols)
					j_max = b->cols;
				for (i = i0; i < i_max; i++) {
					for (k = k0; k < k_max; k++) {
						sum = a->data[i * a->cols + k];
						for (j = j0; j < j_max; j++)
							c->data[i * c->cols + j] +=
								sum * b->data[k * b->cols + j];
					}
				}
			}
		}
	}
	return 0;
}
