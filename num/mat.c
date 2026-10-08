/*
**	Matrix Code
*/
#include "main.h"
#include "gpu.h"

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
	m->d_data = NULL;
	m->rows = rows;
	m->cols = cols;
	m->on_device = 0;
	return m;
}

void mat_free(mat **m)
{
	if (!m || !*m)
		return;
	mat_device_free(*m);
	free((*m)->data);
	free(*m);
	*m = NULL;
}

int mat_device_alloc(mat *m)
{
	size_t n;

	if (!m || !m->data)
		return -1;
	if (m->d_data)
		return 0;
	if (!gpu_available())
		return -1;
	n = (size_t)m->rows * (size_t)m->cols * sizeof(double);
	m->d_data = (double *)gpu_malloc(n);
	return m->d_data ? 0 : -1;
}

void mat_device_free(mat *m)
{
	if (!m || !m->d_data)
		return;
	gpu_free(m->d_data);
	m->d_data = NULL;
	m->on_device = 0;
}

int mat_to_device(mat *m)
{
	size_t n;

	if (!m || !m->data)
		return -1;
	if (mat_device_alloc(m) != 0)
		return -1;
	n = (size_t)m->rows * (size_t)m->cols * sizeof(double);
	if (gpu_memcpy_h2d(m->d_data, m->data, n) != 0)
		return -1;
	m->on_device = 1;
	return 0;
}

int mat_to_host(mat *m)
{
	size_t n;

	if (!m || !m->data || !m->d_data)
		return -1;
	n = (size_t)m->rows * (size_t)m->cols * sizeof(double);
	if (gpu_memcpy_d2h(m->data, m->d_data, n) != 0)
		return -1;
	m->on_device = 0;
	return 0;
}

int mat_ensure_host(mat *m)
{
	if (!m)
		return -1;
	if (m->on_device && m->d_data)
		return mat_to_host(m);
	return 0;
}

void mat_try_pin(mat *m)
{
	if (!m)
		return;
	(void)mat_device_alloc(m);
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

double *mat_row(mat *m, int row)
{
	if (!m || !m->data || row < 0 || row >= m->rows)
		return NULL;
	return &m->data[row * m->cols];
}

const double *mat_row_const(const mat *m, int row)
{
	if (!m || !m->data || row < 0 || row >= m->rows)
		return NULL;
	return &m->data[row * m->cols];
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
	if (gpu_available() && gpu_mat_mul(c, a, b) == 0)
		return 0;

	mat_fill(c, 0.0);
#ifdef _OPENMP
#pragma omp parallel for private(j0, k0, i, j, k, i_max, j_max, k_max, sum)
#endif
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

/* c = a * b^T ; a (m x k), b (n x k) -> c (m x n) */
int mat_mul_bt(mat *c, const mat *a, const mat *b)
{
	int i;
	int j;
	int t;
	double sum;
	const double *ar;
	const double *br;

	if (!c || !a || !b || !c->data || !a->data || !b->data)
		return -1;
	if (a->cols != b->cols || c->rows != a->rows || c->cols != b->rows)
		return -1;
	if (gpu_mat_mul_bt(c, a, b) == 0)
		return 0;
	for (i = 0; i < a->rows; i++) {
		ar = &a->data[i * a->cols];
		for (j = 0; j < b->rows; j++) {
			br = &b->data[j * b->cols];
			sum = 0.0;
			for (t = 0; t < a->cols; t++)
				sum += ar[t] * br[t];
			c->data[i * c->cols + j] = sum;
		}
	}
	return 0;
}
