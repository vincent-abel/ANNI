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
	m->data = calloc((size_t)rows * (size_t)cols, sizeof(ann_real));
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

mat *mat_create_dev(int rows, int cols)
{
	mat *m;
	size_t n;

	if (!gpu_available())
		return mat_create(rows, cols);
	if (rows <= 0 || cols <= 0)
		return NULL;
	m = malloc(sizeof(mat));
	if (!m)
		return NULL;
	m->data = NULL;
	m->d_data = NULL;
	m->rows = rows;
	m->cols = cols;
	m->on_device = MAT_DEV_DEVICE;
	n = (size_t)rows * (size_t)cols * sizeof(ann_real);
	m->d_data = (ann_real *)gpu_malloc(n);
	if (!m->d_data) {
		free(m);
		return NULL;
	}
	(void)gpu_memset(m->d_data, 0, n);
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

void mat_release_host(mat *m)
{
	if (!m || !m->data)
		return;
	free(m->data);
	m->data = NULL;
	if (m->d_data)
		m->on_device = MAT_DEV_DEVICE;
}

int mat_device_alloc(mat *m)
{
	size_t n;

	if (!m)
		return -1;
	if (m->d_data)
		return 0;
	if (!gpu_available())
		return -1;
	n = (size_t)m->rows * (size_t)m->cols * sizeof(ann_real);
	m->d_data = (ann_real *)gpu_malloc(n);
	if (!m->d_data)
		return -1;
	if (!m->data)
		m->on_device = MAT_DEV_DEVICE;
	return 0;
}

void mat_device_free(mat *m)
{
	if (!m || !m->d_data)
		return;
	if (m->on_device != MAT_DEV_ALIASED)
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
	n = (size_t)m->rows * (size_t)m->cols * sizeof(ann_real);
	if (gpu_memcpy_h2d(m->d_data, m->data, n) != 0)
		return -1;
	m->on_device = MAT_DEV_DEVICE;
	return 0;
}

static int mat_alloc_host(mat *m)
{
	size_t n;

	if (!m)
		return -1;
	if (m->data)
		return 0;
	n = (size_t)m->rows * (size_t)m->cols;
	m->data = calloc(n, sizeof(ann_real));
	return m->data ? 0 : -1;
}

int mat_to_host(mat *m)
{
	size_t n;
	int aliased;

	if (!m || !m->d_data)
		return -1;
	aliased = (m->on_device == MAT_DEV_ALIASED);
	if (mat_alloc_host(m) != 0)
		return -1;
	n = (size_t)m->rows * (size_t)m->cols * sizeof(ann_real);
	if (gpu_memcpy_d2h(m->data, m->d_data, n) != 0)
		return -1;
	/* Preserve Adam flat-buffer alias; otherwise host is authoritative. */
	m->on_device = aliased ? MAT_DEV_ALIASED : MAT_DEV_HOST;
	return 0;
}

int mat_ensure_host(mat *m)
{
	if (!m)
		return -1;
	if (m->d_data && (m->on_device == MAT_DEV_DEVICE
	    || m->on_device == MAT_DEV_ALIASED))
		return mat_to_host(m);
	if (!m->data)
		return mat_alloc_host(m);
	return 0;
}

/* Read-only pull: D2H when device-authoritative, host+twin end up identical. */
int mat_pull_host(mat *m)
{
	size_t n;
	int aliased;

	if (!m)
		return -1;
	if ((m->on_device != MAT_DEV_DEVICE && m->on_device != MAT_DEV_ALIASED)
	    || !m->d_data)
		return 0;
	aliased = (m->on_device == MAT_DEV_ALIASED);
	if (mat_alloc_host(m) != 0)
		return -1;
	n = (size_t)m->rows * (size_t)m->cols * sizeof(ann_real);
	if (gpu_memcpy_d2h(m->data, m->d_data, n) != 0)
		return -1;
	m->on_device = aliased ? MAT_DEV_ALIASED : MAT_DEV_SYNCED;
	return 0;
}

int mat_sync_device(mat *m)
{
	size_t n;

	if (!m || !m->data)
		return -1;
	if (mat_device_alloc(m) != 0)
		return -1;
	n = (size_t)m->rows * (size_t)m->cols * sizeof(ann_real);
	if (gpu_memcpy_h2d(m->d_data, m->data, n) != 0)
		return -1;
	m->on_device = MAT_DEV_SYNCED;
	return 0;
}

void mat_host_dirty(mat *m)
{
	if (m)
		m->on_device = MAT_DEV_HOST;
}

void mat_try_pin(mat *m)
{
	if (!m)
		return;
	(void)mat_device_alloc(m);
}

ann_real *mat_compute_ptr(mat *m)
{
	if (!m)
		return NULL;
	if (m->d_data && m->on_device != MAT_DEV_HOST)
		return m->d_data;
	return m->data;
}

const ann_real *mat_compute_ptr_const(const mat *m)
{
	if (!m)
		return NULL;
	if (m->d_data && m->on_device != MAT_DEV_HOST)
		return m->d_data;
	return m->data;
}

void mat_mark_device(mat *m)
{
	if (m && m->d_data && m->on_device != MAT_DEV_ALIASED)
		m->on_device = MAT_DEV_DEVICE;
}

/*
** A kernel wrote through p = mat_compute_ptr(m). When p was the device twin
** the device is authoritative; when it was the host pointer (CPU fallback
** earlier in the step) the host is. Marking device unconditionally lost the
** host result and served the stale twin.
*/
void mat_mark_written(mat *m, const ann_real *p)
{
	if (!m)
		return;
	if (p && p == m->d_data)
		mat_mark_device(m);
	else
		mat_host_dirty(m);
}

ann_real mat_get(const mat *m, int row, int col)
{
	if (!m || row < 0 || col < 0 || row >= m->rows || col >= m->cols)
		return 0.0f;
	if (mat_ensure_host((mat *)m) != 0 || !m->data)
		return 0.0f;
	return m->data[row * m->cols + col];
}

void mat_set(mat *m, int row, int col, ann_real value)
{
	if (!m || row < 0 || col < 0 || row >= m->rows || col >= m->cols)
		return;
	if (mat_ensure_host(m) != 0 || !m->data)
		return;
	mat_host_dirty(m);
	m->data[row * m->cols + col] = value;
}

ann_real *mat_row(mat *m, int row)
{
	if (!m || row < 0 || row >= m->rows)
		return NULL;
	if (mat_ensure_host(m) != 0 || !m->data)
		return NULL;
	mat_host_dirty(m);	/* caller may write through the pointer */
	return &m->data[row * m->cols];
}

const ann_real *mat_row_const(const mat *m, int row)
{
	if (!m || row < 0 || row >= m->rows)
		return NULL;
	if (mat_ensure_host((mat *)m) != 0 || !m->data)
		return NULL;
	return &m->data[row * m->cols];
}

void mat_fill(mat *m, ann_real value)
{
	int i;
	int n;

	if (!m)
		return;
	if (m->d_data && m->on_device != MAT_DEV_HOST && value == 0.0f
	    && gpu_available()) {
		(void)gpu_memset(m->d_data, 0,
		    (size_t)m->rows * (size_t)m->cols * sizeof(ann_real));
		mat_mark_device(m);
		return;
	}
	if (mat_ensure_host(m) != 0 || !m->data)
		return;
	n = m->rows * m->cols;
	for (i = 0; i < n; i++)
		m->data[i] = value;
	mat_host_dirty(m);
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

	if (!c || !a || !b)
		return -1;
	if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols)
		return -1;
	if (gpu_available() && gpu_mat_mul(c, a, b) == 0)
		return 0;

	/* CPU path: inputs may be device-authoritative (BATCH mode). */
	if (mat_ensure_host((mat *)a) != 0 || mat_ensure_host((mat *)b) != 0
	    || mat_ensure_host(c) != 0)
		return -1;
	mat_fill(c, 0.0f);
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
	const ann_real *ar;
	const ann_real *br;

	if (!c || !a || !b)
		return -1;
	if (a->cols != b->cols || c->rows != a->rows || c->cols != b->rows)
		return -1;
	if (gpu_mat_mul_bt(c, a, b) == 0)
		return 0;
	if (mat_ensure_host((mat *)a) != 0 || mat_ensure_host((mat *)b) != 0
	    || mat_ensure_host(c) != 0)
		return -1;
	mat_host_dirty(c);
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
