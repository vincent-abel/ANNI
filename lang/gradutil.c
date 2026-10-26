/*
**	Small helpers for LM gradient buffers
*/
#include <string.h>
#include "gpu.h"
#include "gradutil.h"

void mat_zero(mat *m)
{
	if (!m)
		return;
	if (m->d_data && m->on_device != MAT_DEV_HOST && gpu_available()
	    && gpu_memset(m->d_data, 0,
	    (size_t)m->rows * (size_t)m->cols * sizeof(ann_real)) == 0) {
		mat_mark_device(m);
		return;
	}
	if (mat_ensure_host(m) != 0 || !m->data)
		return;
	memset(m->data, 0, (size_t)m->rows * (size_t)m->cols * sizeof(ann_real));
	mat_host_dirty(m);
}

/*
** Zero a gradient that is only ever written through GPU paths or the
** mat_* helpers (which mark dirty): zero host and twin together so no zero
** upload is needed later. Do NOT use for mats host code writes directly.
*/
void mat_zero_dev(mat *m)
{
	if (!m)
		return;
	if (m->d_data && gpu_available()
	    && gpu_memset(m->d_data, 0,
	    (size_t)m->rows * (size_t)m->cols * sizeof(ann_real)) == 0) {
		if (m->on_device == MAT_DEV_ALIASED)
			return;	/* keep external-buffer alias */
		if (m->data) {
			memset(m->data, 0,
			    (size_t)m->rows * (size_t)m->cols * sizeof(ann_real));
			m->on_device = MAT_DEV_SYNCED;
		} else
			mat_mark_device(m);
		return;
	}
	if (mat_ensure_host(m) != 0 || !m->data)
		return;
	memset(m->data, 0, (size_t)m->rows * (size_t)m->cols * sizeof(ann_real));
	mat_host_dirty(m);
}

void vec_zero(vec *v)
{
	if (v && v->data)
		memset(v->data, 0, (size_t)v->n * sizeof(ann_real));
}

void mat_axpy(mat *y, double a, const mat *x)
{
	int i;
	int n;

	if (!y || !x)
		return;
	if (y->rows != x->rows || y->cols != x->cols)
		return;
	if (gpu_mat_axpy(y, a, x) == 0)
		return;
	if (mat_ensure_host(y) != 0 || mat_ensure_host((mat *)x) != 0)
		return;
	mat_host_dirty(y);
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
	const ann_real *ar;
	const ann_real *br;

	if (!dw || !a || !b)
		return -1;
	if (a->rows != b->rows || dw->rows != a->cols || dw->cols != b->cols)
		return -1;
	if (gpu_mat_outer_add(dw, a, b) == 0)
		return 0;
	if (mat_ensure_host(dw) != 0 || mat_ensure_host((mat *)a) != 0
	    || mat_ensure_host((mat *)b) != 0)
		return -1;
	mat_host_dirty(dw);
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

	if (!c || !a || !b)
		return -1;
	if (a->rows != b->rows || c->rows != a->cols || c->cols != b->cols)
		return -1;
	if (gpu_mat_mul_at(c, a, b) == 0)	/* cuBLAS OP_T, no transpose copy */
		return 0;
	if (mat_ensure_host(c) != 0 || mat_ensure_host((mat *)a) != 0
	    || mat_ensure_host((mat *)b) != 0)
		return -1;
	mat_host_dirty(c);
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

/*
** c = a * w^T ; a (T x n), w (m x n) -> c (T x m). Backward layout for
** y = x W: dx = dy W^T. GPU: one OP_T GEMM straight from the weight twin.
** CPU: host transpose + blocked mat_mul (faster than dot-product bt there).
*/
int mat_mul_wt_add(mat *c, const mat *a, const mat *w, mat *tmp)
{
	if (!c || !a || !w || !tmp)
		return -1;
	if (gpu_mat_mul_bt_acc(c, a, w) == 0)
		return 0;
	if (mat_mul_wt(tmp, a, w) != 0)
		return -1;
	mat_axpy(c, 1.0, tmp);
	return 0;
}

int mat_mul_wt(mat *c, const mat *a, const mat *w)
{
	mat *wt;
	int i;
	int j;
	int rc;

	if (!c || !a || !w)
		return -1;
	if (a->cols != w->cols || c->rows != a->rows || c->cols != w->rows)
		return -1;
	if (gpu_mat_mul_bt(c, a, w) == 0)
		return 0;
	wt = mat_create(w->cols, w->rows);
	if (!wt)
		return -1;
	if (mat_ensure_host((mat *)w) != 0 || mat_ensure_host(c) != 0
	    || mat_ensure_host((mat *)a) != 0) {
		mat_free(&wt);
		return -1;
	}
	for (i = 0; i < w->rows; i++)
		for (j = 0; j < w->cols; j++)
			wt->data[j * wt->cols + i] = w->data[i * w->cols + j];
	rc = mat_mul(c, a, wt);
	mat_free(&wt);
	return rc;
}
