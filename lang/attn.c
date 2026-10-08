/*
**	Attention (QKV + scaled dot-product + multi-head + causal + backward)
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "attn.h"
#include "gradutil.h"
#include "ops.h"
#include "vec.h"

static double rand_w(void)
{
	return ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * 0.02;
}

static void fill_rand(mat *m)
{
	int i;
	int n;

	if (!m || !m->data)
		return;
	n = m->rows * m->cols;
	for (i = 0; i < n; i++)
		m->data[i] = rand_w();
}

static void copy_mat(mat *dst, const mat *src)
{
	memcpy(dst->data, src->data,
	    (size_t)dst->rows * (size_t)dst->cols * sizeof(double));
}

attn *attn_create(void)
{
	attn *a;

	if (LM_D_MODEL % LM_N_HEADS != 0)
		return NULL;
	a = malloc(sizeof(attn));
	if (!a)
		return NULL;
	a->n_heads = LM_N_HEADS;
	a->d_head = LM_D_HEAD;
	a->wq = mat_create(LM_D_MODEL, LM_D_MODEL);
	a->wk = mat_create(LM_D_MODEL, LM_D_MODEL);
	a->wv = mat_create(LM_D_MODEL, LM_D_MODEL);
	a->wo = mat_create(LM_D_MODEL, LM_D_MODEL);
	a->dwq = mat_create(LM_D_MODEL, LM_D_MODEL);
	a->dwk = mat_create(LM_D_MODEL, LM_D_MODEL);
	a->dwv = mat_create(LM_D_MODEL, LM_D_MODEL);
	a->dwo = mat_create(LM_D_MODEL, LM_D_MODEL);
	if (!a->wq || !a->wk || !a->wv || !a->wo
	    || !a->dwq || !a->dwk || !a->dwv || !a->dwo) {
		attn_free(&a);
		return NULL;
	}
	fill_rand(a->wq);
	fill_rand(a->wk);
	fill_rand(a->wv);
	fill_rand(a->wo);
	attn_zero_grad(a);
	return a;
}

void attn_free(attn **a)
{
	if (!a || !*a)
		return;
	mat_free(&(*a)->wq);
	mat_free(&(*a)->wk);
	mat_free(&(*a)->wv);
	mat_free(&(*a)->wo);
	mat_free(&(*a)->dwq);
	mat_free(&(*a)->dwk);
	mat_free(&(*a)->dwv);
	mat_free(&(*a)->dwo);
	free(*a);
	*a = NULL;
}

void attn_zero_grad(attn *a)
{
	if (!a)
		return;
	mat_zero(a->dwq);
	mat_zero(a->dwk);
	mat_zero(a->dwv);
	mat_zero(a->dwo);
}

attn_cache *attn_cache_create(int T, int n_heads, int d_head)
{
	attn_cache *c;
	int h;

	if (T <= 0 || n_heads <= 0 || d_head <= 0)
		return NULL;
	c = malloc(sizeof(attn_cache));
	if (!c)
		return NULL;
	memset(c, 0, sizeof(*c));
	c->T = T;
	c->n_heads = n_heads;
	c->d_head = d_head;
	c->x_in = mat_create(T, LM_D_MODEL);
	c->q = mat_create(T, LM_D_MODEL);
	c->k = mat_create(T, LM_D_MODEL);
	c->v = mat_create(T, LM_D_MODEL);
	c->concat = mat_create(T, LM_D_MODEL);
	c->q_h = calloc((size_t)n_heads, sizeof(mat *));
	c->k_h = calloc((size_t)n_heads, sizeof(mat *));
	c->o_h = calloc((size_t)n_heads, sizeof(mat *));
	c->softmax_tmp = calloc((size_t)n_heads * (size_t)T, sizeof(double));
	c->attn_w = calloc((size_t)n_heads, sizeof(mat *));
	c->v_h = calloc((size_t)n_heads, sizeof(mat *));
	if (!c->x_in || !c->q || !c->k || !c->v || !c->concat
	    || !c->q_h || !c->k_h || !c->o_h || !c->softmax_tmp
	    || !c->attn_w || !c->v_h) {
		attn_cache_free(&c);
		return NULL;
	}
	/* Pin activation mats on GPU when available (no-op on CPU stub). */
	mat_try_pin(c->x_in);
	mat_try_pin(c->q);
	mat_try_pin(c->k);
	mat_try_pin(c->v);
	mat_try_pin(c->concat);
	for (h = 0; h < n_heads; h++) {
		c->attn_w[h] = mat_create(T, T);
		c->v_h[h] = mat_create(T, d_head);
		c->q_h[h] = mat_create(T, d_head);
		c->k_h[h] = mat_create(T, d_head);
		c->o_h[h] = mat_create(T, d_head);
		if (!c->attn_w[h] || !c->v_h[h]
		    || !c->q_h[h] || !c->k_h[h] || !c->o_h[h]) {
			attn_cache_free(&c);
			return NULL;
		}
	}
	return c;
}

void attn_cache_free(attn_cache **c)
{
	int h;

	if (!c || !*c)
		return;
	mat_free(&(*c)->x_in);
	mat_free(&(*c)->q);
	mat_free(&(*c)->k);
	mat_free(&(*c)->v);
	mat_free(&(*c)->concat);
	free((*c)->softmax_tmp);
	if ((*c)->attn_w) {
		for (h = 0; h < (*c)->n_heads; h++)
			mat_free(&(*c)->attn_w[h]);
		free((*c)->attn_w);
	}
	if ((*c)->v_h) {
		for (h = 0; h < (*c)->n_heads; h++)
			mat_free(&(*c)->v_h[h]);
		free((*c)->v_h);
	}
	if ((*c)->q_h) {
		for (h = 0; h < (*c)->n_heads; h++)
			mat_free(&(*c)->q_h[h]);
		free((*c)->q_h);
	}
	if ((*c)->k_h) {
		for (h = 0; h < (*c)->n_heads; h++)
			mat_free(&(*c)->k_h[h]);
		free((*c)->k_h);
	}
	if ((*c)->o_h) {
		for (h = 0; h < (*c)->n_heads; h++)
			mat_free(&(*c)->o_h[h]);
		free((*c)->o_h);
	}
	free(*c);
	*c = NULL;
}

int attn_qkv_project(mat *q, mat *k, mat *v, const attn *a, const mat *x)
{
	if (!q || !k || !v || !a || !x)
		return -1;
	if (mat_mul(q, x, a->wq) != 0)
		return -1;
	if (mat_mul(k, x, a->wk) != 0)
		return -1;
	if (mat_mul(v, x, a->wv) != 0)
		return -1;
	return 0;
}

static int softmax_row(double *row, int n, double *tmp)
{
	vec in;
	vec out;
	int i;
	int rc;

	if (!row || !tmp || n <= 0)
		return -1;
	in.data = row;
	in.n = n;
	out.data = tmp;
	out.n = n;
	rc = vec_softmax(&out, &in);
	if (rc == 0) {
		for (i = 0; i < n; i++)
			row[i] = tmp[i];
	}
	return rc;
}

int attn_self(mat *out, const mat *q, const mat *k, const mat *v, int causal)
{
	int T;
	int i;
	int j;
	double scale;
	mat *scores;
	double *tmp;

	if (!out || !q || !k || !v || !out->data || !q->data || !k->data || !v->data)
		return -1;
	if (q->rows != k->rows || q->rows != v->rows || q->cols != k->cols || q->cols != v->cols)
		return -1;
	if (out->rows != q->rows || out->cols != v->cols)
		return -1;
	T = q->rows;
	scale = 1.0 / sqrt((double)q->cols);
	scores = mat_create(T, T);
	tmp = calloc((size_t)T, sizeof(double));
	if (!scores || !tmp) {
		mat_free(&scores);
		free(tmp);
		return -1;
	}
	if (mat_mul_bt(scores, q, k) != 0) {
		mat_free(&scores);
		free(tmp);
		return -1;
	}
	for (i = 0; i < T; i++) {
		for (j = 0; j < T; j++) {
			scores->data[i * T + j] *= scale;
			if (causal && j > i)
				scores->data[i * T + j] = -1e9;
		}
		if (softmax_row(&scores->data[i * T], T, tmp) != 0) {
			mat_free(&scores);
			free(tmp);
			return -1;
		}
	}
	if (mat_mul(out, scores, v) != 0) {
		mat_free(&scores);
		free(tmp);
		return -1;
	}
	mat_free(&scores);
	free(tmp);
	return 0;
}

static void copy_head(mat *dst, const mat *src, int head, int d_head)
{
	int t;
	int off;

	off = head * d_head;
	for (t = 0; t < src->rows; t++)
		memcpy(&dst->data[t * dst->cols],
		    &src->data[t * src->cols + off],
		    (size_t)d_head * sizeof(double));
}

static void write_head(mat *dst, const mat *src, int head, int d_head)
{
	int t;
	int off;

	off = head * d_head;
	for (t = 0; t < src->rows; t++)
		memcpy(&dst->data[t * dst->cols + off],
		    &src->data[t * src->cols],
		    (size_t)d_head * sizeof(double));
}

static int self_forward_cache(mat *out, const mat *q, const mat *k, const mat *v,
	int causal, mat *attn_w, double *softmax_tmp)
{
	int T;
	int d;
	int i;
	int j;
	double scale;
	double *row;

	if (!softmax_tmp || !out || !q || !k || !v || !attn_w)
		return -1;
	T = q->rows;
	d = q->cols;
	scale = 1.0 / sqrt((double)d);
	if (mat_mul_bt(attn_w, q, k) != 0)
		return -1;
	for (i = 0; i < T; i++) {
		row = &attn_w->data[i * T];
		for (j = 0; j < T; j++) {
			row[j] *= scale;
			if (causal && j > i)
				row[j] = -1e9;
		}
		if (softmax_row(row, T, softmax_tmp) != 0)
			return -1;
	}
	if (mat_mul(out, attn_w, v) != 0)
		return -1;
	return 0;
}

int attn_mha(mat *out, const attn *a, const mat *x, int causal)
{
	attn_cache *c;
	int rc;

	if (!out || !a || !x)
		return -1;
	c = attn_cache_create(x->rows, a->n_heads, a->d_head);
	if (!c)
		return -1;
	rc = attn_mha_forward_cache(out, (attn *)a, x, causal, c);
	attn_cache_free(&c);
	return rc;
}

int attn_mha_forward_cache(mat *out, attn *a, const mat *x, int causal,
	attn_cache *c)
{
	int h;
	int fail;

	if (!out || !a || !x || !c)
		return -1;
	if (c->T != x->rows || !c->q_h || !c->k_h || !c->o_h || !c->softmax_tmp)
		return -1;
	c->causal = causal;
	copy_mat(c->x_in, x);
	if (attn_qkv_project(c->q, c->k, c->v, a, x) != 0)
		return -1;
	mat_fill(c->concat, 0.0);
	fail = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(|:fail)
#endif
	for (h = 0; h < a->n_heads; h++) {
		copy_head(c->q_h[h], c->q, h, a->d_head);
		copy_head(c->k_h[h], c->k, h, a->d_head);
		copy_head(c->v_h[h], c->v, h, a->d_head);
		if (self_forward_cache(c->o_h[h], c->q_h[h], c->k_h[h],
			c->v_h[h], causal, c->attn_w[h],
			&c->softmax_tmp[h * c->T]) != 0)
			fail = 1;
		else
			write_head(c->concat, c->o_h[h], h, a->d_head);
	}
	if (fail)
		return -1;
	if (mat_mul(out, c->concat, a->wo) != 0)
		return -1;
	return 0;
}

static void softmax_bwd_row(const double *y, const double *dy, double *dx, int n)
{
	int i;
	double sum;

	sum = 0.0;
	for (i = 0; i < n; i++)
		sum += y[i] * dy[i];
	for (i = 0; i < n; i++)
		dx[i] = y[i] * (dy[i] - sum);
}

static int mat_transpose_copy(mat *dst, const mat *src)
{
	int i;
	int j;

	if (!dst || !src || dst->rows != src->cols || dst->cols != src->rows)
		return -1;
	for (i = 0; i < src->rows; i++)
		for (j = 0; j < src->cols; j++)
			dst->data[j * dst->cols + i] = src->data[i * src->cols + j];
	return 0;
}

int attn_mha_backward(mat *dx, const mat *dout, attn *a, const attn_cache *c)
{
	mat *dconcat;
	mat *dq;
	mat *dk;
	mat *dv;
	mat *dq_h;
	mat *dk_h;
	mat *dv_h;
	mat *do_h;
	mat *dscores;
	mat *wot;
	mat *tmp;
	mat *k_h;
	mat *q_h;
	double *drow;
	int h;
	int T;
	int i;
	int j;
	double scale;

	if (!dx || !dout || !a || !c)
		return -1;
	T = c->T;
	scale = 1.0 / sqrt((double)a->d_head);
	dconcat = mat_create(T, LM_D_MODEL);
	dq = mat_create(T, LM_D_MODEL);
	dk = mat_create(T, LM_D_MODEL);
	dv = mat_create(T, LM_D_MODEL);
	dq_h = mat_create(T, a->d_head);
	dk_h = mat_create(T, a->d_head);
	dv_h = mat_create(T, a->d_head);
	do_h = mat_create(T, a->d_head);
	dscores = mat_create(T, T);
	wot = mat_create(LM_D_MODEL, LM_D_MODEL);
	tmp = mat_create(T, LM_D_MODEL);
	k_h = mat_create(T, a->d_head);
	q_h = mat_create(T, a->d_head);
	drow = malloc((size_t)T * sizeof(double));
	if (!dconcat || !dq || !dk || !dv || !dq_h || !dk_h || !dv_h || !do_h
	    || !dscores || !wot || !tmp || !k_h || !q_h || !drow)
		goto fail;

	/* Weight grads via outer_add (GPU when available). */
	if (mat_outer_add(a->dwo, c->concat, dout) != 0)
		goto fail;
	/* dconcat = dout * Wo^T  (GEMM / GPU). */
	if (mat_transpose_copy(wot, a->wo) != 0)
		goto fail;
	if (mat_mul(dconcat, dout, wot) != 0)
		goto fail;

	mat_zero(dq);
	mat_zero(dk);
	mat_zero(dv);
	for (h = 0; h < a->n_heads; h++) {
		copy_head(do_h, dconcat, h, a->d_head);
		copy_head(k_h, c->k, h, a->d_head);
		copy_head(q_h, c->q, h, a->d_head);
		/* dV = A^T do ; dA_scores = do V^T  (GEMM). */
		if (mat_mul_at(dv_h, c->attn_w[h], do_h) != 0)
			goto fail;
		if (mat_mul_bt(dscores, do_h, c->v_h[h]) != 0)
			goto fail;
		for (i = 0; i < T; i++) {
			softmax_bwd_row(&c->attn_w[h]->data[i * T],
			    &dscores->data[i * T], drow, T);
			for (j = 0; j < T; j++) {
				if (c->causal && j > i)
					drow[j] = 0.0;
				dscores->data[i * T + j] = drow[j] * scale;
			}
		}
		/* dQ = dS K ; dK = dS^T Q */
		if (mat_mul(dq_h, dscores, k_h) != 0)
			goto fail;
		if (mat_mul_at(dk_h, dscores, q_h) != 0)
			goto fail;
		write_head(dq, dq_h, h, a->d_head);
		write_head(dk, dk_h, h, a->d_head);
		write_head(dv, dv_h, h, a->d_head);
	}

	if (mat_outer_add(a->dwq, c->x_in, dq) != 0)
		goto fail;
	if (mat_outer_add(a->dwk, c->x_in, dk) != 0)
		goto fail;
	if (mat_outer_add(a->dwv, c->x_in, dv) != 0)
		goto fail;

	/* dx = dq Wq^T + dk Wk^T + dv Wv^T */
	if (mat_transpose_copy(wot, a->wq) != 0)
		goto fail;
	if (mat_mul(dx, dq, wot) != 0)
		goto fail;
	if (mat_transpose_copy(wot, a->wk) != 0)
		goto fail;
	if (mat_mul(tmp, dk, wot) != 0)
		goto fail;
	mat_axpy(dx, 1.0, tmp);
	if (mat_transpose_copy(wot, a->wv) != 0)
		goto fail;
	if (mat_mul(tmp, dv, wot) != 0)
		goto fail;
	mat_axpy(dx, 1.0, tmp);

	mat_free(&dconcat);
	mat_free(&dq);
	mat_free(&dk);
	mat_free(&dv);
	mat_free(&dq_h);
	mat_free(&dk_h);
	mat_free(&dv_h);
	mat_free(&do_h);
	mat_free(&dscores);
	mat_free(&wot);
	mat_free(&tmp);
	mat_free(&k_h);
	mat_free(&q_h);
	free(drow);
	return 0;
fail:
	mat_free(&dconcat);
	mat_free(&dq);
	mat_free(&dk);
	mat_free(&dv);
	mat_free(&dq_h);
	mat_free(&dk_h);
	mat_free(&dv_h);
	mat_free(&do_h);
	mat_free(&dscores);
	mat_free(&wot);
	mat_free(&tmp);
	mat_free(&k_h);
	mat_free(&q_h);
	free(drow);
	return -1;
}
