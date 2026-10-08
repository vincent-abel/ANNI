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
	c->q_h = mat_create(T, d_head);
	c->k_h = mat_create(T, d_head);
	c->o_h = mat_create(T, d_head);
	c->softmax_tmp = calloc((size_t)T, sizeof(double));
	c->attn_w = calloc((size_t)n_heads, sizeof(mat *));
	c->v_h = calloc((size_t)n_heads, sizeof(mat *));
	if (!c->x_in || !c->q || !c->k || !c->v || !c->concat
	    || !c->q_h || !c->k_h || !c->o_h || !c->softmax_tmp
	    || !c->attn_w || !c->v_h) {
		attn_cache_free(&c);
		return NULL;
	}
	for (h = 0; h < n_heads; h++) {
		c->attn_w[h] = mat_create(T, T);
		c->v_h[h] = mat_create(T, d_head);
		if (!c->attn_w[h] || !c->v_h[h]) {
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
	mat_free(&(*c)->q_h);
	mat_free(&(*c)->k_h);
	mat_free(&(*c)->o_h);
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
	int d;
	int i;
	int j;
	int t;
	double scale;
	double sum;
	mat *scores;
	double *tmp;

	if (!out || !q || !k || !v || !out->data || !q->data || !k->data || !v->data)
		return -1;
	if (q->rows != k->rows || q->rows != v->rows || q->cols != k->cols || q->cols != v->cols)
		return -1;
	if (out->rows != q->rows || out->cols != v->cols)
		return -1;
	T = q->rows;
	d = q->cols;
	scale = 1.0 / sqrt((double)d);
	scores = mat_create(T, T);
	tmp = calloc((size_t)T, sizeof(double));
	if (!scores || !tmp) {
		mat_free(&scores);
		free(tmp);
		return -1;
	}

	for (i = 0; i < T; i++) {
		for (j = 0; j < T; j++) {
			sum = 0.0;
			for (t = 0; t < d; t++)
				sum += mat_get(q, i, t) * mat_get(k, j, t);
			sum *= scale;
			if (causal && j > i)
				sum = -1e9;
			mat_set(scores, i, j, sum);
		}
		if (softmax_row(&scores->data[i * T], T, tmp) != 0) {
			mat_free(&scores);
			free(tmp);
			return -1;
		}
	}

	mat_fill(out, 0.0);
	for (i = 0; i < T; i++) {
		for (t = 0; t < d; t++) {
			sum = 0.0;
			for (j = 0; j < T; j++)
				sum += mat_get(scores, i, j) * mat_get(v, j, t);
			mat_set(out, i, t, sum);
		}
	}
	mat_free(&scores);
	free(tmp);
	return 0;
}

static void copy_head(mat *dst, const mat *src, int head, int d_head)
{
	int t;
	int j;
	int off;

	off = head * d_head;
	for (t = 0; t < src->rows; t++) {
		for (j = 0; j < d_head; j++)
			mat_set(dst, t, j, mat_get(src, t, off + j));
	}
}

static void write_head(mat *dst, const mat *src, int head, int d_head)
{
	int t;
	int j;
	int off;

	off = head * d_head;
	for (t = 0; t < src->rows; t++) {
		for (j = 0; j < d_head; j++)
			mat_set(dst, t, off + j, mat_get(src, t, j));
	}
}

static int self_forward_cache(mat *out, const mat *q, const mat *k, const mat *v,
	int causal, mat *attn_w, double *softmax_tmp)
{
	int T;
	int d;
	int i;
	int j;
	int t;
	double scale;
	double sum;

	if (!softmax_tmp)
		return -1;
	T = q->rows;
	d = q->cols;
	scale = 1.0 / sqrt((double)d);
	for (i = 0; i < T; i++) {
		for (j = 0; j < T; j++) {
			sum = 0.0;
			for (t = 0; t < d; t++)
				sum += mat_get(q, i, t) * mat_get(k, j, t);
			sum *= scale;
			if (causal && j > i)
				sum = -1e9;
			mat_set(attn_w, i, j, sum);
		}
		if (softmax_row(&attn_w->data[i * T], T, softmax_tmp) != 0)
			return -1;
	}
	mat_fill(out, 0.0);
	for (i = 0; i < T; i++) {
		for (t = 0; t < d; t++) {
			sum = 0.0;
			for (j = 0; j < T; j++)
				sum += mat_get(attn_w, i, j) * mat_get(v, j, t);
			mat_set(out, i, t, sum);
		}
	}
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

	if (!out || !a || !x || !c)
		return -1;
	if (c->T != x->rows || !c->q_h || !c->k_h || !c->o_h || !c->softmax_tmp)
		return -1;
	c->causal = causal;
	copy_mat(c->x_in, x);
	if (attn_qkv_project(c->q, c->k, c->v, a, x) != 0)
		return -1;
	mat_fill(c->concat, 0.0);
	for (h = 0; h < a->n_heads; h++) {
		copy_head(c->q_h, c->q, h, a->d_head);
		copy_head(c->k_h, c->k, h, a->d_head);
		copy_head(c->v_h[h], c->v, h, a->d_head);
		if (self_forward_cache(c->o_h, c->q_h, c->k_h, c->v_h[h], causal,
			c->attn_w[h], c->softmax_tmp) != 0)
			return -1;
		write_head(c->concat, c->o_h, h, a->d_head);
	}
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
	double *drow;
	int h;
	int T;
	int i;
	int j;
	int t;
	int off;
	double scale;
	double sum;

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
	drow = malloc((size_t)T * sizeof(double));
	if (!dconcat || !dq || !dk || !dv || !dq_h || !dk_h || !dv_h || !do_h
	    || !dscores || !drow) {
		mat_free(&dconcat);
		mat_free(&dq);
		mat_free(&dk);
		mat_free(&dv);
		mat_free(&dq_h);
		mat_free(&dk_h);
		mat_free(&dv_h);
		mat_free(&do_h);
		mat_free(&dscores);
		free(drow);
		return -1;
	}

	/* dWo += concat^T dout; dconcat = dout @ Wo^T */
	if (mat_outer_add(a->dwo, c->concat, dout) != 0)
		goto fail;
	for (i = 0; i < T; i++) {
		for (j = 0; j < LM_D_MODEL; j++) {
			sum = 0.0;
			for (t = 0; t < LM_D_MODEL; t++)
				sum += mat_get(dout, i, t) * mat_get(a->wo, j, t);
			mat_set(dconcat, i, j, sum);
		}
	}

	mat_zero(dq);
	mat_zero(dk);
	mat_zero(dv);
	for (h = 0; h < a->n_heads; h++) {
		off = h * a->d_head;
		for (i = 0; i < T; i++) {
			for (j = 0; j < a->d_head; j++)
				mat_set(do_h, i, j, mat_get(dconcat, i, off + j));
		}
		/* dV = A^T do; dA = do V^T */
		mat_zero(dv_h);
		for (i = 0; i < T; i++) {
			for (t = 0; t < a->d_head; t++) {
				sum = 0.0;
				for (j = 0; j < T; j++)
					sum += mat_get(c->attn_w[h], j, i)
					    * mat_get(do_h, j, t);
				mat_set(dv_h, i, t, sum);
			}
		}
		mat_zero(dscores);
		for (i = 0; i < T; i++) {
			for (j = 0; j < T; j++) {
				sum = 0.0;
				for (t = 0; t < a->d_head; t++)
					sum += mat_get(do_h, i, t)
					    * mat_get(c->v_h[h], j, t);
				mat_set(dscores, i, j, sum);
			}
			softmax_bwd_row(&c->attn_w[h]->data[i * T],
			    &dscores->data[i * T], drow, T);
			for (j = 0; j < T; j++) {
				if (c->causal && j > i)
					drow[j] = 0.0;
				mat_set(dscores, i, j, drow[j] * scale);
			}
		}
		/* dQ = dS K; dK = dS^T Q */
		copy_head(dq_h, c->q, h, a->d_head);
		copy_head(dk_h, c->k, h, a->d_head);
		/* reuse dq_h/dk_h as outputs */
		for (i = 0; i < T; i++) {
			for (t = 0; t < a->d_head; t++) {
				sum = 0.0;
				for (j = 0; j < T; j++)
					sum += mat_get(dscores, i, j)
					    * mat_get(c->k, j, off + t);
				mat_set(dq_h, i, t, sum);
				sum = 0.0;
				for (j = 0; j < T; j++)
					sum += mat_get(dscores, j, i)
					    * mat_get(c->q, j, off + t);
				mat_set(dk_h, i, t, sum);
			}
		}
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
	mat_zero(dx);
	for (i = 0; i < T; i++) {
		for (j = 0; j < LM_D_MODEL; j++) {
			sum = 0.0;
			for (t = 0; t < LM_D_MODEL; t++) {
				sum += mat_get(dq, i, t) * mat_get(a->wq, j, t);
				sum += mat_get(dk, i, t) * mat_get(a->wk, j, t);
				sum += mat_get(dv, i, t) * mat_get(a->wv, j, t);
			}
			mat_set(dx, i, j, sum);
		}
	}

	mat_free(&dconcat);
	mat_free(&dq);
	mat_free(&dk);
	mat_free(&dv);
	mat_free(&dq_h);
	mat_free(&dk_h);
	mat_free(&dv_h);
	mat_free(&do_h);
	mat_free(&dscores);
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
	free(drow);
	return -1;
}
