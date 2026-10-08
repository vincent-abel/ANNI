/*
**	Attention (QKV + scaled dot-product + multi-head + causal mask)
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "attn.h"
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
	if (!a->wq || !a->wk || !a->wv || !a->wo) {
		attn_free(&a);
		return NULL;
	}
	fill_rand(a->wq);
	fill_rand(a->wk);
	fill_rand(a->wv);
	fill_rand(a->wo);
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
	free(*a);
	*a = NULL;
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

static int softmax_row(double *row, int n)
{
	vec in;
	vec out;
	double *tmp;
	int i;
	int rc;

	tmp = malloc((size_t)n * sizeof(double));
	if (!tmp)
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
	free(tmp);
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
	if (!scores)
		return -1;

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
		if (softmax_row(&scores->data[i * T], T) != 0) {
			mat_free(&scores);
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

int attn_mha(mat *out, const attn *a, const mat *x, int causal)
{
	mat *q;
	mat *k;
	mat *v;
	mat *q_h;
	mat *k_h;
	mat *v_h;
	mat *o_h;
	mat *concat;
	mat *proj;
	int h;
	int T;
	int rc;

	if (!out || !a || !x)
		return -1;
	T = x->rows;
	q = mat_create(T, LM_D_MODEL);
	k = mat_create(T, LM_D_MODEL);
	v = mat_create(T, LM_D_MODEL);
	concat = mat_create(T, LM_D_MODEL);
	proj = mat_create(T, LM_D_MODEL);
	q_h = mat_create(T, a->d_head);
	k_h = mat_create(T, a->d_head);
	v_h = mat_create(T, a->d_head);
	o_h = mat_create(T, a->d_head);
	if (!q || !k || !v || !concat || !proj || !q_h || !k_h || !v_h || !o_h) {
		rc = -1;
		goto done;
	}
	if (attn_qkv_project(q, k, v, a, x) != 0) {
		rc = -1;
		goto done;
	}
	mat_fill(concat, 0.0);
	for (h = 0; h < a->n_heads; h++) {
		copy_head(q_h, q, h, a->d_head);
		copy_head(k_h, k, h, a->d_head);
		copy_head(v_h, v, h, a->d_head);
		if (attn_self(o_h, q_h, k_h, v_h, causal) != 0) {
			rc = -1;
			goto done;
		}
		write_head(concat, o_h, h, a->d_head);
	}
	if (mat_mul(proj, concat, a->wo) != 0) {
		rc = -1;
		goto done;
	}
	memcpy(out->data, proj->data, (size_t)(T * LM_D_MODEL) * sizeof(double));
	rc = 0;
done:
	mat_free(&q);
	mat_free(&k);
	mat_free(&v);
	mat_free(&concat);
	mat_free(&proj);
	mat_free(&q_h);
	mat_free(&k_h);
	mat_free(&v_h);
	mat_free(&o_h);
	return rc;
}
