/*
**	Position-wise feed-forward block + backward
*/
#include <stdlib.h>
#include <string.h>
#include "ffn.h"
#include "gradutil.h"

static double ffn_rand_weight(void)
{
	double u;

	u = (double)rand() / (double)RAND_MAX;
	return (u * 2.0 - 1.0) * 0.02;
}

static double ffn_relu(double x)
{
	if (x > 0.0)
		return x;
	return 0.0;
}

static void copy_mat(mat *dst, const mat *src)
{
	int n;

	n = dst->rows * dst->cols;
	memcpy(dst->data, src->data, (size_t)n * sizeof(double));
}

ffn *ffn_create(void)
{
	ffn *f;
	int i;
	int n;

	f = malloc(sizeof(ffn));
	if (!f)
		return NULL;
	f->w1 = mat_create(LM_D_MODEL, LM_D_FF);
	f->b1 = vec_create(LM_D_FF);
	f->w2 = mat_create(LM_D_FF, LM_D_MODEL);
	f->b2 = vec_create(LM_D_MODEL);
	f->dw1 = mat_create(LM_D_MODEL, LM_D_FF);
	f->db1 = vec_create(LM_D_FF);
	f->dw2 = mat_create(LM_D_FF, LM_D_MODEL);
	f->db2 = vec_create(LM_D_MODEL);
	if (!f->w1 || !f->b1 || !f->w2 || !f->b2
	    || !f->dw1 || !f->db1 || !f->dw2 || !f->db2) {
		ffn_free(&f);
		return NULL;
	}
	n = f->w1->rows * f->w1->cols;
	for (i = 0; i < n; i++)
		f->w1->data[i] = ffn_rand_weight();
	vec_fill(f->b1, 0.0);
	n = f->w2->rows * f->w2->cols;
	for (i = 0; i < n; i++)
		f->w2->data[i] = ffn_rand_weight();
	vec_fill(f->b2, 0.0);
	ffn_zero_grad(f);
	return f;
}

void ffn_free(ffn **f)
{
	if (!f || !*f)
		return;
	mat_free(&(*f)->w1);
	vec_free(&(*f)->b1);
	mat_free(&(*f)->w2);
	vec_free(&(*f)->b2);
	mat_free(&(*f)->dw1);
	vec_free(&(*f)->db1);
	mat_free(&(*f)->dw2);
	vec_free(&(*f)->db2);
	free(*f);
	*f = NULL;
}

void ffn_zero_grad(ffn *f)
{
	if (!f)
		return;
	mat_zero(f->dw1);
	vec_zero(f->db1);
	mat_zero(f->dw2);
	vec_zero(f->db2);
}

ffn_cache *ffn_cache_create(int T)
{
	ffn_cache *c;

	if (T <= 0)
		return NULL;
	c = malloc(sizeof(ffn_cache));
	if (!c)
		return NULL;
	c->T = T;
	c->x_in = mat_create(T, LM_D_MODEL);
	c->pre_act = mat_create(T, LM_D_FF);
	c->hidden = mat_create(T, LM_D_FF);
	if (!c->x_in || !c->pre_act || !c->hidden) {
		ffn_cache_free(&c);
		return NULL;
	}
	return c;
}

void ffn_cache_free(ffn_cache **c)
{
	if (!c || !*c)
		return;
	mat_free(&(*c)->x_in);
	mat_free(&(*c)->pre_act);
	mat_free(&(*c)->hidden);
	free(*c);
	*c = NULL;
}

int ffn_forward_cache(mat *out, const mat *x, ffn *f, ffn_cache *c)
{
	int t;
	int j;
	int T;

	if (!out || !x || !f || !c || !x->data)
		return -1;
	if (x->cols != LM_D_MODEL || c->T != x->rows)
		return -1;
	T = x->rows;
	copy_mat(c->x_in, x);
	if (mat_mul(c->pre_act, x, f->w1) != 0)
		return -1;
	for (t = 0; t < T; t++) {
		for (j = 0; j < LM_D_FF; j++) {
			mat_set(c->pre_act, t, j,
			    mat_get(c->pre_act, t, j) + f->b1->data[j]);
			mat_set(c->hidden, t, j,
			    ffn_relu(mat_get(c->pre_act, t, j)));
		}
	}
	if (mat_mul(out, c->hidden, f->w2) != 0)
		return -1;
	for (t = 0; t < T; t++) {
		for (j = 0; j < LM_D_MODEL; j++)
			mat_set(out, t, j, mat_get(out, t, j) + f->b2->data[j]);
	}
	return 0;
}

int ffn_forward(mat *x, const ffn *f)
{
	ffn_cache *c;
	mat *out;
	int rc;

	if (!x || !f)
		return -1;
	c = ffn_cache_create(x->rows);
	out = mat_create(x->rows, LM_D_MODEL);
	if (!c || !out) {
		ffn_cache_free(&c);
		mat_free(&out);
		return -1;
	}
	rc = ffn_forward_cache(out, x, (ffn *)f, c);
	if (rc == 0)
		copy_mat(x, out);
	mat_free(&out);
	ffn_cache_free(&c);
	return rc;
}

int ffn_backward(mat *dx, const mat *dout, ffn *f, const ffn_cache *c)
{
	mat *dh;
	int t;
	int j;
	int k;
	int T;
	double sum;

	if (!dx || !dout || !f || !c)
		return -1;
	T = c->T;
	dh = mat_create(T, LM_D_FF);
	if (!dh)
		return -1;

	if (mat_outer_add(f->dw2, c->hidden, dout) != 0) {
		mat_free(&dh);
		return -1;
	}
	for (t = 0; t < T; t++) {
		for (j = 0; j < LM_D_MODEL; j++)
			f->db2->data[j] += mat_get(dout, t, j);
	}
	for (t = 0; t < T; t++) {
		for (j = 0; j < LM_D_FF; j++) {
			sum = 0.0;
			for (k = 0; k < LM_D_MODEL; k++)
				sum += mat_get(dout, t, k) * mat_get(f->w2, j, k);
			if (mat_get(c->pre_act, t, j) <= 0.0)
				sum = 0.0;
			mat_set(dh, t, j, sum);
		}
	}
	if (mat_outer_add(f->dw1, c->x_in, dh) != 0) {
		mat_free(&dh);
		return -1;
	}
	for (t = 0; t < T; t++) {
		for (j = 0; j < LM_D_FF; j++)
			f->db1->data[j] += mat_get(dh, t, j);
	}
	for (t = 0; t < T; t++) {
		for (j = 0; j < LM_D_MODEL; j++) {
			sum = 0.0;
			for (k = 0; k < LM_D_FF; k++)
				sum += mat_get(dh, t, k) * mat_get(f->w1, j, k);
			mat_set(dx, t, j, sum);
		}
	}
	mat_free(&dh);
	return 0;
}
