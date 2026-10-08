/*
**	Position-wise feed-forward block (Transformer MLP)
*/
#include <stdlib.h>
#include "ffn.h"

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
	if (!f->w1 || !f->b1 || !f->w2 || !f->b2) {
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
	free(*f);
	*f = NULL;
}

int ffn_forward(mat *x, const ffn *f)
{
	mat *hidden;
	mat *out;
	int t;
	int j;
	int T;

	if (!x || !f || !f->w1 || !f->b1 || !f->w2 || !f->b2 || !x->data)
		return -1;
	if (x->cols != LM_D_MODEL)
		return -1;
	if (f->w1->rows != LM_D_MODEL || f->w1->cols != LM_D_FF)
		return -1;
	if (f->w2->rows != LM_D_FF || f->w2->cols != LM_D_MODEL)
		return -1;
	if (f->b1->n != LM_D_FF || f->b2->n != LM_D_MODEL)
		return -1;

	T = x->rows;
	hidden = mat_create(T, LM_D_FF);
	out = mat_create(T, LM_D_MODEL);
	if (!hidden || !out) {
		mat_free(&hidden);
		mat_free(&out);
		return -1;
	}
	if (mat_mul(hidden, x, f->w1) != 0) {
		mat_free(&hidden);
		mat_free(&out);
		return -1;
	}
	for (t = 0; t < T; t++) {
		for (j = 0; j < LM_D_FF; j++)
			mat_set(hidden, t, j,
			    ffn_relu(mat_get(hidden, t, j) + f->b1->data[j]));
	}
	if (mat_mul(out, hidden, f->w2) != 0) {
		mat_free(&hidden);
		mat_free(&out);
		return -1;
	}
	for (t = 0; t < T; t++) {
		for (j = 0; j < LM_D_MODEL; j++)
			mat_set(x, t, j,
			    mat_get(out, t, j) + f->b2->data[j]);
	}
	mat_free(&hidden);
	mat_free(&out);
	return 0;
}
