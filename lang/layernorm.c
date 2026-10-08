/*
**	Layer normalization (per token row)
*/
#include <math.h>
#include <stdlib.h>
#include "layernorm.h"

layernorm *layernorm_create(void)
{
	layernorm *ln;

	ln = malloc(sizeof(layernorm));
	if (!ln)
		return NULL;
	ln->gamma = vec_create(LM_D_MODEL);
	ln->beta = vec_create(LM_D_MODEL);
	if (!ln->gamma || !ln->beta) {
		layernorm_free(&ln);
		return NULL;
	}
	vec_fill(ln->gamma, 1.0);
	vec_fill(ln->beta, 0.0);
	ln->eps = 1e-5;
	return ln;
}

void layernorm_free(layernorm **ln)
{
	if (!ln || !*ln)
		return;
	vec_free(&(*ln)->gamma);
	vec_free(&(*ln)->beta);
	free(*ln);
	*ln = NULL;
}

int layernorm_forward(mat *x, const layernorm *ln)
{
	int t;
	int j;
	int T;
	int d;
	double mean;
	double var;
	double inv_std;
	double v;

	if (!x || !ln || !ln->gamma || !ln->beta || !x->data)
		return -1;
	if (x->cols != LM_D_MODEL)
		return -1;
	if (ln->gamma->n != LM_D_MODEL || ln->beta->n != LM_D_MODEL)
		return -1;

	T = x->rows;
	d = x->cols;
	for (t = 0; t < T; t++) {
		mean = 0.0;
		for (j = 0; j < d; j++)
			mean += mat_get(x, t, j);
		mean /= (double)d;

		var = 0.0;
		for (j = 0; j < d; j++) {
			v = mat_get(x, t, j) - mean;
			var += v * v;
		}
		var /= (double)d;

		inv_std = 1.0 / sqrt(var + ln->eps);
		for (j = 0; j < d; j++) {
			v = (mat_get(x, t, j) - mean) * inv_std;
			v = v * ln->gamma->data[j] + ln->beta->data[j];
			mat_set(x, t, j, v);
		}
	}
	return 0;
}
