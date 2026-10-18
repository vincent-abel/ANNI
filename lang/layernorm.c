/*
**	Layer normalization (per token row) + backward
*/
#include <math.h>
#include <stdlib.h>
#include "gpu.h"
#include "gradutil.h"
#include "layernorm.h"
#include "resid.h"

layernorm *layernorm_create(void)
{
	layernorm *ln;

	ln = malloc(sizeof(layernorm));
	if (!ln)
		return NULL;
	ln->gamma = vec_create(LM_D_MODEL);
	ln->beta = vec_create(LM_D_MODEL);
	ln->dgamma = vec_create(LM_D_MODEL);
	ln->dbeta = vec_create(LM_D_MODEL);
	if (!ln->gamma || !ln->beta || !ln->dgamma || !ln->dbeta) {
		layernorm_free(&ln);
		return NULL;
	}
	vec_fill(ln->gamma, 1.0);
	vec_fill(ln->beta, 0.0);
	vec_zero(ln->dgamma);
	vec_zero(ln->dbeta);
	ln->eps = 1e-5;
	return ln;
}

void layernorm_free(layernorm **ln)
{
	if (!ln || !*ln)
		return;
	vec_free(&(*ln)->gamma);
	vec_free(&(*ln)->beta);
	vec_free(&(*ln)->dgamma);
	vec_free(&(*ln)->dbeta);
	free(*ln);
	*ln = NULL;
}

void layernorm_zero_grad(layernorm *ln)
{
	if (!ln)
		return;
	vec_zero(ln->dgamma);
	vec_zero(ln->dbeta);
}

ln_cache *ln_cache_create(int T)
{
	ln_cache *c;

	if (T <= 0)
		return NULL;
	c = malloc(sizeof(ln_cache));
	if (!c)
		return NULL;
	c->T = T;
	c->x_hat = mat_create_dev(T, LM_D_MODEL);
	c->mean = vec_create(T);
	c->inv_std = vec_create(T);
	if (!c->x_hat || !c->mean || !c->inv_std) {
		ln_cache_free(&c);
		return NULL;
	}
	mat_try_pin(c->x_hat);
	return c;
}

void ln_cache_free(ln_cache **c)
{
	if (!c || !*c)
		return;
	mat_free(&(*c)->x_hat);
	vec_free(&(*c)->mean);
	vec_free(&(*c)->inv_std);
	free(*c);
	*c = NULL;
}

int layernorm_forward(mat *x, const layernorm *ln)
{
	ln_cache *c;
	int rc;

	if (!x || !ln)
		return -1;
	c = ln_cache_create(x->rows);
	if (!c)
		return -1;
	rc = layernorm_forward_cache(x, (layernorm *)ln, c);
	ln_cache_free(&c);
	return rc;
}

int layernorm_forward_into(mat *out, const mat *in, layernorm *ln, ln_cache *c)
{
	int T;
	int d;
	ann_real *yp;
	ann_real *xp;
	ann_real *xh;

	if (!out || !in || !ln || !c)
		return -1;
	if (out->rows != in->rows || out->cols != in->cols
	    || in->cols != LM_D_MODEL || c->T != in->rows)
		return -1;
	T = in->rows;
	d = in->cols;
	mat_try_pin((mat *)in);
	mat_try_pin(out);
	mat_try_pin(c->x_hat);
	xp = mat_compute_ptr((mat *)in);
	yp = mat_compute_ptr(out);
	xh = mat_compute_ptr(c->x_hat);
	if (xp && yp && xh && gpu_layernorm_forward_ex(yp, xp, ln->gamma->data,
		ln->beta->data, xh, c->mean->data, c->inv_std->data, T, d,
		ln->eps) == 0) {
		mat_mark_device(out);
		mat_mark_device(c->x_hat);
		return 0;
	}
	if (out != in && resid_copy(out, in) != 0)
		return -1;
	return layernorm_forward_cache(out, ln, c);
}

int layernorm_forward_cache(mat *x, layernorm *ln, ln_cache *c)
{
	int t;
	int j;
	int T;
	int d;
	double mean;
	double var;
	double inv_std;
	double v;
	ann_real *xp;
	ann_real *xh;

	if (!x || !ln || !c)
		return -1;
	if (x->cols != LM_D_MODEL || c->T != x->rows)
		return -1;

	T = x->rows;
	d = x->cols;
	mat_try_pin(x);
	mat_try_pin(c->x_hat);
	xp = mat_compute_ptr(x);
	xh = mat_compute_ptr(c->x_hat);
	if (xp && xh && gpu_layernorm_forward_ex(xp, xp, ln->gamma->data,
		ln->beta->data, xh, c->mean->data, c->inv_std->data, T, d,
		ln->eps) == 0) {
		mat_mark_device(x);
		mat_mark_device(c->x_hat);
		return 0;
	}
	(void)mat_ensure_host(x);
	for (t = 0; t < T; t++) {
		mean = 0.0;
		for (j = 0; j < d; j++)
			mean += mat_get(x, t, j);
		mean /= (double)d;
		c->mean->data[t] = mean;

		var = 0.0;
		for (j = 0; j < d; j++) {
			v = mat_get(x, t, j) - mean;
			var += v * v;
		}
		var /= (double)d;
		inv_std = 1.0 / sqrt(var + ln->eps);
		c->inv_std->data[t] = inv_std;

		for (j = 0; j < d; j++) {
			v = (mat_get(x, t, j) - mean) * inv_std;
			mat_set(c->x_hat, t, j, v);
			mat_set(x, t, j, v * ln->gamma->data[j] + ln->beta->data[j]);
		}
	}
	return 0;
}

int layernorm_backward(mat *dx, const mat *dy, layernorm *ln, const ln_cache *c)
{
	int t;
	int j;
	int d;
	double inv_std;
	double sum_dy;
	double sum_dy_xhat;
	double dxhat;
	double term;
	ann_real *dxp;
	const ann_real *dyp;
	const ann_real *xhp;

	if (!dx || !dy || !ln || !c)
		return -1;
	if (dx->rows != dy->rows || dx->cols != dy->cols)
		return -1;
	d = LM_D_MODEL;
	mat_try_pin(dx);
	mat_try_pin((mat *)dy);
	mat_try_pin(c->x_hat);
	dxp = mat_compute_ptr(dx);
	dyp = mat_compute_ptr_const(dy);
	xhp = mat_compute_ptr_const(c->x_hat);
	if (dxp && dyp && xhp
	    && gpu_layernorm_backward(dx, dy, ln->gamma->data, xhp,
		c->inv_std->data, c->T, d, ln->dgamma->data,
		ln->dbeta->data) == 0) {
		mat_mark_device(dx);
		return 0;
	}
	if (mat_ensure_host((mat *)dy) != 0)
		return -1;
	(void)mat_ensure_host(c->x_hat);
	mat_host_dirty(dx);
	for (t = 0; t < c->T; t++) {
		inv_std = c->inv_std->data[t];
		sum_dy = 0.0;
		sum_dy_xhat = 0.0;
		for (j = 0; j < d; j++) {
			ln->dbeta->data[j] += mat_get(dy, t, j);
			ln->dgamma->data[j] += mat_get(dy, t, j) * mat_get(c->x_hat, t, j);
			dxhat = mat_get(dy, t, j) * ln->gamma->data[j];
			sum_dy += dxhat;
			sum_dy_xhat += dxhat * mat_get(c->x_hat, t, j);
		}
		for (j = 0; j < d; j++) {
			dxhat = mat_get(dy, t, j) * ln->gamma->data[j];
			term = (dxhat - (sum_dy + mat_get(c->x_hat, t, j) * sum_dy_xhat) / (double)d)
			    * inv_std;
			mat_set(dx, t, j, term);
		}
	}
	return 0;
}
