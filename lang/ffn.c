/*
**	Position-wise feed-forward block + backward
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ffn.h"
#include "lm_config.h"
#include "gpu.h"
#include "gradutil.h"
#include "resid.h"

static double ffn_rand_weight(void)
{
	double u;

	u = (double)rand() / (double)RAND_MAX;
	return (u * 2.0 - 1.0) * (0.02 * sqrt(512.0 / (double)LM_D_MODEL));
}

static double ffn_relu(double x)
{
	if (x > 0.0)
		return x;
	return 0.0;
}

static void copy_mat(mat *dst, const mat *src)
{
	if (resid_copy(dst, src) == 0)
		return;
	if (mat_ensure_host((mat *)src) != 0 || mat_ensure_host(dst) != 0)
		return;
	memcpy(dst->data, src->data,
	    (size_t)dst->rows * (size_t)dst->cols * sizeof(ann_real));
	mat_host_dirty(dst);
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
	/* G1: persistent device twins for weights and grads (no-op on CPU). */
	mat_try_pin(f->w1);
	mat_try_pin(f->w2);
	mat_try_pin(f->dw1);
	mat_try_pin(f->dw2);
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
	c->x_in = mat_create_dev(T, LM_D_MODEL);
	c->pre_act = mat_create_dev(T, LM_D_FF);
	c->hidden = mat_create_dev(T, LM_D_FF);
	if (!c->x_in || !c->pre_act || !c->hidden) {
		ffn_cache_free(&c);
		return NULL;
	}
	mat_try_pin(c->x_in);
	mat_try_pin(c->pre_act);
	mat_try_pin(c->hidden);
	return c;
}

void ffn_cache_free(ffn_cache **c)
{
	if (!c || !*c)
		return;
	if (!(*c)->x_in_alias)
		mat_free(&(*c)->x_in);
	mat_free(&(*c)->pre_act);
	mat_free(&(*c)->hidden);
	free(*c);
	*c = NULL;
}

int ffn_forward_cache(mat *out, const mat *x, ffn *f, ffn_cache *c)
{
	return ffn_forward_cache_res(out, x, f, c, NULL);
}

/* out = res + FFN(x) with the residual fused into the W2 GEMM (res may be NULL). */
int ffn_forward_cache_res(mat *out, const mat *x, ffn *f, ffn_cache *c,
	const mat *res)
{
	int t;
	int j;
	int T;
	ann_real *pre;
	ann_real *hid;
	ann_real *op;

	if (!out || !x || !f || !c)
		return -1;
	if (x->cols != LM_D_MODEL || c->T != x->rows)
		return -1;
	T = x->rows;
	/* Alias the block's LN output (see attn_mha_forward_packed). */
	if (!c->x_in_alias)
		mat_free(&c->x_in);
	c->x_in = (mat *)x;
	c->x_in_alias = 1;
	mat_try_pin(c->pre_act);
	mat_try_pin(c->hidden);
	mat_try_pin(out);
	/*
	** Fast path: Lt RELU_BIAS into hidden (one GEMM). ReLU mask for bwd is
	** hidden itself (hidden>0 ⟺ pre>0). Skip when --fp8 so W1 hits E4M3.
	*/
	if (!gpu_fp8()
	    && gpu_mat_mul_bias_relu(c->hidden, x, f->w1, f->b1->data) == 0) {
		mat_mark_device(c->hidden);
	} else {
		int bias_fused;

		bias_fused = (gpu_mat_mul_bias(c->pre_act, x, f->w1,
		    f->b1->data) == 0);
		if (!bias_fused && mat_mul(c->pre_act, x, f->w1) != 0)
			return -1;
		pre = mat_compute_ptr(c->pre_act);
		hid = mat_compute_ptr(c->hidden);
		if (pre && hid
		    && gpu_ffn_w1_finish(hid, pre,
		    bias_fused ? NULL : f->b1->data, T, LM_D_FF) == 0) {
			mat_mark_written(c->pre_act, pre);
			mat_mark_written(c->hidden, hid);
		} else if (!bias_fused && pre
		    && resid_copy(c->hidden, c->pre_act) == 0
		    && (hid = mat_compute_ptr(c->hidden))
		    && gpu_bias_relu_forward(hid, f->b1->data, T, LM_D_FF)
		    == 0) {
			mat_mark_written(c->hidden, hid);
			if (gpu_bias_row_add(pre, f->b1->data, T, LM_D_FF)
			    == 0)
				mat_mark_written(c->pre_act, pre);
		} else if (!bias_fused && pre
		    && gpu_bias_row_add(pre, f->b1->data, T, LM_D_FF) == 0) {
			mat_mark_written(c->pre_act, pre);
			if (resid_copy(c->hidden, c->pre_act) != 0)
				copy_mat(c->hidden, c->pre_act);
			hid = mat_compute_ptr(c->hidden);
			if (hid && gpu_relu_forward(hid, T * LM_D_FF) == 0)
				mat_mark_written(c->hidden, hid);
			else {
				(void)mat_ensure_host(c->hidden);
				for (t = 0; t < T; t++)
					for (j = 0; j < LM_D_FF; j++)
						mat_set(c->hidden, t, j,
						    ffn_relu(mat_get(
						    c->hidden, t, j)));
			}
		} else if (!bias_fused) {
			(void)mat_ensure_host(c->pre_act);
			for (t = 0; t < T; t++) {
				for (j = 0; j < LM_D_FF; j++) {
					mat_set(c->pre_act, t, j,
					    mat_get(c->pre_act, t, j)
					    + f->b1->data[j]);
					mat_set(c->hidden, t, j,
					    ffn_relu(mat_get(c->pre_act, t,
					    j)));
				}
			}
		} else
			return -1;
	}
	if (res && !gpu_fp8()
	    && gpu_mat_mul_res(out, c->hidden, f->w2, f->b2->data, res) == 0)
		return 0;
	if (!gpu_fp8()
	    && gpu_mat_mul_bias(out, c->hidden, f->w2, f->b2->data) == 0)
		return res ? resid_add(out, res) : 0;
	if (mat_mul(out, c->hidden, f->w2) != 0)
		return -1;
	op = mat_compute_ptr(out);
	if (op && gpu_bias_row_add(op, f->b2->data, T, LM_D_MODEL) == 0)
		mat_mark_written(out, op);
	else {
		(void)mat_ensure_host(out);
		for (t = 0; t < T; t++) {
			for (j = 0; j < LM_D_MODEL; j++)
				mat_set(out, t, j,
				    mat_get(out, t, j) + f->b2->data[j]);
		}
	}
	return res ? resid_add(out, res) : 0;
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
	static mat *s_dh;
	static int s_dh_T;
	mat *dh;
	int t;
	int j;
	int T;
	ann_real *dhp;
	const ann_real *prep;

	if (!dx || !dout || !f || !c)
		return -1;
	T = c->T;
	/* One shared dh across blocks (bwd is serial); avoids per-step malloc. */
	if (!s_dh || s_dh_T != T) {
		mat_free(&s_dh);
		s_dh = mat_create_dev(T, LM_D_FF);
		s_dh_T = s_dh ? T : 0;
		if (s_dh)
			mat_try_pin(s_dh);
	}
	dh = s_dh;
	if (!dh)
		return -1;
	mat_try_pin(dx);

	if (mat_outer_add(f->dw2, c->hidden, dout) != 0)
		return -1;
	/* S3: column-sum db2 without forcing dout D2H when device-resident. */
	{
		const ann_real *doutp = mat_compute_ptr_const(dout);

		if (!doutp
		    || gpu_mat_col_sum_add(f->db2->data, doutp, T, LM_D_MODEL)
		    != 0) {
			(void)mat_ensure_host((mat *)dout);
			for (t = 0; t < T; t++) {
				for (j = 0; j < LM_D_MODEL; j++)
					f->db2->data[j] += dout->data[
					    t * LM_D_MODEL + j];
			}
		}
	}
	/* dh = dout * W2^T then ReLU mask (GEMM may use GPU). */
	if (mat_mul_wt(dh, dout, f->w2) != 0)
		return -1;
	dhp = mat_compute_ptr(dh);
	/* hidden>0 ⟺ pre>0 after ReLU; prefer hidden (Lt RELU_BIAS path). */
	prep = mat_compute_ptr_const(c->hidden);
	if (!prep)
		prep = mat_compute_ptr_const(c->pre_act);
	if (!dhp || !prep
	    || gpu_relu_backward(dhp, prep, T * LM_D_FF) != 0) {
		const mat *mask = c->hidden->data ? c->hidden : c->pre_act;

		(void)mat_ensure_host(dh);
		(void)mat_ensure_host((mat *)mask);
		for (t = 0; t < T; t++) {
			for (j = 0; j < LM_D_FF; j++) {
				if (mask->data[t * LM_D_FF + j] <= 0.0)
					dh->data[t * LM_D_FF + j] = 0.0;
			}
		}
		mat_host_dirty(dh);
	} else
		mat_mark_written(dh, dhp);
	if (mat_outer_add(f->dw1, c->x_in, dh) != 0)
		return -1;
	{
		ann_real *dhptr = mat_compute_ptr(dh);

		if (!dhptr
		    || gpu_mat_col_sum_add(f->db1->data, dhptr, T, LM_D_FF)
		    != 0) {
			(void)mat_ensure_host(dh);
			for (t = 0; t < T; t++) {
				for (j = 0; j < LM_D_FF; j++)
					f->db1->data[j] += dh->data[
					    t * LM_D_FF + j];
			}
		}
	}
	if (mat_mul_wt(dx, dh, f->w1) != 0)
		return -1;
	return 0;
}
