/*
**	Attention (QKV + scaled dot-product + multi-head + causal + backward)
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "attn.h"
#include "gpu.h"
#include "gradutil.h"
#include "ops.h"
#include "resid.h"
#include "vec.h"

static double rand_w(void)
{
	/* Scale noise with width (0.02 at D=512); D=1024 needs smaller init. */
	double s = 0.02 * sqrt(512.0 / (double)LM_D_MODEL);

	return ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * s;
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
	if (resid_copy(dst, src) == 0)
		return;
	if (mat_ensure_host((mat *)src) != 0 || mat_ensure_host(dst) != 0)
		return;
	memcpy(dst->data, src->data,
	    (size_t)dst->rows * (size_t)dst->cols * sizeof(ann_real));
	mat_host_dirty(dst);
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
	/* G1: persistent device twins for weights and grads (no-op on CPU). */
	mat_try_pin(a->wq);
	mat_try_pin(a->wk);
	mat_try_pin(a->wv);
	mat_try_pin(a->wo);
	mat_try_pin(a->dwq);
	mat_try_pin(a->dwk);
	mat_try_pin(a->dwv);
	mat_try_pin(a->dwo);
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

/* lean != 0: score/head scratch only (packed per-sequence caches). */
static attn_cache *attn_cache_alloc(int T, int n_heads, int d_head, int lean)
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
	if (!lean) {
		c->x_in = mat_create(T, LM_D_MODEL);
		c->q = mat_create(T, LM_D_MODEL);
		c->k = mat_create(T, LM_D_MODEL);
		c->v = mat_create(T, LM_D_MODEL);
		c->concat = mat_create(T, LM_D_MODEL);
	}
	c->q_h = calloc((size_t)n_heads, sizeof(mat *));
	c->k_h = calloc((size_t)n_heads, sizeof(mat *));
	c->o_h = calloc((size_t)n_heads, sizeof(mat *));
	c->softmax_tmp = calloc((size_t)n_heads * (size_t)T, sizeof(ann_real));
	c->attn_w = calloc((size_t)n_heads, sizeof(mat *));
	c->v_h = calloc((size_t)n_heads, sizeof(mat *));
	if ((!lean && (!c->x_in || !c->q || !c->k || !c->v || !c->concat))
	    || !c->q_h || !c->k_h || !c->o_h || !c->softmax_tmp
	    || !c->attn_w || !c->v_h) {
		attn_cache_free(&c);
		return NULL;
	}
	/* Pin activation mats on GPU when available (no-op on CPU stub). */
	if (!lean) {
		mat_try_pin(c->x_in);
		mat_try_pin(c->q);
		mat_try_pin(c->k);
		mat_try_pin(c->v);
		mat_try_pin(c->concat);
	}
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

attn_cache *attn_cache_create(int T, int n_heads, int d_head)
{
	return attn_cache_alloc(T, n_heads, d_head, 0);
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
	/*
	** Fused X@[Wq|Wk|Wv] when CUDA (one GemmEx). Falls back to three
	** mat_muls if pack/GEMM/split fails.
	*/
	if (gpu_available()
	    && gpu_attn_qkv_fused(q, k, v, x, a->wq, a->wk, a->wv) == 0)
		return 0;
	if (mat_mul(q, x, a->wq) != 0)
		return -1;
	if (mat_mul(k, x, a->wk) != 0)
		return -1;
	if (mat_mul(v, x, a->wv) != 0)
		return -1;
	return 0;
}

static int softmax_row(ann_real *row, int n, ann_real *tmp)
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

/* Softmax over every row of a T x T score block: GPU first, CPU fallback. */
static int softmax_rows_inplace(ann_real *rows, int T, ann_real *tmp)
{
	int i;

	if (gpu_softmax_rows(rows, T, T) == 0)
		return 0;
	for (i = 0; i < T; i++) {
		if (softmax_row(&rows[i * T], T, tmp) != 0)
			return -1;
	}
	return 0;
}

int attn_self(mat *out, const mat *q, const mat *k, const mat *v, int causal)
{
	int T;
	int i;
	int j;
	double scale;
	mat *scores;
	ann_real *tmp;

	if (!out || !q || !k || !v || !out->data || !q->data || !k->data || !v->data)
		return -1;
	if (q->rows != k->rows || q->rows != v->rows || q->cols != k->cols || q->cols != v->cols)
		return -1;
	if (out->rows != q->rows || out->cols != v->cols)
		return -1;
	mat_ensure_host((mat *)q);
	mat_ensure_host((mat *)k);
	mat_ensure_host((mat *)v);
	T = q->rows;
	scale = 1.0 / sqrt((double)q->cols);
	scores = mat_create(T, T);
	tmp = calloc((size_t)T, sizeof(ann_real));
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
	(void)mat_ensure_host(scores);
	for (i = 0; i < T; i++) {
		for (j = 0; j < T; j++) {
			scores->data[i * T + j] *= scale;
			if (causal && j > i)
				scores->data[i * T + j] = -1e9;
		}
	}
	if (softmax_rows_inplace(scores->data, T, tmp) != 0) {
		mat_free(&scores);
		free(tmp);
		return -1;
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
		    (size_t)d_head * sizeof(ann_real));
	dst->on_device = MAT_DEV_HOST;	/* host write (src already host-valid) */
}

static void write_head(mat *dst, const mat *src, int head, int d_head)
{
	int t;
	int off;

	off = head * d_head;
	for (t = 0; t < src->rows; t++)
		memcpy(&dst->data[t * dst->cols + off],
		    &src->data[t * src->cols],
		    (size_t)d_head * sizeof(ann_real));
	dst->on_device = MAT_DEV_HOST;	/* host write: twin is stale */
}

static int self_forward_cache(mat *out, const mat *q, const mat *k, const mat *v,
	int causal, mat *attn_w, ann_real *softmax_tmp)
{
	int T;
	int d;
	int i;
	int j;
	double scale;
	ann_real *row;

	if (!softmax_tmp || !out || !q || !k || !v || !attn_w)
		return -1;
	T = q->rows;
	d = q->cols;
	scale = 1.0 / sqrt((double)d);
	if (mat_mul_bt(attn_w, q, k) != 0)
		return -1;
	(void)mat_ensure_host(attn_w);
	for (i = 0; i < T; i++) {
		row = &attn_w->data[i * T];
		for (j = 0; j < T; j++) {
			row[j] *= scale;
			if (causal && j > i)
				row[j] = -1e9;
		}
	}
	if (softmax_rows_inplace(attn_w->data, T, softmax_tmp) != 0)
		return -1;
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
	/*
	** BATCH mode leaves q/k/v device-authoritative; the per-head softmax
	** path below is host code, so pull them down once (before the OpenMP
	** region so threads never race on a D2H).
	*/
	if (mat_ensure_host(c->q) != 0 || mat_ensure_host(c->k) != 0
	    || mat_ensure_host(c->v) != 0)
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
	}
	if (fail)
		return -1;
	/* BATCH leaves o_h on the device; D2H outside the OpenMP region. */
	for (h = 0; h < a->n_heads; h++) {
		if (mat_ensure_host(c->o_h[h]) != 0)
			return -1;
		write_head(c->concat, c->o_h[h], h, a->d_head);
	}
	/* Host wrote concat (mat_fill may have marked its twin current). */
	mat_host_dirty(c->concat);
	if (mat_mul(out, c->concat, a->wo) != 0)
		return -1;
	return 0;
}

static void softmax_bwd_row(const ann_real *y, const ann_real *dy, ann_real *dx, int n)
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
	mat *tmp;
	mat *k_h;
	mat *q_h;
	ann_real *drow;
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
	tmp = mat_create(T, LM_D_MODEL);
	k_h = mat_create(T, a->d_head);
	q_h = mat_create(T, a->d_head);
	drow = malloc((size_t)T * sizeof(ann_real));
	if (!dconcat || !dq || !dk || !dv || !dq_h || !dk_h || !dv_h || !do_h
	    || !dscores || !tmp || !k_h || !q_h || !drow)
		goto fail;

	/* Weight grads via outer_add (GPU when available). */
	if (mat_outer_add(a->dwo, c->concat, dout) != 0)
		goto fail;
	/* dconcat = dout * Wo^T  (GEMM / GPU). */
	if (mat_mul_wt(dconcat, dout, a->wo) != 0)
		goto fail;

	mat_zero(dq);
	mat_zero(dk);
	mat_zero(dv);
	/* Host copy_head reads below need host-valid q/k. */
	if (mat_ensure_host(c->k) != 0 || mat_ensure_host(c->q) != 0)
		goto fail;
	for (h = 0; h < a->n_heads; h++) {
		copy_head(do_h, dconcat, h, a->d_head);
		copy_head(k_h, c->k, h, a->d_head);
		copy_head(q_h, c->q, h, a->d_head);
		/* dV = A^T do ; dA_scores = do V^T  (GEMM). */
		if (mat_mul_at(dv_h, c->attn_w[h], do_h) != 0)
			goto fail;
		if (mat_mul_bt(dscores, do_h, c->v_h[h]) != 0)
			goto fail;
		(void)mat_ensure_host(dscores);
		if (gpu_softmax_bwd_rows(dscores->data, c->attn_w[h]->data,
			dscores->data, T, T) == 0) {
			/* In place on device: only causal mask + scale left. */
			for (i = 0; i < T; i++) {
				for (j = 0; j < T; j++) {
					if (c->causal && j > i)
						dscores->data[i * T + j] = 0.0;
					else
						dscores->data[i * T + j] *= scale;
				}
			}
		} else {
			for (i = 0; i < T; i++) {
				softmax_bwd_row(&c->attn_w[h]->data[i * T],
				    &dscores->data[i * T], drow, T);
				for (j = 0; j < T; j++) {
					if (c->causal && j > i)
						drow[j] = 0.0;
					dscores->data[i * T + j] = drow[j] * scale;
				}
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
	if (mat_mul_wt(dx, dq, a->wq) != 0)
		goto fail;
	if (mat_mul_wt_add(dx, dk, a->wk, tmp) != 0)
		goto fail;
	if (mat_mul_wt_add(dx, dv, a->wv, tmp) != 0)
		goto fail;

	mat_free(&dconcat);
	mat_free(&dq);
	mat_free(&dk);
	mat_free(&dv);
	mat_free(&dq_h);
	mat_free(&dk_h);
	mat_free(&dv_h);
	mat_free(&do_h);
	mat_free(&dscores);
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
	mat_free(&tmp);
	mat_free(&k_h);
	mat_free(&q_h);
	free(drow);
	return -1;
}

/*
** ---- Token-parallel packed attention ---------------------------------
** Rows [b*T, (b+1)*T) of every (B*T) x D matrix belong to sequence b.
*/

static int g_flash_attn;

void attn_set_flash(int on)
{
	g_flash_attn = on ? 1 : 0;
}

int attn_flash_enabled(void)
{
	return g_flash_attn;
}

attn_pcache *attn_pcache_create(int B, int T, int n_heads, int d_head)
{
	attn_pcache *c;
	int b;
	int n;

	if (B <= 0 || T <= 0 || n_heads <= 0 || d_head <= 0)
		return NULL;
	c = malloc(sizeof(attn_pcache));
	if (!c)
		return NULL;
	memset(c, 0, sizeof(*c));
	n = B * T;
	c->B = B;
	c->T = T;
	c->n = n;
	c->n_heads = n_heads;
	c->d_head = d_head;
	c->x_in = mat_create_dev(n, LM_D_MODEL);
	c->q = mat_create_dev(n, LM_D_MODEL);
	c->k = mat_create_dev(n, LM_D_MODEL);
	c->v = mat_create_dev(n, LM_D_MODEL);
	c->concat = mat_create_dev(n, LM_D_MODEL);
	c->seq = calloc((size_t)B, sizeof(attn_cache *));
	if (!c->x_in || !c->q || !c->k || !c->v || !c->concat || !c->seq) {
		attn_pcache_free(&c);
		return NULL;
	}
	mat_try_pin(c->x_in);
	mat_try_pin(c->q);
	mat_try_pin(c->k);
	mat_try_pin(c->v);
	mat_try_pin(c->concat);
	/*
	** Per-layer causal logsumexp (B·H·T), not a shared global. The tiled
	** backward of this block must use the softmax this forward wrote.
	*/
	c->lse = mat_create(B * n_heads * T, 1);
	if (c->lse) {
		mat_try_pin(c->lse);
		if (gpu_available() && !c->lse->d_data)
			mat_free(&c->lse);
	}
	/*
	** Score buffer (B·H·T)×T for TC GemmEx when it fits (≤256 MiB).
	** Prefer TC even under --flash-attn at moderate T (e.g. 64): tiled FA
	** saves VRAM for huge T but is slower than strided GemmEx here.
	** When scores do not fit, leave w_all NULL and use tiled FA.
	*/
	/* The fused WMMA path needs no score buffer: 67 MB per layer saved. */
	if (gpu_available() && T <= 256
	    && !(g_flash_attn && gpu_attn_fa_usable(T, d_head))) {
		size_t sb = (size_t)B * (size_t)n_heads * (size_t)T
		    * (size_t)T * sizeof(ann_real);

		if (sb <= (size_t)256 * 1024 * 1024) {
			c->w_all = mat_create_dev(B * n_heads * T, T);
			if (c->w_all && !c->w_all->d_data)
				mat_free(&c->w_all);
		}
	}
	/* Host per-sequence caches only when no GPU score buffer / flash. */
	if (!c->w_all && !g_flash_attn) {
		for (b = 0; b < B; b++) {
			c->seq[b] = attn_cache_alloc(T, n_heads, d_head, 1);
			if (!c->seq[b]) {
				attn_pcache_free(&c);
				return NULL;
			}
			c->seq[b]->causal = 1;
		}
	}
	return c;
}

/* Fallback when a fused launch fails after seq[] was skipped at create. */
static int pcache_ensure_seq(attn_pcache *c)
{
	int b;

	for (b = 0; b < c->B; b++) {
		if (c->seq[b])
			continue;
		c->seq[b] = attn_cache_alloc(c->T, c->n_heads, c->d_head, 1);
		if (!c->seq[b])
			return -1;
		c->seq[b]->causal = 1;
	}
	return 0;
}

static int pcache_ensure_scratch(attn_pcache *c)
{
	if (!c->dq) {
		c->dq = mat_create_dev(c->n, LM_D_MODEL);
		c->dk = mat_create_dev(c->n, LM_D_MODEL);
		c->dv = mat_create_dev(c->n, LM_D_MODEL);
		c->dconcat = mat_create_dev(c->n, LM_D_MODEL);
		c->dtmp = mat_create_dev(c->n, LM_D_MODEL);
		if (c->dq && c->dk && c->dv && c->dconcat && c->dtmp) {
			mat_try_pin(c->dq);
			mat_try_pin(c->dk);
			mat_try_pin(c->dv);
			mat_try_pin(c->dconcat);
			mat_try_pin(c->dtmp);
		}
	}
	return (c->dq && c->dk && c->dv && c->dconcat && c->dtmp) ? 0 : -1;
}

void attn_pcache_free(attn_pcache **c)
{
	int b;

	if (!c || !*c)
		return;
	if (!(*c)->x_in_alias)
		mat_free(&(*c)->x_in);
	mat_free(&(*c)->q);
	mat_free(&(*c)->k);
	mat_free(&(*c)->v);
	mat_free(&(*c)->concat);
	mat_free(&(*c)->w_all);
	mat_free(&(*c)->lse);
	mat_free(&(*c)->dq);
	mat_free(&(*c)->dk);
	mat_free(&(*c)->dv);
	mat_free(&(*c)->dconcat);
	mat_free(&(*c)->dtmp);
	if ((*c)->seq) {
		for (b = 0; b < (*c)->B; b++)
			attn_cache_free(&(*c)->seq[b]);
		free((*c)->seq);
	}
	free(*c);
	*c = NULL;
}

/* dst (T x d_head) <- columns of head h from rows [row0, row0+T) of src. */
static void copy_head_rows(mat *dst, const mat *src, int row0, int head,
	int d_head)
{
	int t;

	for (t = 0; t < dst->rows; t++)
		memcpy(&dst->data[t * dst->cols],
		    &src->data[(size_t)(row0 + t) * src->cols + head * d_head],
		    (size_t)d_head * sizeof(ann_real));
	dst->on_device = MAT_DEV_HOST;
}

static void write_head_rows(mat *dst, const mat *src, int row0, int head,
	int d_head)
{
	int t;

	for (t = 0; t < src->rows; t++)
		memcpy(&dst->data[(size_t)(row0 + t) * dst->cols + head * d_head],
		    &src->data[t * src->cols],
		    (size_t)d_head * sizeof(ann_real));
}

/* Output projection: out = concat Wo (+ res when given, fused on BF16). */
static int attn_project_out(mat *out, attn *a, attn_pcache *c, const mat *res)
{
	if (res && gpu_mat_mul_res(out, c->concat, a->wo, NULL, res) == 0)
		return 0;
	if (mat_mul(out, c->concat, a->wo) != 0)
		return -1;
	return res ? resid_add(out, res) : 0;
}

int attn_mha_forward_packed(mat *out, attn *a, const mat *x, attn_pcache *c)
{
	return attn_mha_forward_packed_res(out, a, x, c, NULL);
}

/*
** Packed forward with the block's residual folded into the output
** projection: out = res + Attn(x) Wo (res may be NULL).
*/
int attn_mha_forward_packed_res(mat *out, attn *a, const mat *x,
	attn_pcache *c, const mat *res)
{
	int i;
	int T;
	int fail;
	int nh;

	if (!out || !a || !x || !c || x->rows != c->n || out->rows != c->n
	    || c->n_heads != a->n_heads || c->d_head != a->d_head)
		return -1;
	T = c->T;
	nh = a->n_heads;
	/*
	** x is the block's LayerNorm output, which lives in the block cache
	** until the backward: keep a pointer instead of a 16 MB copy, so the
	** weight gradients also reuse the BF16 twin the LayerNorm published.
	*/
	if (!c->x_in_alias)
		mat_free(&c->x_in);
	c->x_in = (mat *)x;
	c->x_in_alias = 1;
	/* Fat projections: one (B*T) x D GEMM each. */
	if (attn_qkv_project(c->q, c->k, c->v, a, x) != 0)
		return -1;
	/*
	** Prefer TC GemmEx when w_all fits (faster at T=64 meter; FA ~3×
	** slower here). Else tiled FA under --flash-attn (no T×T HBM). Then
	** legacy / host.
	*/
	c->fused = 0;
	/*
	** Fused tensor-core attention (no T x T scores in HBM): BF16 mode,
	** T % 64 == 0, no seqlens; the lse it writes feeds its backward.
	*/
	if (g_flash_attn && c->lse
	    && gpu_attn_fa_fwd(c->concat, c->q, c->k, c->v, c->B, T, nh,
	    a->d_head, c->lse) == 0) {
		static int s_fa_wmma_once;

		c->fused = 5;
		if (!s_fa_wmma_once) {
			s_fa_wmma_once = 1;
			fprintf(stderr, "[attn] fused tensor-core attention (WMMA)\n");
			fflush(stderr);
		}
		return attn_project_out(out, a, c, res);
	}
	if (c->w_all
	    && gpu_attn_tc_fwd(c->concat, c->w_all, c->q, c->k, c->v, c->B, T,
	    nh, a->d_head) == 0) {
		static int s_tc_once;

		c->fused = 3;
		if (!s_tc_once) {
			s_tc_once = 1;
			fprintf(stderr, "[attn] Tensor-Core strided QK/AV path\n");
			fflush(stderr);
		}
		return attn_project_out(out, a, c, res);
	}
	if (g_flash_attn
	    && gpu_attn_flash_fwd(c->concat, c->q, c->k, c->v, c->B, T, nh,
	    a->d_head, c->lse) == 0) {
		static int s_fa_once;

		c->fused = 4;
		if (!s_fa_once) {
			s_fa_once = 1;
			fprintf(stderr,
			    "[attn] tiled flash (logsumexp, no T×T HBM)\n");
			fflush(stderr);
		}
		return attn_project_out(out, a, c, res);
	}
	if (c->w_all && gpu_attn_causal_fwd(c->concat, c->w_all, c->q, c->k,
	    c->v, c->B, T, nh, a->d_head) == 0) {
		c->fused = 1;
		return attn_project_out(out, a, c, res);
	}
	if (pcache_ensure_seq(c) != 0)
		return -1;
	if (mat_ensure_host(c->q) != 0 || mat_ensure_host(c->k) != 0
	    || mat_ensure_host(c->v) != 0)
		return -1;
	fail = 0;
	/* Causal attention only inside each sequence: B*H independent items. */
#ifdef _OPENMP
#pragma omp parallel for reduction(|:fail)
#endif
	for (i = 0; i < c->B * nh; i++) {
		attn_cache *s;
		int b;
		int h;

		b = i / nh;
		h = i % nh;
		s = c->seq[b];
		copy_head_rows(s->q_h[h], c->q, b * T, h, a->d_head);
		copy_head_rows(s->k_h[h], c->k, b * T, h, a->d_head);
		copy_head_rows(s->v_h[h], c->v, b * T, h, a->d_head);
		if (self_forward_cache(s->o_h[h], s->q_h[h], s->k_h[h],
		    s->v_h[h], 1, s->attn_w[h], &s->softmax_tmp[h * T]) != 0)
			fail = 1;
		else
			write_head_rows(c->concat, s->o_h[h], b * T, h,
			    a->d_head);
	}
	if (fail)
		return -1;
	mat_host_dirty(c->concat);
	return attn_project_out(out, a, c, res);
}

int attn_mha_backward_packed(mat *dx, const mat *dout, attn *a,
	const attn_pcache *c_in)
{
	attn_pcache *c;
	mat *dconcat;
	mat *dq;
	mat *dk;
	mat *dv;
	mat *tmp;
	int T;
	int nh;
	int fail;
	double scale;

	if (!dx || !dout || !a || !c_in || dout->rows != c_in->n
	    || dx->rows != c_in->n)
		return -1;
	c = (attn_pcache *)c_in;	/* scratch mats are mutable cache state */
	T = c->T;
	nh = a->n_heads;
	scale = 1.0 / sqrt((double)a->d_head);
	if (pcache_ensure_scratch(c) != 0)
		return -1;
	dconcat = c->dconcat;
	dq = c->dq;
	dk = c->dk;
	dv = c->dv;
	tmp = c->dtmp;
	/* Fat weight grads + dconcat over all B*T rows. */
	if (mat_outer_add(a->dwo, c->concat, dout) != 0)
		return -1;
	if (mat_mul_wt(dconcat, dout, a->wo) != 0)
		return -1;
	if (c->fused == 5) {
		if (gpu_attn_fa_bwd(dq, dk, dv, dconcat, c->concat, c->q, c->k,
		    c->v, c->lse, c->B, T, nh, a->d_head) != 0)
			return -1;
		goto proj;
	}
	if (c->fused == 4) {
		if (gpu_attn_tiled_bwd(dq, dk, dv, dconcat, c->concat, c->q,
		    c->k, c->v, c->lse, c->B, T, nh, a->d_head) != 0)
			return -1;
		goto proj;
	}
	if (c->fused == 3) {
		if (!c->w_all
		    || gpu_attn_tc_bwd(dq, dk, dv, dconcat, c->w_all, c->q,
		    c->k, c->v, c->B, T, nh, a->d_head) != 0)
			return -1;
		goto proj;
	}
	if (c->fused == 2) {
		/*
		** Slow path used to fill a B·H·T·T score tensor
		** (gpu_attn_recompute_w) and then a second dS tensor.
		** Recompute only the row logsumexp of the same causal
		** softmax, then the tiled backward. T stays the train window.
		*/
		if (gpu_attn_recompute_w(c->lse, c->q, c->k, c->B, T, nh,
		    a->d_head) != 0
		    || gpu_attn_tiled_bwd(dq, dk, dv, dconcat, c->concat,
		    c->q, c->k, c->v, c->lse, c->B, T, nh, a->d_head) != 0)
			return -1;
		goto proj;
	}
	if (c->fused == 1) {
		if (gpu_attn_causal_bwd(dq, dk, dv, dconcat, c->w_all, c->q,
		    c->k, c->v, c->B, T, nh, a->d_head) != 0)
			return -1;
		goto proj;
	}
	(void)mat_ensure_host(dconcat);
	if (mat_ensure_host(c->k) != 0 || mat_ensure_host(c->q) != 0)
		return -1;
	mat_zero(dq);
	mat_zero(dk);
	mat_zero(dv);
	fail = 0;
#ifdef _OPENMP
#pragma omp parallel reduction(|:fail)
#endif
	{
		mat *dq_h;
		mat *dk_h;
		mat *dv_h;
		mat *do_h;
		mat *dscores;
		mat *k_h;
		mat *q_h;
		ann_real *drow;
		int i;
		int r;
		int cc;

		dq_h = mat_create(T, a->d_head);
		dk_h = mat_create(T, a->d_head);
		dv_h = mat_create(T, a->d_head);
		do_h = mat_create(T, a->d_head);
		dscores = mat_create(T, T);
		k_h = mat_create(T, a->d_head);
		q_h = mat_create(T, a->d_head);
		drow = malloc((size_t)T * sizeof(ann_real));
		if (!dq_h || !dk_h || !dv_h || !do_h || !dscores || !k_h || !q_h
		    || !drow)
			fail = 1;
#ifdef _OPENMP
#pragma omp for
#endif
		for (i = 0; i < c->B * nh; i++) {
			const attn_cache *s;
			int b;
			int h;

			if (fail)
				continue;
			b = i / nh;
			h = i % nh;
			s = c->seq[b];
			copy_head_rows(do_h, dconcat, b * T, h, a->d_head);
			copy_head_rows(k_h, c->k, b * T, h, a->d_head);
			copy_head_rows(q_h, c->q, b * T, h, a->d_head);
			if (mat_mul_at(dv_h, s->attn_w[h], do_h) != 0
			    || mat_mul_bt(dscores, do_h, s->v_h[h]) != 0) {
				fail = 1;
				continue;
			}
			(void)mat_ensure_host(dscores);
			if (gpu_softmax_bwd_rows(dscores->data,
			    s->attn_w[h]->data, dscores->data, T, T) == 0) {
				for (r = 0; r < T; r++)
					for (cc = 0; cc < T; cc++) {
						if (cc > r)
							dscores->data[r * T + cc] = 0.0;
						else
							dscores->data[r * T + cc] *= scale;
					}
			} else {
				for (r = 0; r < T; r++) {
					softmax_bwd_row(&s->attn_w[h]->data[r * T],
					    &dscores->data[r * T], drow, T);
					for (cc = 0; cc < T; cc++) {
						if (cc > r)
							drow[cc] = 0.0;
						dscores->data[r * T + cc] =
						    drow[cc] * scale;
					}
				}
			}
			mat_host_dirty(dscores);
			if (mat_mul(dq_h, dscores, k_h) != 0
			    || mat_mul_at(dk_h, dscores, q_h) != 0) {
				fail = 1;
				continue;
			}
			(void)mat_ensure_host(dq_h);
			(void)mat_ensure_host(dk_h);
			(void)mat_ensure_host(dv_h);
			write_head_rows(dq, dq_h, b * T, h, a->d_head);
			write_head_rows(dk, dk_h, b * T, h, a->d_head);
			write_head_rows(dv, dv_h, b * T, h, a->d_head);
		}
		mat_free(&dq_h);
		mat_free(&dk_h);
		mat_free(&dv_h);
		mat_free(&do_h);
		mat_free(&dscores);
		mat_free(&k_h);
		mat_free(&q_h);
		free(drow);
	}
	if (fail)
		return -1;
	mat_host_dirty(dq);
	mat_host_dirty(dk);
	mat_host_dirty(dv);
proj:
	if (mat_outer_add(a->dwq, c->x_in, dq) != 0
	    || mat_outer_add(a->dwk, c->x_in, dk) != 0
	    || mat_outer_add(a->dwv, c->x_in, dv) != 0)
		return -1;
	if (mat_mul_wt(dx, dq, a->wq) != 0)
		return -1;
	if (mat_mul_wt_add(dx, dk, a->wk, tmp) != 0)
		return -1;
	if (mat_mul_wt_add(dx, dv, a->wv, tmp) != 0)
		return -1;
	return 0;
}
