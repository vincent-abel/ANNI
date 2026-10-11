/*
**	Language model forward + backward through Transformer
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"
#include "gradutil.h"
#include "lm_adam.h"
#include "lm.h"
#include "ops.h"
#include "posenc.h"
#include "tokenize.h"

static double rand_w(void)
{
	return ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * 0.02;
}

static void copy_mat(mat *dst, const mat *src)
{
	memcpy(dst->data, src->data,
	    (size_t)dst->rows * (size_t)dst->cols * sizeof(ann_real));
}

static void softmax_inplace(ann_real *x, int n, ann_real *tmp)
{
	vec in;
	vec out;
	int i;

	if (!x || !tmp || n <= 0)
		return;
	in.data = x;
	in.n = n;
	out.data = tmp;
	out.n = n;
	if (vec_softmax(&out, &in) == 0) {
		for (i = 0; i < n; i++)
			x[i] = tmp[i];
	}
}

static lm *lm_create_with(vocab *v, bpe *tok)
{
	lm *model;
	int i;
	int n;

	if (!v)
		return NULL;
	model = malloc(sizeof(lm));
	if (!model) {
		vocab_free(&v);
		return NULL;
	}
	model->vocab = v;
	model->bpe = tok;
	model->vocab_size = vocab_size(model->vocab);
	model->embed = embed_create(model->vocab_size);
	model->tr = transformer_create();
	model->w_head = mat_create(LM_D_MODEL, model->vocab_size);
	model->b_head = vec_create(model->vocab_size);
	model->dw_head = mat_create(LM_D_MODEL, model->vocab_size);
	model->db_head = vec_create(model->vocab_size);
	model->w_head_q = NULL;
	if (!model->embed || !model->tr || !model->w_head || !model->b_head
	    || !model->dw_head || !model->db_head) {
		model->bpe = NULL;	/* caller keeps ownership on failure */
		lm_free(&model);
		return NULL;
	}
	n = model->w_head->rows * model->w_head->cols;
	for (i = 0; i < n; i++)
		model->w_head->data[i] = rand_w();
	vec_fill(model->b_head, 0.0);
	lm_zero_grad(model);
	/* G1: persistent device twins for head weights/grads (no-op on CPU). */
	mat_try_pin(model->w_head);
	mat_try_pin(model->dw_head);
	return model;
}

lm *lm_create(void)
{
	return lm_create_with(vocab_byte_create(), NULL);
}

lm *lm_create_bpe(bpe *tok)
{
	lm *model;

	if (!tok)
		return NULL;
	model = lm_create_with(bpe_vocab_create(tok), tok);
	if (!model)
		return NULL;
	return model;
}

int lm_attach_bpe(lm *model, bpe *tok)
{
	vocab *v;

	if (!model || !tok || tok->vocab_size != model->vocab_size)
		return -1;
	v = bpe_vocab_create(tok);
	if (!v)
		return -1;
	vocab_free(&model->vocab);
	bpe_free(&model->bpe);
	model->vocab = v;
	model->bpe = tok;
	return 0;
}

int lm_encode(const lm *model, const char *text, int *ids, int max_ids)
{
	if (model && model->bpe)
		return bpe_encode(model->bpe, text, ids, max_ids);
	return tokenize_encode(text, ids, max_ids);
}

int lm_decode(const lm *model, const int *ids, int n_ids, char *out, int out_cap)
{
	if (model && model->bpe)
		return bpe_decode(model->bpe, ids, n_ids, out, out_cap);
	return tokenize_decode(ids, n_ids, out, out_cap);
}

void lm_free(lm **model)
{
	if (!model || !*model)
		return;
	vocab_free(&(*model)->vocab);
	bpe_free(&(*model)->bpe);
	embed_free(&(*model)->embed);
	transformer_free(&(*model)->tr);
	mat_free(&(*model)->w_head);
	vec_free(&(*model)->b_head);
	mat_free(&(*model)->dw_head);
	vec_free(&(*model)->db_head);
	qmat_free(&(*model)->w_head_q);
	free(*model);
	*model = NULL;
}

int lm_quantize_head(lm *model)
{
	if (!model || !model->w_head)
		return -1;
	qmat_free(&model->w_head_q);
	(void)mat_pull_host(model->w_head);
	model->w_head_q = qmat_from_mat(model->w_head);
	return model->w_head_q ? 0 : -1;
}

void lm_zero_grad(lm *model)
{
	if (!model)
		return;
	embed_zero_grad(model->embed);
	transformer_zero_grad(model->tr);
	mat_zero(model->dw_head);
	vec_zero(model->db_head);
}

/*
** mode 0: upload weight twin; 1: weight host-dirty; 2: grad to host;
** 3: pull device-authoritative weight to host (state stays "synced");
** 4: zero grad on host + device twin (no H2D of zeros later).
*/
static void wgrad_visit(int mode, mat *w, mat *dw)
{
	if (mode == 0)
		(void)mat_sync_device(w);
	else if (mode == 1)
		mat_host_dirty(w);
	else if (mode == 2)
		(void)mat_ensure_host(dw);
	else if (mode == 3)
		(void)mat_pull_host(w);
	else
		mat_zero_dev(dw);
}

/* Visit every weight/grad mat pair that has a GPU consumer (GEMM/outer). */
static void lm_each_wgrad(lm *model, int mode)
{
	int i;
	tblock *b;

	if (!model)
		return;
	wgrad_visit(mode, model->w_head, model->dw_head);
	if (!model->tr)
		return;
	for (i = 0; i < model->tr->n_blocks; i++) {
		b = &model->tr->blocks[i];
		wgrad_visit(mode, b->attn->wq, b->attn->dwq);
		wgrad_visit(mode, b->attn->wk, b->attn->dwk);
		wgrad_visit(mode, b->attn->wv, b->attn->dwv);
		wgrad_visit(mode, b->attn->wo, b->attn->dwo);
		wgrad_visit(mode, b->ffn->w1, b->ffn->dw1);
		wgrad_visit(mode, b->ffn->w2, b->ffn->dw2);
	}
}

void lm_weights_to_device(lm *model)
{
	lm_each_wgrad(model, 0);
}

void lm_weights_host_dirty(lm *model)
{
	lm_each_wgrad(model, 1);
	if (model && model->embed)
		mat_host_dirty(model->embed->table);
}

void lm_weights_ensure_host(lm *model)
{
	lm_each_wgrad(model, 3);
	if (model && model->embed)
		(void)mat_pull_host(model->embed->table);
}

void lm_wgrads_zero_device(lm *model)
{
	lm_each_wgrad(model, 4);
}

void lm_grads_ensure_host(lm *model)
{
	lm_each_wgrad(model, 2);
}

lm_cache *lm_cache_create(int n)
{
	lm_cache *c;

	if (n <= 0)
		return NULL;
	c = malloc(sizeof(lm_cache));
	if (!c)
		return NULL;
	memset(c, 0, sizeof(*c));
	c->n = n;
	c->B = 1;
	c->T = n;
	c->ids = calloc((size_t)n, sizeof(int));
	c->hidden = mat_create(n, LM_D_MODEL);
	if (!c->ids || !c->hidden) {
		lm_cache_free(&c);
		return NULL;
	}
	mat_try_pin(c->hidden);
	return c;
}

void lm_cache_free(lm_cache **c)
{
	if (!c || !*c)
		return;
	free((*c)->ids);
	mat_free(&(*c)->hidden);
	transformer_cache_free(&(*c)->trc);
	free(*c);
	*c = NULL;
}

int lm_forward_cache(lm *model, const int *ids, int n, lm_cache *c)
{
	int i;

	if (!model || !ids || !c || n <= 0 || c->n != n || c->B != 1)
		return -1;
	for (i = 0; i < n; i++)
		c->ids[i] = ids[i];
	/* Reuse persistent transformer cache when sequence length matches. */
	if (!c->trc || c->trc->T != n || c->trc->B != 1) {
		transformer_cache_free(&c->trc);
		c->trc = transformer_cache_create(n, model->tr);
	}
	if (!c->trc)
		return -1;
	if (embed_lookup(c->hidden, model->embed, ids, n) != 0)
		return -1;
	posenc_sinusoidal_add(c->hidden);
	if (transformer_forward_cache(c->hidden, model->tr, c->trc) != 0)
		return -1;
	return 0;
}

int lm_forward(mat *hidden, const lm *model, const int *ids, int n)
{
	lm_cache *c;
	int rc;

	if (!hidden || !model || !ids || n <= 0)
		return -1;
	c = lm_cache_create(n);
	if (!c)
		return -1;
	rc = lm_forward_cache((lm *)model, ids, n, c);
	if (rc == 0)
		copy_mat(hidden, c->hidden);
	lm_cache_free(&c);
	return rc;
}

int lm_logits_at(vec *logits, const lm *model, const mat *hidden, int t)
{
	mat hrow;
	mat lrow;
	int v;
	const ann_real *h;

	if (!logits || !model || !hidden || !logits->data || !hidden->data)
		return -1;
	if (t < 0 || t >= hidden->rows)
		return -1;
	if (logits->n != model->vocab_size)
		return -1;
	h = &hidden->data[t * hidden->cols];
	if (model->w_head_q) {
		if (qmat_gemv_t(logits, model->w_head_q, h) != 0)
			return -1;
	} else {
		/* Stack row views are host-only: no device twin. */
		hrow.data = (ann_real *)h;
		hrow.d_data = NULL;
		hrow.on_device = MAT_DEV_HOST;
		hrow.rows = 1;
		hrow.cols = LM_D_MODEL;
		lrow.data = logits->data;
		lrow.d_data = NULL;
		lrow.on_device = MAT_DEV_HOST;
		lrow.rows = 1;
		lrow.cols = model->vocab_size;
		if (mat_mul(&lrow, &hrow, model->w_head) != 0)
			return -1;
	}
	for (v = 0; v < model->vocab_size; v++)
		logits->data[v] += model->b_head->data[v];
	return 0;
}

/*
** H1: batched LM head + softmax-CE over all T-1 predicting positions.
**   logits = hidden[0:T-1] @ W_head + b      (one GEMM, GPU when available)
**   delta  = softmax(logits) - onehot(next)  (gpu_softmax_rows or host rows)
**   dW_head += H^T delta                     (one GEMM via mat_outer_add)
**   dhidden = delta W_head^T                 (one OP_T GEMM via mat_mul_wt)
**   db_head += column sums of delta
** The T x V logits/delta matrix is built once and reused in place.
*/
/*
** Shared CE core: hv is rows x D (hidden of the predicting positions), tgt[t]
** the target of row t. Fills dhv (rows x D, host view) with dL/dhidden and
** accumulates dW_head / db_head. Also used by the packed path with gathered rows.
*/
static int ce_rows_backward(lm *model, const mat *hv, const int *tgt, int rows,
	mat *dhv)
{
	mat *delta;
	vec lrow;
	ann_real *tmp;
	ann_real *row;
	int V;
	int t;
	int v;

	V = model->vocab_size;
	for (t = 0; t < rows; t++)
		if (tgt[t] < 0 || tgt[t] >= V)
			return -1;
	delta = mat_create(rows, V);
	if (!delta)
		return -1;
	mat_try_pin(delta);
	mat_try_pin(dhv);

	/* 1. logits for every position. */
	if (model->w_head_q) {
		for (t = 0; t < rows; t++) {
			lrow.data = &delta->data[(size_t)t * V];
			lrow.n = V;
			if (lm_logits_at(&lrow, model, hv, t) != 0)
				goto fail;
		}
		mat_host_dirty(delta);
	} else {
		ann_real *dp;

		if (mat_mul(delta, hv, model->w_head) != 0)
			goto fail;
		dp = mat_compute_ptr(delta);
		if (dp && gpu_bias_row_add(dp, model->b_head->data, rows, V) == 0)
			mat_mark_device(delta);
		else {
			(void)mat_ensure_host(delta);
			for (t = 0; t < rows; t++) {
				row = &delta->data[(size_t)t * V];
				for (v = 0; v < V; v++)
					row[v] += model->b_head->data[v];
			}
			mat_host_dirty(delta);
		}
	}

	/* 2. softmax rows (GPU first, host rows otherwise). */
	{
		ann_real *dp = mat_compute_ptr(delta);

		if (dp && gpu_softmax_rows(dp, rows, V) == 0)
			mat_mark_device(delta);
		else {
			(void)mat_ensure_host(delta);
			if (gpu_softmax_rows(delta->data, rows, V) != 0) {
				tmp = malloc((size_t)V * sizeof(ann_real));
				if (!tmp)
					goto fail;
				for (t = 0; t < rows; t++)
					softmax_inplace(&delta->data[(size_t)t * V],
					    V, tmp);
				free(tmp);
			}
			mat_host_dirty(delta);
		}
	}

	/* 3. CE delta and db_head reduce (needs host one-hot edit). */
	(void)mat_ensure_host(delta);
	for (t = 0; t < rows; t++) {
		row = &delta->data[(size_t)t * V];
		row[tgt[t]] -= 1.0f;
		for (v = 0; v < V; v++)
			model->db_head->data[v] += row[v];
	}
	mat_host_dirty(delta);

	/* 4. dW_head += H^T delta. */
	if (mat_outer_add(model->dw_head, hv, delta) != 0)
		goto fail;

	/* 5. dhv = delta W_head^T. */
	if (mat_mul_wt(dhv, delta, model->w_head) != 0)
		goto fail;
	mat_free(&delta);
	return 0;
fail:
	mat_free(&delta);
	return -1;
}

/*
** H1: batched LM head + softmax-CE over all T-1 predicting positions.
**   logits = hidden[0:T-1] @ W_head + b      (one GEMM, GPU when available)
**   delta  = softmax(logits) - onehot(next)  (gpu_softmax_rows or host rows)
**   dW_head += H^T delta                     (one GEMM via mat_outer_add)
**   dhidden = delta W_head^T                 (one OP_T GEMM via mat_mul_wt)
**   db_head += column sums of delta
** The T x V logits/delta matrix is built once and reused in place.
*/
int lm_loss_backward(lm *model, const lm_cache *c, const int *ids, int n,
	mat *dhidden)
{
	mat hv;
	mat dhv;
	int rows;

	if (!model || !c || !ids || !dhidden || n < 2)
		return -1;
	rows = n - 1;
	if (!c->hidden || c->hidden->rows < rows || c->hidden->cols != LM_D_MODEL
	    || dhidden->rows < rows || dhidden->cols != LM_D_MODEL)
		return -1;
	/* Row-prefix view of hidden (contiguous): shares host and device twin. */
	hv.data = c->hidden->data;
	hv.d_data = c->hidden->d_data;
	hv.on_device = c->hidden->on_device;
	hv.rows = rows;
	hv.cols = LM_D_MODEL;
	/* Last row has no loss term. */
	mat_zero(dhidden);
	dhv.data = dhidden->data;
	dhv.d_data = NULL;
	dhv.on_device = MAT_DEV_HOST;
	dhv.rows = rows;
	dhv.cols = LM_D_MODEL;
	if (ce_rows_backward(model, &hv, ids + 1, rows, &dhv) != 0)
		return -1;
	mat_host_dirty(dhidden);
	return 0;
}

int lm_backward(lm *model, const mat *dhidden, const lm_cache *c)
{
	mat *dx;

	if (!model || !dhidden || !c || !c->trc)
		return -1;
	dx = mat_create(c->n, LM_D_MODEL);
	if (!dx)
		return -1;
	if (transformer_backward(dx, dhidden, model->tr, c->trc) != 0) {
		mat_free(&dx);
		return -1;
	}
	/* posenc is constant add — gradient passes through unchanged */
	if (embed_backward(model->embed, dx, c->ids, c->n) != 0) {
		mat_free(&dx);
		return -1;
	}
	mat_free(&dx);
	return 0;
}

lm_cache *lm_cache_create_packed(int B, int T)
{
	lm_cache *c;

	if (B <= 0 || T < 2)
		return NULL;
	if (B == 1)
		return lm_cache_create(T);
	c = malloc(sizeof(lm_cache));
	if (!c)
		return NULL;
	memset(c, 0, sizeof(*c));
	c->B = B;
	c->T = T;
	c->n = B * T;
	c->ids = calloc((size_t)c->n, sizeof(int));
	c->hidden = mat_create(c->n, LM_D_MODEL);
	/* n_blocks / heads are the LM_* constants (transformer_create). */
	if (c->ids && c->hidden)
		c->trc = transformer_cache_create_packed_dims(B, T,
		    LM_N_BLOCKS, LM_N_HEADS, LM_D_HEAD);
	if (!c->ids || !c->hidden || !c->trc) {
		lm_cache_free(&c);
		return NULL;
	}
	mat_try_pin(c->hidden);
	return c;
}

int lm_forward_cache_packed(lm *model, const int *ids, int B, int T,
	lm_cache *c)
{
	int b;
	int t;
	int n;

	if (!model || !ids || !c || B <= 0 || T < 2 || c->B != B || c->T != T
	    || c->n != B * T)
		return -1;
	n = c->n;
	if (B == 1)
		return lm_forward_cache(model, ids, n, c);
	if (!c->trc || c->trc->B != B || c->trc->Tseq != T) {
		transformer_cache_free(&c->trc);
		c->trc = transformer_cache_create_packed(B, T, model->tr);
	}
	if (!c->trc)
		return -1;
	memcpy(c->ids, ids, (size_t)n * sizeof(int));
	/* Embed all B*T ids once, then add positions restarting per sequence. */
	if (embed_lookup(c->hidden, model->embed, ids, n) != 0)
		return -1;
	for (b = 0; b < B; b++)
		for (t = 0; t < T; t++)
			posenc_add_row(&c->hidden->data[(size_t)(b * T + t)
			    * LM_D_MODEL], t);
	mat_host_dirty(c->hidden);
	return transformer_forward_cache(c->hidden, model->tr, c->trc);
}

/*
** Packed CE: gather hidden rows t = 0..T-2 of every sequence into a compact
** (B*(T-1)) x D matrix, run ONE head GEMM / softmax / dW_head GEMM over all of
** them, then scatter dhidden back (the last row of each sequence stays zero).
*/
int lm_loss_backward_packed(lm *model, const lm_cache *c, const int *ids,
	int B, int T, mat *dhidden)
{
	mat *hc;
	mat *dhc;
	int *tgt;
	int rows;
	int b;
	int t;
	int rc;
	size_t rb;

	if (!model || !c || !ids || !dhidden || B <= 0 || T < 2 || c->B != B
	    || c->T != T || !c->hidden || dhidden->rows < B * T
	    || dhidden->cols != LM_D_MODEL)
		return -1;
	if (B == 1)
		return lm_loss_backward(model, c, ids, T, dhidden);
	rows = B * (T - 1);
	rb = (size_t)LM_D_MODEL * sizeof(ann_real);
	hc = mat_create(rows, LM_D_MODEL);
	dhc = mat_create(rows, LM_D_MODEL);
	tgt = malloc((size_t)rows * sizeof(int));
	if (!hc || !dhc || !tgt) {
		mat_free(&hc);
		mat_free(&dhc);
		free(tgt);
		return -1;
	}
	mat_try_pin(hc);
	mat_try_pin(dhc);
	mat_try_pin(dhidden);
	/* Gather: prefer D2D when hidden is device-resident (R1). */
	if (c->hidden->d_data && c->hidden->on_device != MAT_DEV_HOST
	    && hc->d_data && gpu_available()) {
		for (b = 0; b < B; b++) {
			if (gpu_memcpy_d2d(
			    hc->d_data + (size_t)b * (T - 1) * LM_D_MODEL,
			    c->hidden->d_data + (size_t)b * T * LM_D_MODEL,
			    (size_t)(T - 1) * rb) != 0)
				goto gather_host;
			for (t = 0; t < T - 1; t++)
				tgt[b * (T - 1) + t] = ids[b * T + t + 1];
		}
		hc->on_device = MAT_DEV_DEVICE;
		goto gather_done;
	}
gather_host:
	(void)mat_ensure_host(c->hidden);
	for (b = 0; b < B; b++) {
		memcpy(&hc->data[(size_t)b * (T - 1) * LM_D_MODEL],
		    &c->hidden->data[(size_t)b * T * LM_D_MODEL],
		    (size_t)(T - 1) * rb);
		for (t = 0; t < T - 1; t++)
			tgt[b * (T - 1) + t] = ids[b * T + t + 1];
	}
	mat_host_dirty(hc);
gather_done:
	rc = ce_rows_backward(model, hc, tgt, rows, dhc);
	if (rc == 0) {
		mat_zero_dev(dhidden);
		if (dhc->d_data && dhc->on_device != MAT_DEV_HOST
		    && dhidden->d_data && gpu_available()) {
			for (b = 0; b < B; b++) {
				if (gpu_memcpy_d2d(
				    dhidden->d_data + (size_t)b * T * LM_D_MODEL,
				    dhc->d_data + (size_t)b * (T - 1) * LM_D_MODEL,
				    (size_t)(T - 1) * rb) != 0)
					goto scatter_host;
			}
			dhidden->on_device = MAT_DEV_DEVICE;
		} else {
scatter_host:
			(void)mat_ensure_host(dhc);
			(void)mat_ensure_host(dhidden);
			for (b = 0; b < B; b++)
				memcpy(&dhidden->data[(size_t)b * T * LM_D_MODEL],
				    &dhc->data[(size_t)b * (T - 1) * LM_D_MODEL],
				    (size_t)(T - 1) * rb);
			mat_host_dirty(dhidden);
		}
	}
	mat_free(&hc);
	mat_free(&dhc);
	free(tgt);
	return rc;
}

int lm_backward_packed(lm *model, const mat *dhidden, const lm_cache *c)
{
	if (!model || !dhidden || !c || !c->trc || dhidden->rows != c->n)
		return -1;
	/* c->n = B*T rows: transformer_backward / embed_backward are row-wise. */
	return lm_backward(model, dhidden, c);
}

/*
** Rough working set of one packed step. Per layer, in B*T*D floats: block
** buffers (x_in, ln1, attn_out, x_mid, ln2, ffn_out = 6), LN x_hat (2), attn
** (x_in, q, k, v, concat = 5), FFN (x_in + pre_act + hidden = 1 + 2*D_FF/D) and
** ~6 live backward temporaries; plus B*H*T*T attention scores per layer. Each
** is doubled for device twins / staging. Head: logits+delta over B*(T-1) x V,
** plus the gathered hidden / dhidden.
*/
size_t lm_packed_bytes(int B, int T, int vocab_size)
{
	double per_layer;
	double rows;
	double bytes;

	if (B <= 0 || T <= 0 || vocab_size <= 0)
		return 0;
	rows = (double)B * (double)T;
	per_layer = (6.0 + 2.0 + 5.0 + 1.0 + 2.0 * (double)LM_D_FF / LM_D_MODEL
	    + 6.0) * rows * LM_D_MODEL
	    + (double)B * LM_N_HEADS * (double)T * (double)T;
	bytes = per_layer * (double)LM_N_BLOCKS;
	bytes += 2.0 * rows * (double)vocab_size;	/* logits + delta */
	bytes += 4.0 * rows * LM_D_MODEL;
	/* R3: 1.6× after R1 residency (was 2.0 for heavy host staging twins). */
	return (size_t)(bytes * sizeof(float) * 1.6);
}

int lm_batch_auto(int T, int vocab_size, size_t free_bytes, double frac,
	int max_b, int fallback)
{
	int b;
	double budget;

	if (max_b < 1)
		max_b = 1;
	if (fallback < 1)
		fallback = 1;
	if (free_bytes == 0)
		return fallback < max_b ? fallback : max_b;
	if (frac <= 0.0 || frac > 1.0)
		frac = 0.85;
	budget = frac * (double)free_bytes;
	for (b = 1; b < max_b; b++)
		if ((double)lm_packed_bytes(b + 1, T, vocab_size) >= budget)
			break;
	return b;
}

double lm_nll(const lm *model, const int *ids, int n)
{
	mat *hidden;
	vec *logits;
	double loss;
	int t;
	int count;

	if (!model || !ids || n < 2)
		return -1.0;
	hidden = mat_create(n, LM_D_MODEL);
	logits = vec_create(model->vocab_size);
	if (!hidden || !logits) {
		mat_free(&hidden);
		vec_free(&logits);
		return -1.0;
	}
	if (lm_forward(hidden, model, ids, n) != 0) {
		mat_free(&hidden);
		vec_free(&logits);
		return -1.0;
	}
	loss = 0.0;
	count = 0;
	for (t = 0; t < n - 1; t++) {
		if (lm_logits_at(logits, model, hidden, t) != 0) {
			mat_free(&hidden);
			vec_free(&logits);
			return -1.0;
		}
		loss += vec_softmax_cross_entropy(logits, ids[t + 1]);
		count++;
	}
	mat_free(&hidden);
	vec_free(&logits);
	if (count == 0)
		return -1.0;
	return loss / (double)count;
}

double lm_perplexity(const lm *model, const int *ids, int n)
{
	double nll;

	nll = lm_nll(model, ids, n);
	if (nll < 0.0)
		return -1.0;
	return exp(nll);
}

double lm_train_steps(lm *model, const char *text, int steps, double lr)
{
	int ids[LM_MAX_CTX];
	int n;
	int step;
	lm_cache *cache;
	mat *dhidden;
	double loss;

	if (!model || !text || steps <= 0 || lr <= 0.0)
		return -1.0;
	n = lm_encode(model, text, ids, LM_MAX_CTX);
	if (n < 2)
		return -1.0;

	cache = lm_cache_create(n);
	dhidden = mat_create(n, LM_D_MODEL);
	if (!cache || !dhidden) {
		lm_cache_free(&cache);
		mat_free(&dhidden);
		return -1.0;
	}

	{
		lm_adam *opt;

		opt = lm_adam_create(model);
		if (!opt) {
			lm_cache_free(&cache);
			mat_free(&dhidden);
			return -1.0;
		}
		for (step = 0; step < steps; step++) {
			lm_adam_zero_grad(model);
			if (lm_forward_cache(model, ids, n, cache) != 0)
				break;
			if (lm_loss_backward(model, cache, ids, n, dhidden) != 0)
				break;
			if (lm_backward(model, dhidden, cache) != 0)
				break;
			lm_adam_step(opt, lr);
		}
		lm_weights_ensure_host(model);
		lm_adam_free(&opt);
	}

	loss = lm_nll(model, ids, n);
	lm_cache_free(&cache);
	mat_free(&dhidden);
	return loss;
}
