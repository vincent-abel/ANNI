/*
**	Adam optimizer over flat LM param/grad slots
*/
#include <math.h>
#include <stdlib.h>
#include "gpu.h"
#include "gradutil.h"
#include "lm_adam.h"

static void add_seg(lm_adam *opt, mat *w, mat *dw, vec *wv, vec *dwv,
	int off, int n, int host_use)
{
	lm_adam_seg *s;

	s = &opt->segs[opt->n_segs++];
	s->w = w;
	s->dw = dw;
	s->wv = wv;
	s->dwv = dwv;
	s->off = off;
	s->n = n;
	s->host_use = host_use;
}

static void add_mat(lm_adam *opt, int *i, mat *w, mat *dw, int host_use)
{
	int k;
	int n;

	if (!w || !dw || !w->data || !dw->data)
		return;
	n = w->rows * w->cols;
	add_seg(opt, w, dw, NULL, NULL, *i, n, host_use);
	for (k = 0; k < n; k++) {
		opt->slots[*i].param = &w->data[k];
		opt->slots[*i].grad = &dw->data[k];
		opt->slots[*i].m = 0.0f;
		opt->slots[*i].v = 0.0f;
		(*i)++;
	}
}

static void add_vec(lm_adam *opt, int *i, vec *w, vec *dw)
{
	int k;

	if (!w || !dw || !w->data || !dw->data)
		return;
	add_seg(opt, NULL, NULL, w, dw, *i, w->n, 1);
	for (k = 0; k < w->n; k++) {
		opt->slots[*i].param = &w->data[k];
		opt->slots[*i].grad = &dw->data[k];
		opt->slots[*i].m = 0.0f;
		opt->slots[*i].v = 0.0f;
		(*i)++;
	}
}

int lm_param_count(const lm *model)
{
	int n;
	int i;

	if (!model || !model->w_head || !model->b_head || !model->embed
	    || !model->embed->table || !model->tr)
		return 0;
	n = model->w_head->rows * model->w_head->cols + model->b_head->n;
	n += model->embed->table->rows * model->embed->table->cols;
	for (i = 0; i < model->tr->n_blocks; i++) {
		n += 4 * LM_D_MODEL * LM_D_MODEL;
		n += LM_D_MODEL * LM_D_FF + LM_D_FF + LM_D_FF * LM_D_MODEL
		    + LM_D_MODEL;
		n += 4 * LM_D_MODEL;
	}
	return n;
}

lm_adam *lm_adam_create(lm *model)
{
	lm_adam *opt;
	int i;
	int n;
	tblock *b;

	if (!model)
		return NULL;
	n = lm_param_count(model);
	opt = malloc(sizeof(lm_adam));
	if (!opt)
		return NULL;
	opt->slots = calloc((size_t)n, sizeof(lm_adam_slot));
	opt->segs = calloc((size_t)(4 + 12 * model->tr->n_blocks),
	    sizeof(lm_adam_seg));
	if (!opt->slots || !opt->segs) {
		free(opt->slots);
		free(opt->segs);
		free(opt);
		return NULL;
	}
	opt->n = n;
	opt->n_segs = 0;
	opt->t = 0;
	opt->h_m = opt->h_v = NULL;
	opt->d_param = opt->d_grad = opt->d_m = opt->d_v = NULL;
	opt->dev_mv = 0;
	opt->params_valid = 0;
	opt->beta1 = 0.9;
	opt->beta2 = 0.999;
	opt->eps = 1e-8;
	opt->weight_decay = 0.0;
	opt->grad_clip = 0.0;
	i = 0;
	add_mat(opt, &i, model->w_head, model->dw_head, 0);
	add_vec(opt, &i, model->b_head, model->db_head);
	/* Embedding lookup/scatter are host code: table/dtable stay host-used. */
	add_mat(opt, &i, model->embed->table, model->embed->dtable, 0); /* R2 */
	for (n = 0; n < model->tr->n_blocks; n++) {
		b = &model->tr->blocks[n];
		add_mat(opt, &i, b->attn->wq, b->attn->dwq, 0);
		add_mat(opt, &i, b->attn->wk, b->attn->dwk, 0);
		add_mat(opt, &i, b->attn->wv, b->attn->dwv, 0);
		add_mat(opt, &i, b->attn->wo, b->attn->dwo, 0);
		add_mat(opt, &i, b->ffn->w1, b->ffn->dw1, 0);
		add_vec(opt, &i, b->ffn->b1, b->ffn->db1);
		add_mat(opt, &i, b->ffn->w2, b->ffn->dw2, 0);
		add_vec(opt, &i, b->ffn->b2, b->ffn->db2);
		add_vec(opt, &i, b->ln1->gamma, b->ln1->dgamma);
		add_vec(opt, &i, b->ln1->beta, b->ln1->dbeta);
		add_vec(opt, &i, b->ln2->gamma, b->ln2->dgamma);
		add_vec(opt, &i, b->ln2->beta, b->ln2->dbeta);
	}
	opt->n = i;
	return opt;
}

static void release_gpu(lm_adam *opt)
{
	gpu_free(opt->d_param);
	gpu_free(opt->d_grad);
	gpu_free(opt->d_m);
	gpu_free(opt->d_v);
	free(opt->h_m);
	free(opt->h_v);
	opt->d_param = opt->d_grad = opt->d_m = opt->d_v = NULL;
	opt->h_m = opt->h_v = NULL;
	opt->dev_mv = 0;
	opt->params_valid = 0;
}

void lm_adam_free(lm_adam **opt)
{
	if (!opt || !*opt)
		return;
	release_gpu(*opt);
	free((*opt)->slots);
	free((*opt)->segs);
	free(*opt);
	*opt = NULL;
}

void lm_adam_zero_grad(lm *model)
{
	lm_zero_grad(model);
	/* dW twins are only written on device: zero them there, no H2D later. */
	lm_wgrads_zero_device(model);
}

void lm_adam_invalidate(lm_adam *opt)
{
	if (opt)
		opt->params_valid = 0;
}

/* Allocate persistent device flats once; m/v are uploaded on first use. */
static int gpu_buffers_init(lm_adam *opt)
{
	size_t nbytes;
	int i;

	if (opt->d_param)
		return 0;
	nbytes = (size_t)opt->n * sizeof(ann_real);
	opt->h_m = malloc(nbytes);
	opt->h_v = malloc(nbytes);
	opt->d_param = gpu_malloc(nbytes);
	opt->d_grad = gpu_malloc(nbytes);
	opt->d_m = gpu_malloc(nbytes);
	opt->d_v = gpu_malloc(nbytes);
	if (!opt->h_m || !opt->h_v
	    || !opt->d_param || !opt->d_grad || !opt->d_m || !opt->d_v) {
		release_gpu(opt);
		return -1;
	}
	for (i = 0; i < opt->n; i++) {
		opt->h_m[i] = opt->slots[i].m;
		opt->h_v[i] = opt->slots[i].v;
	}
	if (gpu_memcpy_h2d(opt->d_m, opt->h_m, nbytes) != 0
	    || gpu_memcpy_h2d(opt->d_v, opt->h_v, nbytes) != 0) {
		release_gpu(opt);
		return -1;
	}
	return 0;
}

void lm_adam_sync_host(lm_adam *opt)
{
	size_t nbytes;
	int i;

	if (!opt)
		return;
	/* Weights: device-authoritative mats -> host (state becomes synced). */
	for (i = 0; i < opt->n_segs; i++)
		if (opt->segs[i].w)
			(void)mat_pull_host(opt->segs[i].w);
	if (!opt->dev_mv || !opt->d_m || !opt->d_v)
		return;
	nbytes = (size_t)opt->n * sizeof(ann_real);
	if (gpu_memcpy_d2h(opt->h_m, opt->d_m, nbytes) != 0
	    || gpu_memcpy_d2h(opt->h_v, opt->d_v, nbytes) != 0)
		return;
	for (i = 0; i < opt->n; i++) {
		opt->slots[i].m = opt->h_m[i];
		opt->slots[i].v = opt->h_v[i];
	}
	opt->dev_mv = 0;
}

/* Bring this segment's param + grad into the flat device buffers. */
static int seg_gather(lm_adam *opt, const lm_adam_seg *s)
{
	size_t nb;
	ann_real *dp;
	ann_real *dg;
	mat *w;
	mat *dw;

	nb = (size_t)s->n * sizeof(ann_real);
	dp = opt->d_param + s->off;
	dg = opt->d_grad + s->off;
	if (s->wv) {
		/* Bias / LN vecs are host-resident (tiny): plain H2D. */
		if (gpu_memcpy_h2d(dp, s->wv->data, nb) != 0
		    || gpu_memcpy_h2d(dg, s->dwv->data, nb) != 0)
			return -1;
		return 0;
	}
	w = s->w;
	dw = s->dw;
	if (w->d_data && w->on_device != MAT_DEV_HOST) {
		/* Device twin is current (device- or host-synced). */
		if (!opt->params_valid
		    && gpu_memcpy_d2d(dp, w->d_data, nb) != 0)
			return -1;
	} else if (gpu_memcpy_h2d(dp, w->data, nb) != 0) {
		return -1;	/* host is authoritative (or no twin) */
	}
	if (dw->d_data && !s->host_use && dw->on_device != MAT_DEV_HOST)
		return gpu_memcpy_d2d(dg, dw->d_data, nb) == 0 ? 0 : -1;
	return gpu_memcpy_h2d(dg, dw->data, nb) == 0 ? 0 : -1;
}

/* Publish updated params to the weight's device twin (+ host if needed). */
static int seg_scatter(lm_adam *opt, const lm_adam_seg *s)
{
	size_t nb;
	ann_real *dp;
	mat *w;

	nb = (size_t)s->n * sizeof(ann_real);
	dp = opt->d_param + s->off;
	if (s->wv) {
		if (gpu_memcpy_d2h(s->wv->data, dp, nb) != 0)
			return -1;
		vec_zero(s->dwv);
		return 0;
	}
	w = s->w;
	if (w->d_data && gpu_memcpy_d2d(w->d_data, dp, nb) == 0) {
		if (!s->host_use)
			w->on_device = MAT_DEV_DEVICE;	/* host stale until pulled */
		else if (gpu_memcpy_d2h(w->data, dp, nb) == 0)
			w->on_device = MAT_DEV_SYNCED;
		else
			return -1;
	} else {
		if (gpu_memcpy_d2h(w->data, dp, nb) != 0)
			return -1;
		mat_host_dirty(w);
	}
	if (s->host_use)
		mat_zero(s->dw);
	else
		mat_zero_dev(s->dw);
	return 0;
}

/*
** Device-resident step: param/grad/m/v live on the GPU. dW is gathered D2D
** from the grad twins, the update runs in place on d_param, and the result is
** scattered D2D into each weight's d_data (no host pack/unpack, no D2H). Only
** host-consumed segments (embedding table, bias/LN vecs) cross PCIe.
*/
static int adam_step_gpu(lm_adam *opt, double lr, double grad_scale)
{
	int i;

	if (gpu_buffers_init(opt) != 0)
		return -1;
	gpu_device_sync();	/* grads written by queued batch kernels */
	for (i = 0; i < opt->n_segs; i++)
		if (seg_gather(opt, &opt->segs[i]) != 0)
			goto fail;
	if (gpu_adam_step_dev(opt->d_param, opt->d_grad, opt->d_m, opt->d_v,
	    opt->n, lr, opt->beta1, opt->beta2, opt->eps, opt->t, grad_scale) != 0)
		goto fail;
	opt->dev_mv = 1;
	for (i = 0; i < opt->n_segs; i++)
		if (seg_scatter(opt, &opt->segs[i]) != 0)
			goto fail;
	opt->params_valid = 1;
	return 0;
fail:
	opt->params_valid = 0;
	return -1;
}

void lm_adam_step(lm_adam *opt, double lr)
{
	lm_adam_step_scaled(opt, lr, 1.0);
}

void lm_adam_step_scaled(lm_adam *opt, double lr, double grad_scale)
{
	int i;
	double g;
	double mhat;
	double vhat;
	double bc1;
	double bc2;

	if (!opt || lr <= 0.0)
		return;
	opt->t++;
	if (gpu_available() && adam_step_gpu(opt, lr, grad_scale) == 0)
		return;
	/* Host fallback: bring resident m/v, weights and grads back first. */
	lm_adam_sync_host(opt);
	for (i = 0; i < opt->n_segs; i++) {
		if (opt->segs[i].w)
			(void)mat_ensure_host(opt->segs[i].w);
		if (opt->segs[i].dw)
			(void)mat_ensure_host(opt->segs[i].dw);
	}
	if (opt->d_m) {
		/* Moments now live in slots; re-upload next GPU step. */
		release_gpu(opt);
	}
	bc1 = 1.0 - pow(opt->beta1, (double)opt->t);
	bc2 = 1.0 - pow(opt->beta2, (double)opt->t);
	if (opt->grad_clip > 0.0) {
		double norm = 0.0;
		double s;

		for (i = 0; i < opt->n; i++) {
			g = *opt->slots[i].grad * grad_scale;
			norm += g * g;
		}
		norm = sqrt(norm);
		if (norm > opt->grad_clip) {
			s = opt->grad_clip / (norm + 1e-12);
			grad_scale *= s;
		}
	}
	for (i = 0; i < opt->n; i++) {
		g = *opt->slots[i].grad * grad_scale;
		opt->slots[i].m = opt->beta1 * opt->slots[i].m + (1.0 - opt->beta1) * g;
		opt->slots[i].v = opt->beta2 * opt->slots[i].v + (1.0 - opt->beta2) * g * g;
		mhat = opt->slots[i].m / bc1;
		vhat = opt->slots[i].v / bc2;
		*opt->slots[i].param -= lr * mhat / (sqrt(vhat) + opt->eps);
		if (opt->weight_decay > 0.0)
			*opt->slots[i].param -= lr * opt->weight_decay
			    * *opt->slots[i].param;
		*opt->slots[i].grad = 0.0f;
	}
	/* Host wrote weights and zeroed grads: device twins are stale. */
	for (i = 0; i < opt->n_segs; i++) {
		if (opt->segs[i].w)
			mat_host_dirty(opt->segs[i].w);
		if (opt->segs[i].dw)
			mat_host_dirty(opt->segs[i].dw);
	}
}
