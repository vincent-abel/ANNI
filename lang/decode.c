/*
**	Device-resident incremental decode (KV cache on the GPU)
*/
#include <stdlib.h>
#include "block.h"
#include "decode.h"
#include "lm_config.h"

/* Device twin that matches the host weight (uploads when host is newer). */
static const ann_real *dev_weight(mat *m)
{
	if (!m)
		return NULL;
	if (!m->d_data || m->on_device == MAT_DEV_HOST)
		if (mat_sync_device(m) != 0)
			return NULL;
	return m->d_data;
}

lm_decoder *lm_decoder_create(lm *model)
{
	lm_decoder *d;
	gpu_dec_layer_w w;
	tblock *b;
	int i;

	if (!model || !model->tr || !model->w_head
	    || !gpu_available())
		return NULL;
	d = calloc(1, sizeof(*d));
	if (!d)
		return NULL;
	d->model = model;
	d->dec = gpu_dec_create(model->tr->n_blocks, LM_D_MODEL, LM_D_FF,
	    model->vocab_size, LM_MAX_CTX, model->tr->blocks[0].attn->n_heads);
	if (!d->dec)
		goto fail;
	for (i = 0; i < model->tr->n_blocks; i++) {
		b = &model->tr->blocks[i];
		w.wq = dev_weight(b->attn->wq);
		w.wk = dev_weight(b->attn->wk);
		w.wv = dev_weight(b->attn->wv);
		w.wo = dev_weight(b->attn->wo);
		w.w1 = dev_weight(b->ffn->w1);
		w.w2 = dev_weight(b->ffn->w2);
		w.ln1_g = b->ln1->gamma->data;
		w.ln1_b = b->ln1->beta->data;
		w.ln2_g = b->ln2->gamma->data;
		w.ln2_b = b->ln2->beta->data;
		w.b1 = b->ffn->b1->data;
		w.b2 = b->ffn->b2->data;
		w.eps1 = b->ln1->eps;
		w.eps2 = b->ln2->eps;
		if (gpu_dec_set_layer(d->dec, i, &w) != 0)
			goto fail;
	}
	if (gpu_dec_set_io(d->dec, dev_weight(model->embed->table),
	    dev_weight(model->w_head), model->b_head->data) != 0)
		goto fail;
	return d;
fail:
	lm_decoder_free(&d);
	return NULL;
}

void lm_decoder_free(lm_decoder **d)
{
	if (!d || !*d)
		return;
	gpu_dec_free(&(*d)->dec);
	free(*d);
	*d = NULL;
}

int lm_decoder_prefill(lm_decoder *d, const int *ids, int n, vec *logits)
{
	lm_cache *c;
	attn_cache *ac;
	int i;
	int rc;

	if (!d || !ids || n <= 0 || n > LM_MAX_CTX || !logits)
		return -1;
	c = lm_cache_create(n);
	if (!c)
		return -1;
	rc = -1;
	if (lm_forward_cache(d->model, ids, n, c) != 0
	    || lm_logits_at(logits, d->model, c->hidden, n - 1) != 0)
		goto out;
	for (i = 0; i < c->trc->n_blocks; i++) {
		ac = c->trc->bc[i].attnc;
		if (!ac || mat_pull_host(ac->k) != 0 || mat_pull_host(ac->v) != 0
		    || !ac->k->data || !ac->v->data
		    || gpu_dec_kv_load(d->dec, i, ac->k->data, ac->v->data, n)
		    != 0)
			goto out;
	}
	d->len = n;
	rc = 0;
out:
	lm_cache_free(&c);
	return rc;
}

int lm_decoder_step(lm_decoder *d, int id, vec *logits)
{
	if (!d || !logits || logits->n != d->model->vocab_size
	    || d->len >= LM_MAX_CTX)
		return -1;
	if (gpu_dec_step(d->dec, id, d->len, logits->data) != 0)
		return -1;
	d->len++;
	return 0;
}
