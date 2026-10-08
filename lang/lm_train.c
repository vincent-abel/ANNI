/*
**	Mini-batch corpus training for the LM
*/
#include <stdlib.h>
#include "gradutil.h"
#include "lm_train.h"

static void scale_grads(lm *model, double scale)
{
	int i;
	int n;
	tblock *b;

	if (!model || scale == 1.0)
		return;
	n = model->dw_head->rows * model->dw_head->cols;
	for (i = 0; i < n; i++)
		model->dw_head->data[i] *= scale;
	for (i = 0; i < model->db_head->n; i++)
		model->db_head->data[i] *= scale;
	n = model->embed->dtable->rows * model->embed->dtable->cols;
	for (i = 0; i < n; i++)
		model->embed->dtable->data[i] *= scale;
	for (i = 0; i < model->tr->n_blocks; i++) {
		int k;
		b = &model->tr->blocks[i];
		n = LM_D_MODEL * LM_D_MODEL;
		for (k = 0; k < n; k++) {
			b->attn->dwq->data[k] *= scale;
			b->attn->dwk->data[k] *= scale;
			b->attn->dwv->data[k] *= scale;
			b->attn->dwo->data[k] *= scale;
		}
		n = LM_D_MODEL * LM_D_FF;
		for (k = 0; k < n; k++)
			b->ffn->dw1->data[k] *= scale;
		n = LM_D_FF * LM_D_MODEL;
		for (k = 0; k < n; k++)
			b->ffn->dw2->data[k] *= scale;
		for (k = 0; k < LM_D_FF; k++)
			b->ffn->db1->data[k] *= scale;
		for (k = 0; k < LM_D_MODEL; k++) {
			b->ffn->db2->data[k] *= scale;
			b->ln1->dgamma->data[k] *= scale;
			b->ln1->dbeta->data[k] *= scale;
			b->ln2->dgamma->data[k] *= scale;
			b->ln2->dbeta->data[k] *= scale;
		}
	}
}

double lm_train_corpus(lm *model, text_dataset *ds, lm_adam *opt,
	int epochs, int batch_size, double lr)
{
	int ep;
	int nw;
	int idx;
	int b;
	int n;
	int ids[LM_MAX_CTX];
	lm_cache *cache;
	mat *dhidden;
	double last;

	if (!model || !ds || !opt || epochs <= 0 || batch_size <= 0 || lr <= 0.0)
		return -1.0;
	nw = text_dataset_n_windows(ds);
	if (nw <= 0)
		return -1.0;
	cache = lm_cache_create(ds->window);
	dhidden = mat_create(ds->window, LM_D_MODEL);
	if (!cache || !dhidden) {
		lm_cache_free(&cache);
		mat_free(&dhidden);
		return -1.0;
	}
	last = -1.0;
	for (ep = 0; ep < epochs; ep++) {
		idx = 0;
		while (idx < nw) {
			lm_adam_zero_grad(model);
			b = 0;
			while (b < batch_size && idx < nw) {
				if (text_dataset_window(ds, idx, ids, &n) != 0)
					break;
				if (lm_forward_cache(model, ids, n, cache) != 0)
					break;
				if (lm_loss_backward(model, cache, ids, n, dhidden) != 0)
					break;
				if (lm_backward(model, dhidden, cache) != 0)
					break;
				b++;
				idx++;
			}
			if (b <= 0)
				break;
			scale_grads(model, 1.0 / (double)b);
			lm_adam_step(opt, lr);
		}
		last = lm_nll(model, ds->ids, ds->n_ids < LM_MAX_CTX ? ds->n_ids : LM_MAX_CTX);
	}
	lm_cache_free(&cache);
	mat_free(&dhidden);
	return last;
}
