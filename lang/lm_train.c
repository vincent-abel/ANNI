/*
**	Mini-batch corpus training for the LM
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gpu.h"
#include "gradutil.h"
#include "lm_train.h"
#include "mat.h"

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

static void dbg_stamp(const char *msg)
{
	time_t now;
	struct tm *tm;
	char buf[32];

	now = time(NULL);
	tm = localtime(&now);
	if (tm)
		strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
	else
		snprintf(buf, sizeof(buf), "unknown-time");
	fprintf(stderr, "[%s] %s\n", buf, msg);
}

static void dbg_fmt_eta(char *out, size_t out_n, double sec)
{
	int h;
	int m;
	int s;

	if (!out || out_n == 0)
		return;
	if (sec < 0.0 || sec > 86400.0 * 7.0) {
		snprintf(out, out_n, "--:--:--");
		return;
	}
	s = (int)(sec + 0.5);
	h = s / 3600;
	s %= 3600;
	m = s / 60;
	s %= 60;
	if (h > 0)
		snprintf(out, out_n, "%d:%02d:%02d", h, m, s);
	else
		snprintf(out, out_n, "%02d:%02d", m, s);
}

static void dbg_progress(int ep, int epochs, int done, int total,
	double nll, double win_nll, clock_t t0, int tokens_done)
{
	int width;
	int filled;
	int i;
	double pct;
	double elapsed;
	double eta;
	double ms_tok;
	time_t now;
	struct tm *tm;
	char buf[32];
	char eta_buf[16];

	width = 28;
	if (total <= 0)
		total = 1;
	pct = 100.0 * (double)done / (double)total;
	filled = (int)((double)width * (double)done / (double)total + 0.5);
	if (filled > width)
		filled = width;
	elapsed = (double)(clock() - t0) / (double)CLOCKS_PER_SEC;
	eta = -1.0;
	ms_tok = -1.0;
	if (done > 0 && elapsed > 0.0) {
		eta = elapsed * (double)(total - done) / (double)done;
		if (tokens_done > 0)
			ms_tok = 1000.0 * elapsed / (double)tokens_done;
	}
	dbg_fmt_eta(eta_buf, sizeof(eta_buf), eta);
	now = time(NULL);
	tm = localtime(&now);
	if (tm)
		strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
	else
		snprintf(buf, sizeof(buf), "unknown-time");
	fprintf(stderr, "\r[%s] [", buf);
	for (i = 0; i < width; i++)
		fputc(i < filled ? '#' : '-', stderr);
	fprintf(stderr, "] %5.1f%%  epoch %d/%d  windows %d/%d",
	    pct, ep + 1, epochs, done, total);
	if (nll > 0.0)
		fprintf(stderr, "  NLL %.4f", nll);
	if (win_nll > 0.0)
		fprintf(stderr, "  win NLL %.4f", win_nll);
	if (ms_tok > 0.0)
		fprintf(stderr, "  %.2f ms/tok", ms_tok);
	if (eta >= 0.0)
		fprintf(stderr, "  ETA %s", eta_buf);
	fflush(stderr);
}

double lm_nll_windows(const lm *model, const text_dataset *ds)
{
	int ids[LM_MAX_CTX];
	int n;
	int i;
	int nw;
	double sum;

	if (!model || !ds)
		return -1.0;
	nw = text_dataset_n_windows(ds);
	if (nw <= 0)
		return -1.0;
	sum = 0.0;
	for (i = 0; i < nw; i++) {
		if (text_dataset_window(ds, i, ids, &n) != 0)
			return -1.0;
		sum += lm_nll(model, ids, n);
	}
	return sum / (double)nw;
}

double lm_train_corpus(lm *model, text_dataset *ds, lm_adam *opt,
	int epochs, int batch_size, double lr, int debug)
{
	int ep;
	int nw;
	int idx;
	int b;
	int n;
	int ids[LM_MAX_CTX];
	int total_steps;
	int done_steps;
	int tokens_done;
	lm_cache *cache;
	mat *dhidden;
	double last;
	double win;
	clock_t t0;
	char msg[160];

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
	mat_try_pin(dhidden);
	mat_try_pin(model->dw_head);
	total_steps = epochs * nw;
	done_steps = 0;
	tokens_done = 0;
	last = lm_nll(model, ds->ids,
	    ds->n_ids < LM_MAX_CTX ? ds->n_ids : LM_MAX_CTX);
	win = lm_nll_windows(model, ds);
	t0 = clock();
	if (debug) {
		snprintf(msg, sizeof(msg),
		    "train loop: %d epochs, %d windows/epoch, batch %d, lr %.4g",
		    epochs, nw, batch_size, lr);
		dbg_stamp(msg);
		snprintf(msg, sizeof(msg),
		    "start NLL %.4f  window NLL %.4f", last, win);
		dbg_stamp(msg);
	}
	for (ep = 0; ep < epochs; ep++) {
		if (debug) {
			snprintf(msg, sizeof(msg), "epoch %d/%d starting",
			    ep + 1, epochs);
			dbg_stamp(msg);
		}
		idx = 0;
		while (idx < nw) {
			lm_adam_zero_grad(model);
			/* Defer GEMM D2H until the optimizer step boundary. */
			gpu_set_sync_mode(GPU_SYNC_BATCH);
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
				done_steps++;
				tokens_done += n;
			}
			if (b <= 0) {
				gpu_set_sync_mode(GPU_SYNC_EAGER);
				break;
			}
			/* Sync grads/activations to host for scale + Adam pack. */
			mat_ensure_host(dhidden);
			mat_ensure_host(cache->hidden);
			mat_ensure_host(model->dw_head);
			gpu_set_sync_mode(GPU_SYNC_EAGER);
			scale_grads(model, 1.0 / (double)b);
			lm_adam_step(opt, lr);
			if (debug)
				dbg_progress(ep, epochs, done_steps, total_steps,
				    last, win, t0, tokens_done);
		}
		/* Epoch boundary: long ctx NLL + mean over train windows. */
		last = lm_nll(model, ds->ids,
		    ds->n_ids < LM_MAX_CTX ? ds->n_ids : LM_MAX_CTX);
		win = lm_nll_windows(model, ds);
		if (debug) {
			fputc('\n', stderr);
			snprintf(msg, sizeof(msg),
			    "epoch %d/%d done  NLL %.4f  window NLL %.4f",
			    ep + 1, epochs, last, win);
			dbg_stamp(msg);
		}
	}
	gpu_set_sync_mode(GPU_SYNC_EAGER);
	if (debug)
		dbg_stamp("train loop finished");
	lm_cache_free(&cache);
	mat_free(&dhidden);
	return last;
}
