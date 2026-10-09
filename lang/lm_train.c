/*
**	Mini-batch corpus training for the LM
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "gpu.h"
#include "gradutil.h"
#include "lm_serialize.h"
#include "lm_train.h"
#include "mat.h"

/* Wall time — clock() freezes while waiting on CUDA. */
static double wall_sec(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

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

static int g_progress_open;

static void dbg_progress_close(void)
{
	if (!g_progress_open)
		return;
	fputc('\n', stderr);
	fflush(stderr);
	g_progress_open = 0;
}

static void dbg_stamp(const char *msg)
{
	time_t now;
	struct tm *tm;
	char buf[32];

	dbg_progress_close();
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
		snprintf(out, out_n, "--:--");
		return;
	}
	s = (int)(sec + 0.5);
	h = s / 3600;
	s %= 3600;
	m = s / 60;
	s %= 60;
	if (h > 0)
		snprintf(out, out_n, "%dh%02dm", h, m);
	else
		snprintf(out, out_n, "%02d:%02d", m, s);
}

/* Single in-place progress line (\\r). Close it before any stamp/log line. */
static void dbg_progress(int ep, int epochs, int done, int total,
	int epoch_done, int nw, double nll, double win_nll, double t0,
	int tokens_done, int force)
{
	int width;
	int filled;
	int i;
	double ep_pct;
	double elapsed;
	double eta;
	double ms_tok;
	double now;
	char eta_buf[16];
	char bar[16];
	char line[96];
	int nline;
	static double last_print;

	now = wall_sec();
	if (!force && last_print > 0.0 && (now - last_print) < 0.25)
		return;
	last_print = now;

	width = 12;
	if (nw <= 0)
		nw = 1;
	if (total <= 0)
		total = 1;
	if (epoch_done < 0)
		epoch_done = 0;
	if (epoch_done > nw)
		epoch_done = nw;
	ep_pct = 100.0 * (double)epoch_done / (double)nw;
	filled = (int)((double)width * (double)epoch_done / (double)nw + 0.5);
	if (filled > width)
		filled = width;
	for (i = 0; i < width; i++)
		bar[i] = i < filled ? '#' : '-';
	bar[width] = '\0';
	elapsed = now - t0;
	eta = -1.0;
	ms_tok = -1.0;
	if (done > 0 && elapsed > 0.0) {
		eta = elapsed * (double)(total - done) / (double)done;
		if (tokens_done > 0)
			ms_tok = 1000.0 * elapsed / (double)tokens_done;
	}
	dbg_fmt_eta(eta_buf, sizeof(eta_buf), eta);
	nline = snprintf(line, sizeof(line),
	    "ep %d/%d [%s] %4.0f%% %d/%d",
	    ep + 1, epochs, bar, ep_pct, epoch_done, nw);
	if (nline < 0)
		nline = 0;
	if (nll > 0.0 && (size_t)nline < sizeof(line))
		nline += snprintf(line + nline, sizeof(line) - (size_t)nline,
		    " NLL %.3f", nll);
	if (win_nll > 0.0 && (size_t)nline < sizeof(line))
		nline += snprintf(line + nline, sizeof(line) - (size_t)nline,
		    " win %.3f", win_nll);
	if (ms_tok > 0.0 && (size_t)nline < sizeof(line))
		nline += snprintf(line + nline, sizeof(line) - (size_t)nline,
		    " %.2fms/t", ms_tok);
	if (eta >= 0.0 && (size_t)nline < sizeof(line))
		snprintf(line + nline, sizeof(line) - (size_t)nline,
		    " ETA %s", eta_buf);
	/* Pad/clear leftovers from a longer previous frame. */
	fprintf(stderr, "\r%-96s", line);
	fflush(stderr);
	g_progress_open = 1;
	if (force)
		dbg_progress_close();
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
	int epochs, int batch_size, double lr, int debug,
	const char *ckpt_path, int epoch_base)
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
	int epoch_num;
	int *order;
	int tmp;
	int j;
	int r;
	lm_cache *cache;
	mat *dhidden;
	double last;
	double win;
	double t0;
	char msg[192];

	if (!model || !ds || !opt || epochs <= 0 || batch_size <= 0 || lr <= 0.0)
		return -1.0;
	if (epoch_base < 0)
		epoch_base = 0;
	nw = text_dataset_n_windows(ds);
	if (nw <= 0)
		return -1.0;
	order = malloc((size_t)nw * sizeof(int));
	cache = lm_cache_create(ds->window);
	dhidden = mat_create(ds->window, LM_D_MODEL);
	if (!order || !cache || !dhidden) {
		free(order);
		lm_cache_free(&cache);
		mat_free(&dhidden);
		return -1.0;
	}
	for (j = 0; j < nw; j++)
		order[j] = j;
	mat_try_pin(dhidden);
	mat_try_pin(model->dw_head);
	total_steps = epochs * nw;
	done_steps = 0;
	tokens_done = 0;
	last = lm_nll(model, ds->ids,
	    ds->n_ids < LM_MAX_CTX ? ds->n_ids : LM_MAX_CTX);
	win = lm_nll_windows(model, ds);
	t0 = wall_sec();
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
		/* Seeded shuffle so parallel --seed runs diverge after resume. */
		for (j = nw - 1; j > 0; j--) {
			r = rand() % (j + 1);
			tmp = order[j];
			order[j] = order[r];
			order[r] = tmp;
		}
		idx = 0;
		while (idx < nw) {
			lm_adam_zero_grad(model);
			/* Defer GEMM D2H until the optimizer step boundary. */
			gpu_set_sync_mode(GPU_SYNC_BATCH);
			b = 0;
			while (b < batch_size && idx < nw) {
				if (text_dataset_window(ds, order[idx], ids, &n) != 0)
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
				    idx, nw, last, win, t0, tokens_done, 0);
		}
		/* Epoch boundary: long ctx NLL + mean over train windows. */
		last = lm_nll(model, ds->ids,
		    ds->n_ids < LM_MAX_CTX ? ds->n_ids : LM_MAX_CTX);
		win = lm_nll_windows(model, ds);
		epoch_num = epoch_base + ep + 1;
		if (ckpt_path) {
			if (lm_save_checkpoint(model, opt, epoch_num, 0,
				ckpt_path) != 0) {
				if (debug)
					dbg_stamp("checkpoint save failed");
				free(order);
				lm_cache_free(&cache);
				mat_free(&dhidden);
				return -1.0;
			}
			if (debug) {
				snprintf(msg, sizeof(msg),
				    "saved checkpoint %s (epoch %d)",
				    ckpt_path, epoch_num);
				dbg_stamp(msg);
			}
		}
		if (debug) {
			dbg_progress(ep, epochs, done_steps, total_steps,
			    nw, nw, last, win, t0, tokens_done, 1);
			snprintf(msg, sizeof(msg),
			    "epoch %d/%d done  NLL %.4f  window NLL %.4f",
			    ep + 1, epochs, last, win);
			dbg_stamp(msg);
		}
	}
	gpu_set_sync_mode(GPU_SYNC_EAGER);
	if (debug)
		dbg_stamp("train loop finished");
	free(order);
	lm_cache_free(&cache);
	mat_free(&dhidden);
	return last;
}
