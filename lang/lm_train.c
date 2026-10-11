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
#include "lm_config.h"
#include "lm_serialize.h"
#include "lm_train.h"
#include "mat.h"

/* E10: tokens/sec, rough TFLOP/s, VRAM used (MB). */
static void fill_throughput(train_progress *tp, int tokens_done, double elapsed)
{
	size_t fr;
	size_t tot;
	long params;

	tp->tok_sec = -1.0;
	tp->tflops_est = -1.0;
	tp->vram_used_mb = -1.0;
	if (tokens_done > 0 && elapsed > 0.0) {
		tp->tok_sec = (double)tokens_done / elapsed;
		params = (long)LM_N_BLOCKS
		    * (12L * LM_D_MODEL * LM_D_MODEL
		    + 2L * LM_D_MODEL * LM_D_FF);
		tp->tflops_est = 6.0 * (double)params * tp->tok_sec / 1e12;
	}
	fr = 0;
	tot = 0;
	if (gpu_available() && gpu_mem_info(&fr, &tot) == 0 && tot > 0)
		tp->vram_used_mb = (double)(tot - fr) / 1048576.0;
}

static train_progress_fn g_progress_fn;
static void *g_progress_ud;
static train_control *g_train_ctl;

void lm_train_set_progress(train_progress_fn fn, void *userdata)
{
	g_progress_fn = fn;
	g_progress_ud = userdata;
}

void lm_train_set_control(train_control *ctl)
{
	g_train_ctl = ctl;
}

static double train_step_lr(double lr)
{
	if (g_train_ctl && g_train_ctl->lr > 0.0)
		return g_train_ctl->lr;
	return lr;
}

static void emit_progress(const train_progress *p)
{
	if (g_progress_fn)
		g_progress_fn(p, g_progress_ud);
}

/*
** While paused, keep calling the progress callback so the TUI can drain keys.
** Returns 1 if quit_save was requested.
*/
static void train_sleep_ms(int ms)
{
	struct timespec ts;

	if (ms <= 0)
		return;
	ts.tv_sec = ms / 1000;
	ts.tv_nsec = (long)(ms % 1000) * 1000000L;
	nanosleep(&ts, NULL);
}

static int train_poll_control(train_progress *p)
{
	if (!g_train_ctl)
		return 0;
	while (g_train_ctl->pause && !g_train_ctl->quit_save) {
		if (p) {
			p->event = TRAIN_EV_STEP;
			p->lr = train_step_lr(p->lr);
			emit_progress(p);
		}
		train_sleep_ms(50);
	}
	return g_train_ctl->quit_save != 0;
}

static int train_maybe_save(lm *model, lm_adam *opt, const char *ckpt_path,
	int epoch_num, train_progress *p)
{
	char msg[160];

	if (!g_train_ctl || !g_train_ctl->save_now || !ckpt_path || !model
	    || !opt)
		return 0;
	g_train_ctl->save_now = 0;
	if (lm_save_checkpoint(model, opt, epoch_num, 0, ckpt_path) != 0)
		return -1;
	if (p) {
		snprintf(msg, sizeof(msg), "saved checkpoint %s (epoch %d)",
		    ckpt_path, epoch_num);
		p->event = TRAIN_EV_SAVE;
		p->msg = msg;
		p->ckpt_path = ckpt_path;
		emit_progress(p);
		p->msg = NULL;
	}
	return 0;
}

/* Wall time — clock() freezes while waiting on CUDA. */
static double wall_sec(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
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
	char line[160];
	int nline;
	double tok_sec;
	double tflops;
	size_t fr;
	size_t tot;
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
	tok_sec = -1.0;
	tflops = -1.0;
	if (done > 0 && elapsed > 0.0) {
		eta = elapsed * (double)(total - done) / (double)done;
		if (tokens_done > 0) {
			ms_tok = 1000.0 * elapsed / (double)tokens_done;
			tok_sec = (double)tokens_done / elapsed;
			/* Rough decoder train FLOPs ~ 6 * N_params * tokens. */
			tflops = 6.0 * (double)(LM_N_BLOCKS
			    * (12L * LM_D_MODEL * LM_D_MODEL
			    + 2L * LM_D_MODEL * LM_D_FF))
			    * tok_sec / 1e12;
		}
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
	if (tok_sec > 0.0 && (size_t)nline < sizeof(line))
		nline += snprintf(line + nline, sizeof(line) - (size_t)nline,
		    " %.0ft/s", tok_sec);
	if (tflops > 0.0 && (size_t)nline < sizeof(line))
		nline += snprintf(line + nline, sizeof(line) - (size_t)nline,
		    " ~%.2fTF", tflops);
	fr = 0;
	tot = 0;
	if (gpu_available() && gpu_mem_info(&fr, &tot) == 0 && tot > 0
	    && (size_t)nline < sizeof(line))
		nline += snprintf(line + nline, sizeof(line) - (size_t)nline,
		    " %.0fMB", (double)(tot - fr) / 1048576.0);
	if (eta >= 0.0 && (size_t)nline < sizeof(line))
		snprintf(line + nline, sizeof(line) - (size_t)nline,
		    " ETA %s", eta_buf);
	/* Pad/clear leftovers from a longer previous frame. */
	fprintf(stderr, "\r%-160s", line);
	fflush(stderr);
	g_progress_open = 1;
	if (force)
		dbg_progress_close();
}


/* ---- Token-parallel packing -------------------------------------------- */

static int g_grad_accum = 1;
static double g_vram_frac = 0.85;

void lm_train_set_grad_accum(int k)
{
	g_grad_accum = k < 1 ? 1 : k;
}

void lm_train_set_vram_frac(double f)
{
	g_vram_frac = (f > 0.0 && f <= 1.0) ? f : 0.85;
}

/*
** One packed microbatch shape: a cache for exactly B sequences plus its
** (B*T) x D dhidden. The last, shorter microbatch of an epoch uses a second
** "tail" slot, so no shape is ever re-allocated inside the step loop.
*/
typedef struct	pack_slot {
	int		B;
	lm_cache	*c;
	mat		*dh;
} pack_slot;

typedef struct	pack_ctx {
	int		Bmax;
	int		T;
	int		*ids;		/* Bmax * T (E4 steady arena) */
	int		ids_pinned;	/* E5: cudaHostAlloc */
	pack_slot	slot[2];	/* [0] = Bmax, [1] = tail */
} pack_ctx;

static void pack_slot_free(pack_slot *s)
{
	lm_cache_free(&s->c);
	mat_free(&s->dh);
	s->B = 0;
}

static void pack_ctx_free(pack_ctx *p)
{
	if (!p)
		return;
	if (p->ids) {
		if (p->ids_pinned)
			gpu_host_free(p->ids);
		else
			free(p->ids);
	}
	p->ids = NULL;
	p->ids_pinned = 0;
	pack_slot_free(&p->slot[0]);
	pack_slot_free(&p->slot[1]);
}

static int pack_slot_get(pack_slot *s, int B, int T)
{
	if (s->B == B && s->c && s->dh)
		return 0;
	pack_slot_free(s);
	s->c = lm_cache_create_packed(B, T);
	s->dh = mat_create(B * T, LM_D_MODEL);
	if (!s->c || !s->dh) {
		pack_slot_free(s);
		return -1;
	}
	mat_try_pin(s->dh);
	s->B = B;
	return 0;
}

static int pack_ctx_init_once(pack_ctx *p, int Bmax, int T)
{
	size_t nbytes;

	memset(p, 0, sizeof(*p));
	p->Bmax = Bmax;
	p->T = T;
	nbytes = (size_t)Bmax * (size_t)T * sizeof(int);
	p->ids = (int *)gpu_host_alloc(nbytes);
	if (p->ids)
		p->ids_pinned = 1;
	else
		p->ids = malloc(nbytes);
	if (!p->ids || pack_slot_get(&p->slot[0], Bmax, T) != 0) {
		pack_ctx_free(p);
		return -1;
	}
	return 0;
}

/* Halve B until the packed working set can be allocated (host/device OOM). */
static int pack_ctx_init(pack_ctx *p, int *Bmax, int T, int debug)
{
	char msg[96];

	while (pack_ctx_init_once(p, *Bmax, T) != 0) {
		if (*Bmax <= 1)
			return -1;
		*Bmax /= 2;
		if (debug) {
			snprintf(msg, sizeof(msg),
			    "packed alloc failed; falling back to B=%d", *Bmax);
			dbg_stamp(msg);
		}
	}
	return 0;
}

/* Forward + loss + backward for the first b sequences already in p->ids. */
static int pack_run(lm *model, pack_ctx *p, int b)
{
	pack_slot *s;

	s = (b == p->Bmax) ? &p->slot[0] : &p->slot[1];
	if (pack_slot_get(s, b, p->T) != 0)
		return -1;
	if (lm_forward_cache_packed(model, p->ids, b, p->T, s->c) != 0)
		return -1;
	if (lm_loss_backward_packed(model, s->c, p->ids, b, p->T, s->dh) != 0)
		return -1;
	return lm_backward_packed(model, s->dh, s->c);
}

/*
** Resolve --batch: batch_size > 0 is taken as is; <= 0 means auto (VRAM
** heuristic on CUDA, small fixed fallback on CPU). max_b caps it (windows/epoch).
*/
static int resolve_batch(int batch_size, int T, int vocab, int max_b,
	int debug)
{
	size_t fr;
	size_t tot;
	int have;
	int B;
	char msg[192];

	fr = 0;
	tot = 0;
	have = gpu_available() && gpu_mem_info(&fr, &tot) == 0;
	if (batch_size > 0)
		B = batch_size;
	else
		B = lm_batch_auto(T, vocab, have ? fr : 0, g_vram_frac,
		    max_b < LM_BATCH_AUTO_MAX ? max_b : LM_BATCH_AUTO_MAX,
		    8);
	if (max_b > 0 && B > max_b)
		B = max_b;
	if (B < 1)
		B = 1;
	if (debug) {
		if (have)
			snprintf(msg, sizeof(msg),
			    "VRAM free %.0f MB / total %.0f MB (frac %.2f); "
			    "auto cap B<=%d (R3); large fixed --batch halves on OOM (R4)",
			    (double)fr / 1048576.0, (double)tot / 1048576.0,
			    g_vram_frac, LM_BATCH_AUTO_MAX);
		else
			snprintf(msg, sizeof(msg), "VRAM n/a (cpu backend)");
		dbg_stamp(msg);
		if (have && tot >= (size_t)14 * 1024 * 1024 * 1024)
			dbg_stamp("16GB-class card: try --batch auto or --batch 256..1024");
		snprintf(msg, sizeof(msg),
		    "batch %s: B=%d T=%d tokens/step=%d grad_accum=%d "
		    "est activations %.1f MB",
		    batch_size > 0 ? "fixed" : "auto", B, T, B * T,
		    g_grad_accum,
		    (double)lm_packed_bytes(B, T, vocab) / 1048576.0);
		dbg_stamp(msg);
	}
	return B;
}

double lm_nll_windows(const lm *model, const text_dataset *ds, int max_windows)
{
	int ids[LM_MAX_CTX];
	int n;
	int i;
	int nw;
	int lim;
	double sum;

	if (!model || !ds)
		return -1.0;
	nw = text_dataset_n_windows(ds);
	if (nw <= 0)
		return -1.0;
	lim = nw;
	if (max_windows > 0 && max_windows < lim)
		lim = max_windows;
	sum = 0.0;
	for (i = 0; i < lim; i++) {
		if (text_dataset_window(ds, i, ids, &n) != 0)
			return -1.0;
		sum += lm_nll(model, ids, n);
	}
	return sum / (double)lim;
}

double lm_train_corpus(lm *model, text_dataset *ds, lm_adam *opt,
	int epochs, int batch_size, double lr, int debug,
	const char *ckpt_path, int epoch_base,
	int eval_every, int eval_windows)
{
	return lm_train_corpus_val(model, ds, opt, epochs, batch_size, lr, debug,
	    ckpt_path, epoch_base, eval_every, eval_windows, NULL);
}

double lm_train_corpus_val(lm *model, text_dataset *ds, lm_adam *opt,
	int epochs, int batch_size, double lr, int debug,
	const char *ckpt_path, int epoch_base,
	int eval_every, int eval_windows, const text_dataset *val)
{
	int ep;
	int nw;
	int idx;
	int b;
	int n;
	int total_steps;
	int done_steps;
	int tokens_done;
	int epoch_num;
	int *order;
	int tmp;
	int j;
	int r;
	int k;
	int bb;
	int win_tot;
	int ok;
	pack_ctx pk;
	double last;
	double win;
	double t0;
	char msg[192];

	if (!model || !ds || !opt || epochs <= 0 || lr <= 0.0)
		return -1.0;
	if (epoch_base < 0)
		epoch_base = 0;
	nw = text_dataset_n_windows(ds);
	if (nw <= 0)
		return -1.0;
	/* batch_size <= 0: auto (see resolve_batch). */
	batch_size = resolve_batch(batch_size, ds->window, model->vocab_size,
	    nw, debug);
	order = malloc((size_t)nw * sizeof(int));
	if (!order || pack_ctx_init(&pk, &batch_size, ds->window, debug) != 0) {
		free(order);
		return -1.0;
	}
	for (j = 0; j < nw; j++)
		order[j] = j;
	mat_try_pin(model->dw_head);
	lm_weights_to_device(model);	/* G1: twins in sync for the whole run */
	lm_adam_invalidate(opt);	/* A1: re-read weights into d_param once */
	total_steps = epochs * nw;
	done_steps = 0;
	tokens_done = 0;
	if (eval_every < 0)
		eval_every = 1;
	if (eval_windows < 0)
		eval_windows = 0;
	last = lm_nll(model, ds->ids,
	    ds->n_ids < LM_MAX_CTX ? ds->n_ids : LM_MAX_CTX);
	win = -1.0;
	if (eval_every > 0)
		win = lm_nll_windows(model, val ? val : ds, eval_windows);
	t0 = wall_sec();
	{
		train_progress tp;

		memset(&tp, 0, sizeof(tp));
		tp.epochs = epochs;
		tp.nw = nw;
		tp.total_steps = total_steps;
		tp.nll = last;
		tp.win_nll = win;
		tp.val_nll = val ? win : -1.0;
		tp.lr = lr;
		tp.batch = batch_size;
		tp.grad_accum = g_grad_accum;
		tp.ckpt_path = ckpt_path;
		tp.adam_t = opt->t;
		if (debug && !g_progress_fn) {
			snprintf(msg, sizeof(msg),
			    "train loop: %d epochs, %d windows/epoch, batch %d x accum %d, lr %.4g",
			    epochs, nw, batch_size, g_grad_accum, lr);
			dbg_stamp(msg);
			if (win >= 0.0)
				snprintf(msg, sizeof(msg),
				    "start NLL %.4f  window NLL %.4f", last, win);
			else
				snprintf(msg, sizeof(msg),
				    "start NLL %.4f  window NLL skipped", last);
			dbg_stamp(msg);
		}
		for (ep = 0; ep < epochs; ep++) {
			int quit;

			quit = 0;
			if (debug && !g_progress_fn) {
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
				double step_lr;
				double elapsed;

				tp.epoch = ep;
				tp.epoch_done = idx;
				tp.done_steps = done_steps;
				tp.tokens_done = tokens_done;
				tp.nll = last;
				tp.win_nll = win;
				tp.val_nll = val ? win : -1.0;
				tp.lr = train_step_lr(lr);
				tp.adam_t = opt->t;
				elapsed = wall_sec() - t0;
				tp.elapsed_sec = elapsed;
				tp.ms_tok = (tokens_done > 0 && elapsed > 0.0)
				    ? 1000.0 * elapsed / (double)tokens_done
				    : -1.0;
				fill_throughput(&tp, tokens_done, elapsed);
				if (train_poll_control(&tp)) {
					quit = 1;
					break;
				}
				if (train_maybe_save(model, opt, ckpt_path,
				    epoch_base + ep + 1, &tp) != 0) {
					lm_weights_ensure_host(model);
					lm_weights_host_dirty(model);
					free(order);
					pack_ctx_free(&pk);
					return -1.0;
				}
				lm_adam_zero_grad(model);
				/* Defer GEMM D2H until the optimizer step boundary. */
				gpu_set_sync_mode(GPU_SYNC_BATCH);
				/*
				** grad_accum packed microbatches of up to batch_size
				** windows each, summed into the grads before one Adam step.
				*/
				win_tot = 0;
				ok = 1;
				for (k = 0; k < g_grad_accum && idx < nw && ok; k++) {
					bb = nw - idx < batch_size ? nw - idx
					    : batch_size;
					for (j = 0; j < bb; j++) {
						if (text_dataset_window(ds,
						    order[idx + j],
						    &pk.ids[j * ds->window], &n)
						    != 0 || n != ds->window) {
							ok = 0;
							break;
						}
					}
					if (!ok || pack_run(model, &pk, bb) != 0) {
						ok = 0;
						break;
					}
					idx += bb;
					win_tot += bb;
					done_steps += bb;
					tokens_done += bb * ds->window;
				}
				if (!ok) {
					gpu_set_sync_mode(GPU_SYNC_EAGER);
					lm_weights_ensure_host(model);
					lm_weights_host_dirty(model);
					free(order);
					pack_ctx_free(&pk);
					tp.event = TRAIN_EV_FAIL;
					tp.msg = "train step failed";
					emit_progress(&tp);
					return -1.0;
				}
				b = win_tot;
				if (b <= 0) {
					gpu_set_sync_mode(GPU_SYNC_EAGER);
					break;
				}
				/*
				** G3: one device sync per microbatch (not per GEMM),
				** then pull device-authoritative dW to host.
				*/
				if (gpu_available())
					gpu_device_sync();
				gpu_set_sync_mode(GPU_SYNC_EAGER);
				/*
				** A1: Adam gathers device-authoritative dW D2D and
				** leaves weights device-authoritative.
				*/
				step_lr = train_step_lr(lr);
				lm_adam_step_scaled(opt, step_lr, 1.0 / (double)b);
				elapsed = wall_sec() - t0;
				tp.epoch = ep;
				tp.epoch_done = idx;
				tp.done_steps = done_steps;
				tp.tokens_done = tokens_done;
				tp.lr = step_lr;
				tp.adam_t = opt->t;
				tp.elapsed_sec = elapsed;
				tp.ms_tok = (tokens_done > 0 && elapsed > 0.0)
				    ? 1000.0 * elapsed / (double)tokens_done
				    : -1.0;
				fill_throughput(&tp, tokens_done, elapsed);
				tp.event = TRAIN_EV_STEP;
				tp.msg = NULL;
				if (g_progress_fn)
					emit_progress(&tp);
				else if (debug)
					dbg_progress(ep, epochs, done_steps,
					    total_steps, idx, nw, last, win, t0,
					    tokens_done, 0);
				if (g_train_ctl && g_train_ctl->quit_save) {
					quit = 1;
					break;
				}
			}
			/* Epoch boundary: long ctx NLL; optional window subsample. */
			last = lm_nll(model, ds->ids,
			    ds->n_ids < LM_MAX_CTX ? ds->n_ids : LM_MAX_CTX);
			if (eval_every > 0 && ((ep + 1) % eval_every) == 0)
				win = lm_nll_windows(model, val ? val : ds,
				    eval_windows);
			epoch_num = epoch_base + ep + 1;
			if (ckpt_path) {
				if (lm_save_checkpoint(model, opt, epoch_num, 0,
				    ckpt_path) != 0) {
					if (debug && !g_progress_fn)
						dbg_stamp("checkpoint save failed");
					lm_weights_ensure_host(model);
					lm_weights_host_dirty(model);
					free(order);
					pack_ctx_free(&pk);
					tp.event = TRAIN_EV_FAIL;
					tp.msg = "checkpoint save failed";
					emit_progress(&tp);
					return -1.0;
				}
				snprintf(msg, sizeof(msg),
				    "saved checkpoint %s (epoch %d)",
				    ckpt_path, epoch_num);
				if (g_progress_fn) {
					tp.event = TRAIN_EV_SAVE;
					tp.msg = msg;
					tp.nll = last;
					tp.win_nll = win;
					emit_progress(&tp);
					tp.msg = NULL;
				} else if (debug) {
					dbg_stamp(msg);
				}
			}
			tp.epoch = ep;
			tp.epoch_done = nw;
			tp.done_steps = done_steps;
			tp.tokens_done = tokens_done;
			tp.nll = last;
			tp.win_nll = win;
			tp.val_nll = val ? win : -1.0;
			tp.lr = train_step_lr(lr);
			tp.adam_t = opt->t;
			tp.elapsed_sec = wall_sec() - t0;
			tp.ms_tok = (tokens_done > 0 && tp.elapsed_sec > 0.0)
			    ? 1000.0 * tp.elapsed_sec / (double)tokens_done
			    : -1.0;
			tp.event = TRAIN_EV_EPOCH;
			if (win >= 0.0)
				snprintf(msg, sizeof(msg),
				    "epoch %d/%d done  NLL %.4f  window NLL %.4f",
				    ep + 1, epochs, last, win);
			else
				snprintf(msg, sizeof(msg),
				    "epoch %d/%d done  NLL %.4f  window NLL skipped",
				    ep + 1, epochs, last);
			tp.msg = msg;
			if (g_progress_fn) {
				emit_progress(&tp);
			} else if (debug) {
				dbg_progress(ep, epochs, done_steps, total_steps,
				    nw, nw, last, win, t0, tokens_done, 1);
				dbg_stamp(msg);
			}
			tp.msg = NULL;
			if (quit)
				break;
		}
	}
	gpu_set_sync_mode(GPU_SYNC_EAGER);
	lm_weights_ensure_host(model);	/* A1: single D2H at end of training */
	lm_weights_host_dirty(model);	/* host owns weights again */
	{
		train_progress tp;

		memset(&tp, 0, sizeof(tp));
		tp.event = TRAIN_EV_DONE;
		tp.epochs = epochs;
		tp.epoch = ep < epochs ? ep : epochs - 1;
		tp.nw = nw;
		tp.total_steps = total_steps;
		tp.done_steps = done_steps;
		tp.tokens_done = tokens_done;
		tp.nll = last;
		tp.win_nll = win;
		tp.lr = train_step_lr(lr);
		tp.batch = batch_size;
		tp.grad_accum = g_grad_accum;
		tp.ckpt_path = ckpt_path;
		tp.adam_t = opt->t;
		tp.elapsed_sec = wall_sec() - t0;
		tp.ms_tok = (tokens_done > 0 && tp.elapsed_sec > 0.0)
		    ? 1000.0 * tp.elapsed_sec / (double)tokens_done : -1.0;
		tp.msg = "train loop finished";
		emit_progress(&tp);
	}
	if (debug && !g_progress_fn)
		dbg_stamp("train loop finished");
	free(order);
	pack_ctx_free(&pk);
	return last;
}

double lm_nll_stream(const lm *model, text_stream *st, int max_windows)
{
	int ids[LM_MAX_CTX];
	int n;
	int cnt;
	int r;
	double sum;

	if (!model || !st || text_stream_rewind(st) != 0)
		return -1.0;
	sum = 0.0;
	cnt = 0;
	while ((max_windows <= 0 || cnt < max_windows)
	    && (r = text_stream_next(st, ids, &n)) == 1) {
		sum += lm_nll(model, ids, n);
		cnt++;
	}
	if (cnt == 0)
		return -1.0;
	return sum / (double)cnt;
}

double lm_train_stream(lm *model, text_stream *train, lm_adam *opt,
	int epochs, int batch_size, double lr, int debug,
	const char *ckpt_path, int epoch_base,
	int eval_every, int eval_windows, text_stream *val)
{
	int ep;
	int b;
	int r;
	int n;
	int n0;
	int ids0[LM_MAX_CTX];
	int k;
	int bb;
	int T;
	pack_ctx pk;
	int epoch_num;
	int done;
	int ok;
	long steps;
	long tokens;
	double last;
	double win;
	double t0;
	double el;
	char msg[192];

	if (!model || !train || !opt || epochs <= 0 || lr <= 0.0)
		return -1.0;
	if (epoch_base < 0)
		epoch_base = 0;
	if (eval_every < 0)
		eval_every = 1;
	T = text_stream_window(train);
	/* batch_size <= 0: auto (see resolve_batch). */
	batch_size = resolve_batch(batch_size, T, model->vocab_size, 0, debug);
	if (pack_ctx_init(&pk, &batch_size, T, debug) != 0)
		return -1.0;
	mat_try_pin(model->dw_head);
	lm_weights_to_device(model);
	lm_adam_invalidate(opt);	/* A1: re-read weights into d_param once */
	last = -1.0;
	win = -1.0;
	t0 = wall_sec();
	{
		train_progress tp;

		memset(&tp, 0, sizeof(tp));
		tp.epochs = epochs;
		tp.nw = 0;
		tp.total_steps = 0;
		tp.lr = lr;
		tp.batch = batch_size;
		tp.grad_accum = g_grad_accum;
		tp.ckpt_path = ckpt_path;
		tp.adam_t = opt->t;
		if (debug && !g_progress_fn) {
			snprintf(msg, sizeof(msg),
			    "stream train: %d epochs, window %d stride %d, batch %d, lr %.4g",
			    epochs, text_stream_window(train),
			    text_stream_stride(train), batch_size, lr);
			dbg_stamp(msg);
		}
		for (ep = 0; ep < epochs; ep++) {
			int quit;

			quit = 0;
			if ((ep > 0 || text_stream_count(train) > 0)
			    && text_stream_rewind(train) != 0)
				goto fail;
			steps = 0;
			tokens = 0;
			n0 = 0;
			done = 0;
			while (!done) {
				double step_lr;

				tp.epoch = ep;
				tp.epoch_done = (int)text_stream_count(train);
				tp.nw = tp.epoch_done > 0 ? tp.epoch_done : 1;
				tp.done_steps = (int)steps;
				tp.tokens_done = (int)tokens;
				tp.nll = last;
				tp.win_nll = win;
				tp.val_nll = val ? win : -1.0;
				tp.lr = train_step_lr(lr);
				tp.adam_t = opt->t;
				el = wall_sec() - t0;
				tp.elapsed_sec = el;
				tp.ms_tok = (tokens > 0 && el > 0.0)
				    ? 1000.0 * el / (double)tokens : -1.0;
				if (train_poll_control(&tp)) {
					quit = 1;
					break;
				}
				if (train_maybe_save(model, opt, ckpt_path,
				    epoch_base + ep + 1, &tp) != 0)
					goto fail;
				lm_adam_zero_grad(model);
				gpu_set_sync_mode(GPU_SYNC_BATCH);
				b = 0;
				ok = 1;
				for (k = 0; k < g_grad_accum && !done && ok; k++) {
					bb = 0;
					while (bb < batch_size) {
						r = text_stream_next(train,
						    &pk.ids[bb * T], &n);
						if (r < 0)
							ok = 0;
						if (r != 1 || n != T) {
							done = 1;
							break;
						}
						if (n0 == 0) {
							memcpy(ids0, &pk.ids[bb * T],
							    (size_t)n * sizeof(int));
							n0 = n;
						}
						bb++;
					}
					if (!ok)
						break;
					if (bb > 0 && pack_run(model, &pk, bb) != 0) {
						ok = 0;
						done = 1;
						break;
					}
					b += bb;
					tokens += (long)bb * T;
				}
				/* G3: single sync per microbatch. */
				if (gpu_available())
					gpu_device_sync();
				gpu_set_sync_mode(GPU_SYNC_EAGER);
				if (!ok)
					goto fail;
				if (b <= 0)
					break;
				step_lr = train_step_lr(lr);
				lm_adam_step_scaled(opt, step_lr, 1.0 / (double)b);
				steps++;
				el = wall_sec() - t0;
				tp.epoch = ep;
				tp.epoch_done = (int)text_stream_count(train);
				tp.nw = tp.epoch_done > 0 ? tp.epoch_done : 1;
				tp.done_steps = (int)steps;
				tp.tokens_done = (int)tokens;
				tp.lr = step_lr;
				tp.adam_t = opt->t;
				tp.elapsed_sec = el;
				tp.ms_tok = tokens > 0 && el > 0.0
				    ? 1000.0 * el / (double)tokens : -1.0;
				tp.event = TRAIN_EV_STEP;
				tp.msg = NULL;
				if (g_progress_fn) {
					emit_progress(&tp);
				} else if (debug && (steps % 200) == 0) {
					snprintf(msg, sizeof(msg),
					    "epoch %d/%d: %ld windows, %.2f ms/token",
					    ep + 1, epochs, text_stream_count(train),
					    tp.ms_tok > 0.0 ? tp.ms_tok : 0.0);
					dbg_stamp(msg);
				}
				if (g_train_ctl && g_train_ctl->quit_save) {
					quit = 1;
					break;
				}
			}
			if (n0 == 0)
				goto fail;	/* empty stream */
			last = lm_nll(model, ids0, n0);
			if (val && eval_every > 0 && ((ep + 1) % eval_every) == 0)
				win = lm_nll_stream(model, val, eval_windows);
			epoch_num = epoch_base + ep + 1;
			if (ckpt_path && lm_save_checkpoint(model, opt, epoch_num, 0,
			    ckpt_path) != 0)
				goto fail;
			tp.epoch = ep;
			tp.nll = last;
			tp.win_nll = win;
			tp.val_nll = val ? win : -1.0;
			tp.tokens_done = (int)tokens;
			tp.done_steps = (int)steps;
			tp.epoch_done = (int)text_stream_count(train);
			tp.nw = tp.epoch_done > 0 ? tp.epoch_done : 1;
			tp.lr = train_step_lr(lr);
			tp.adam_t = opt->t;
			tp.elapsed_sec = wall_sec() - t0;
			tp.ms_tok = tokens > 0 && tp.elapsed_sec > 0.0
			    ? 1000.0 * tp.elapsed_sec / (double)tokens : -1.0;
			tp.event = TRAIN_EV_EPOCH;
			if (win >= 0.0)
				snprintf(msg, sizeof(msg),
				    "epoch %d/%d done: %ld windows  NLL %.4f  val NLL %.4f",
				    ep + 1, epochs, text_stream_count(train),
				    last, win);
			else
				snprintf(msg, sizeof(msg),
				    "epoch %d/%d done: %ld windows  NLL %.4f",
				    ep + 1, epochs, text_stream_count(train), last);
			tp.msg = msg;
			if (g_progress_fn)
				emit_progress(&tp);
			else if (debug)
				dbg_stamp(msg);
			tp.msg = NULL;
			if (quit)
				break;
		}
		tp.event = TRAIN_EV_DONE;
		tp.msg = "train loop finished";
		tp.nll = last;
		emit_progress(&tp);
	}
	lm_weights_ensure_host(model);
	lm_weights_host_dirty(model);
	pack_ctx_free(&pk);
	return last;
fail:
	gpu_set_sync_mode(GPU_SYNC_EAGER);
	lm_weights_ensure_host(model);
	lm_weights_host_dirty(model);
	{
		train_progress tp;

		memset(&tp, 0, sizeof(tp));
		tp.event = TRAIN_EV_FAIL;
		tp.msg = "stream train failed";
		emit_progress(&tp);
	}
	pack_ctx_free(&pk);
	return -1.0;
}
