/*
**	Mini-batch corpus training for the LM
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "attn.h"
#include "block.h"
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
static int g_warmup_steps;
static int g_decay_steps;
static char g_decay_kind[16] = "none";
static double g_grad_clip;
static double g_weight_decay;
static int g_tokens_per_step;
static int g_grad_accum = 1;
static double g_vram_frac = 0.85;
static int g_early_stop_patience;
static double g_diverge_delta;
static int g_act_checkpoint;
static int g_varlen;
static int g_cuda_graphs;
static int g_graph_B;
static int g_graph_T;
static int g_graph_ready;
static int g_compute_prec;
static double g_loss_scale = 1.0;
static int g_loss_scale_good;

void lm_train_set_progress(train_progress_fn fn, void *userdata)
{
	g_progress_fn = fn;
	g_progress_ud = userdata;
}

void lm_train_set_control(train_control *ctl)
{
	g_train_ctl = ctl;
}

static double train_step_lr(double base_lr, int step)
{
	double lr;
	double p;
	int t;

	if (g_train_ctl && g_train_ctl->lr > 0.0)
		return g_train_ctl->lr;
	lr = base_lr;
	if (g_warmup_steps > 0 && step < g_warmup_steps)
		return lr * ((double)step + 1.0) / (double)g_warmup_steps;
	if (g_decay_steps <= 0 || strcmp(g_decay_kind, "none") == 0)
		return lr;
	t = step - g_warmup_steps;
	if (t < 0)
		t = 0;
	if (t > g_decay_steps)
		t = g_decay_steps;
	p = (double)t / (double)g_decay_steps;
	if (strcmp(g_decay_kind, "linear") == 0)
		return lr * (1.0 - p);
	/* cosine */
	return lr * 0.5 * (1.0 + cos(3.14159265358979323846 * p));
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
			p->lr = train_step_lr(p->lr, p->adam_t);
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
	if (total <= 0)
		total = 1;
	if (epoch_done < 0)
		epoch_done = 0;
	/* nw<=0 means unknown stream total — do not fake N/N at 100%. */
	if (nw > 0) {
		if (epoch_done > nw)
			epoch_done = nw;
		ep_pct = 100.0 * (double)epoch_done / (double)nw;
		filled = (int)((double)width * (double)epoch_done
		    / (double)nw + 0.5);
	} else {
		ep_pct = -1.0;
		filled = 0;
	}
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
		if (nw > 0 && total >= done)
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
	if (nw > 0)
		nline = snprintf(line, sizeof(line),
		    "ep %d/%d [%s] %4.0f%% %d/%d",
		    ep + 1, epochs, bar, ep_pct, epoch_done, nw);
	else
		nline = snprintf(line, sizeof(line),
		    "ep %d/%d [%s]    ?%% %d/?",
		    ep + 1, epochs, bar, epoch_done);
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

void lm_train_set_grad_accum(int k)
{
	g_grad_accum = k < 1 ? 1 : k;
}

void lm_train_set_vram_frac(double f)
{
	g_vram_frac = (f > 0.0 && f <= 1.0) ? f : 0.85;
}

void lm_train_set_lr_schedule(int warmup_steps, int decay_steps, const char *kind)
{
	g_warmup_steps = warmup_steps < 0 ? 0 : warmup_steps;
	g_decay_steps = decay_steps < 0 ? 0 : decay_steps;
	if (!kind)
		kind = "none";
	strncpy(g_decay_kind, kind, sizeof(g_decay_kind) - 1);
	g_decay_kind[sizeof(g_decay_kind) - 1] = '\0';
}

void lm_train_set_grad_clip(double max_norm)
{
	g_grad_clip = max_norm > 0.0 ? max_norm : 0.0;
}

void lm_train_set_weight_decay(double wd)
{
	g_weight_decay = wd > 0.0 ? wd : 0.0;
}

void lm_train_set_tokens_per_step(int n)
{
	g_tokens_per_step = n < 0 ? 0 : n;
}

void lm_train_set_early_stop(int patience)
{
	g_early_stop_patience = patience < 0 ? 0 : patience;
}

void lm_train_set_diverge_delta(double delta)
{
	g_diverge_delta = delta > 0.0 ? delta : 0.0;
}

void lm_train_set_act_checkpoint(int on)
{
	g_act_checkpoint = on ? 1 : 0;
	block_set_act_checkpoint(g_act_checkpoint);
}

void lm_train_set_flash_attn(int on)
{
	attn_set_flash(on);
}

void lm_train_set_adam_offload(int on)
{
	lm_adam_set_offload(on);
}

void lm_train_set_varlen(int on)
{
	g_varlen = on ? 1 : 0;
}

void lm_train_set_cuda_graphs(int on)
{
	g_cuda_graphs = on ? 1 : 0;
	g_graph_ready = 0;
	g_graph_B = 0;
	g_graph_T = 0;
	gpu_graph_destroy();
}

void lm_train_set_compute_prec(int prec)
{
	if (prec != GPU_PREC_BF16 && prec != GPU_PREC_FP16)
		prec = GPU_PREC_FP32;
	g_compute_prec = prec;
	gpu_set_compute_prec(prec);
	g_loss_scale = (prec == GPU_PREC_FP16) ? 1024.0 : 1.0;
	g_loss_scale_good = 0;
	/* Dtype change invalidates captured graphs. */
	if (g_cuda_graphs)
		lm_train_set_cuda_graphs(1);
}

double lm_train_loss_scale(void)
{
	return g_loss_scale > 0.0 ? g_loss_scale : 1.0;
}

/* Unscale factor for Adam; update dynamic loss scale for FP16. */
static double train_grad_unscale(int step_ok)
{
	if (g_compute_prec != GPU_PREC_FP16)
		return 1.0;
	if (!step_ok) {
		g_loss_scale *= 0.5;
		if (g_loss_scale < 1.0)
			g_loss_scale = 1.0;
		g_loss_scale_good = 0;
		return 0.0;	/* caller skips Adam */
	}
	g_loss_scale_good++;
	if (g_loss_scale_good >= 2000) {
		g_loss_scale *= 2.0;
		if (g_loss_scale > 65536.0)
			g_loss_scale = 65536.0;
		g_loss_scale_good = 0;
	}
	return 1.0 / g_loss_scale;
}

static void apply_tokens_per_step(int B, int T)
{
	int tok;

	if (g_tokens_per_step <= 0 || B <= 0 || T <= 0)
		return;
	tok = B * T;
	g_grad_accum = (g_tokens_per_step + tok - 1) / tok;
	if (g_grad_accum < 1)
		g_grad_accum = 1;
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
	int		*ids_next;	/* E3: prefetch buffer */
	int		*lens;		/* E8: real length per sequence */
	int		*lens_next;
	int		ids_pinned;	/* E5: cudaHostAlloc */
	int		next_bb;	/* E3: windows waiting in ids_next */
	int		next_done;	/* E3: stream exhausted while prefetching */
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
	if (p->ids_next) {
		if (p->ids_pinned)
			gpu_host_free(p->ids_next);
		else
			free(p->ids_next);
	}
	free(p->lens);
	free(p->lens_next);
	p->ids = NULL;
	p->ids_next = NULL;
	p->lens = NULL;
	p->lens_next = NULL;
	p->ids_pinned = 0;
	p->next_bb = 0;
	p->next_done = 0;
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
	/* S9: ensure non-blocking copy stream exists for async H2D overlap. */
	(void)gpu_copy_stream();
	nbytes = (size_t)Bmax * (size_t)T * sizeof(int);
	p->ids = (int *)gpu_host_alloc(nbytes);
	if (p->ids)
		p->ids_pinned = 1;
	else
		p->ids = malloc(nbytes);
	if (p->ids_pinned)
		p->ids_next = (int *)gpu_host_alloc(nbytes);
	else
		p->ids_next = malloc(nbytes);
	p->lens = malloc((size_t)Bmax * sizeof(int));
	p->lens_next = malloc((size_t)Bmax * sizeof(int));
	if (!p->ids || !p->ids_next || !p->lens || !p->lens_next
	    || pack_slot_get(&p->slot[0], Bmax, T) != 0) {
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

/* Blend microbatch mean CE into a live train-NLL EMA (ema < 0 = unset). */
static void train_ema_ce(double *ema, double ce, int weight)
{
	const double alpha = 0.05;

	if (!ema || weight <= 0 || ce < 0.0 || !isfinite(ce))
		return;
	if (*ema < 0.0)
		*ema = ce;
	else
		*ema = (1.0 - alpha) * (*ema) + alpha * ce;
}

/*
** Default mid-epoch val subsample when --eval-windows is unset (0 = all would
** be too expensive to run 20× per epoch).
*/
#define TRAIN_MID_VAL_DEFAULT_WINS	32

/* Advance *next_pct (5,10,...,95) after a probe; return 1 if due now. */
static int train_mid_val_due(int done, int total, int *next_pct)
{
	int pct;

	if (!next_pct || total <= 0 || *next_pct <= 0 || *next_pct >= 100)
		return 0;
	pct = (int)(100.0 * (double)done / (double)total);
	if (pct < *next_pct)
		return 0;
	while (*next_pct <= pct && *next_pct < 100)
		*next_pct += 5;
	return 1;
}

/* Forward + loss + backward for the first b sequences already in p->ids.
** mean_ce may be NULL; otherwise receives mean CE over predicting rows.
** S4: optional CUDA Graph capture/replay when B,T steady and no varlen lens. */
static int pack_run_body(lm *model, pack_ctx *p, int b, double *mean_ce,
	pack_slot *s, const int *lens)
{
	gpu_attn_set_seqlens(lens);
	if (lm_forward_cache_packed(model, p->ids, b, p->T, s->c) != 0) {
		gpu_attn_set_seqlens(NULL);
		return -1;
	}
	if (lm_loss_backward_packed_lens(model, s->c, p->ids, b, p->T, lens,
	    s->dh, mean_ce) != 0) {
		gpu_attn_set_seqlens(NULL);
		return -1;
	}
	gpu_attn_set_seqlens(NULL);
	return lm_backward_packed(model, s->dh, s->c);
}

static int pack_run(lm *model, pack_ctx *p, int b, double *mean_ce)
{
	pack_slot *s;
	const int *lens;
	int i;
	int need_lens;

	s = (b == p->Bmax) ? &p->slot[0] : &p->slot[1];
	if (pack_slot_get(s, b, p->T) != 0)
		return -1;
	need_lens = 0;
	lens = NULL;
	if (g_varlen && p->lens) {
		for (i = 0; i < b; i++) {
			if (p->lens[i] < p->T) {
				need_lens = 1;
				break;
			}
		}
		if (need_lens)
			lens = p->lens;
	}
	/*
	** Graphs need fixed device addresses and no host ops in the region.
	** Skip when varlen seqlens change the mask, or B/T changes.
	*/
	/* mean_ce unavailable on replay (capture uses NULL); leave unset. */
	if (g_cuda_graphs && gpu_available() && !lens) {
		if (g_graph_ready && g_graph_B == b && g_graph_T == p->T) {
			if (gpu_graph_replay() == 0) {
				if (mean_ce)
					*mean_ce = -1.0;
				return 0;
			}
			g_graph_ready = 0;
			gpu_graph_destroy();
		}
		if ((!g_graph_ready || g_graph_B != b || g_graph_T != p->T)
		    && gpu_graph_begin() == 0) {
			if (pack_run_body(model, p, b, NULL, s, NULL) == 0
			    && gpu_graph_end() == 0) {
				g_graph_ready = 1;
				g_graph_B = b;
				g_graph_T = p->T;
				if (mean_ce)
					*mean_ce = -1.0;
				return 0;
			}
			gpu_graph_destroy();
			g_graph_ready = 0;
		}
	}
	return pack_run_body(model, p, b, mean_ce, s, lens);
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
	/* max_b<=0 (stream path) means no window cap — use LM_BATCH_AUTO_MAX. */
	if (max_b <= 0)
		max_b = LM_BATCH_AUTO_MAX;
	if (batch_size > 0)
		B = batch_size;
	else
		B = lm_batch_auto(T, vocab, have ? fr : 0, g_vram_frac,
		    max_b < LM_BATCH_AUTO_MAX ? max_b : LM_BATCH_AUTO_MAX,
		    8);
	if (B > max_b)
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
	apply_tokens_per_step(batch_size, ds->window);
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
	(void)gpu_copy_stream();	/* R8: create async copy stream if CUDA */
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
		double best_val = -1.0;
		int stall = 0;
		char best_path[640];

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
		best_path[0] = '\0';
		if (ckpt_path)
			snprintf(best_path, sizeof(best_path), "%s.best", ckpt_path);
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
			int next_mid_pct;
			int mid_wins;

			quit = 0;
			next_mid_pct = 5;
			mid_wins = eval_windows > 0 ? eval_windows
			    : TRAIN_MID_VAL_DEFAULT_WINS;
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
				tp.lr = train_step_lr(lr, opt->t);
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
				{
					double ce_sum = 0.0;
					int ce_w = 0;

					for (k = 0; k < g_grad_accum && idx < nw
					    && ok; k++) {
						double mb_ce = -1.0;

						bb = nw - idx < batch_size
						    ? nw - idx : batch_size;
						for (j = 0; j < bb; j++) {
							if (text_dataset_window(ds,
							    order[idx + j],
							    &pk.ids[j * ds->window],
							    &n) != 0
							    || n != ds->window) {
								ok = 0;
								break;
							}
						}
						if (!ok || pack_run(model, &pk, bb,
						    &mb_ce) != 0) {
							ok = 0;
							break;
						}
						if (mb_ce >= 0.0) {
							ce_sum += mb_ce
							    * (double)bb;
							ce_w += bb;
						}
						idx += bb;
						win_tot += bb;
						done_steps += bb;
						tokens_done += bb * ds->window;
					}
					if (ok && ce_w > 0)
						train_ema_ce(&last,
						    ce_sum / (double)ce_w, 1);
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
				step_lr = train_step_lr(lr, opt->t);
				opt->weight_decay = g_weight_decay;
				opt->grad_clip = g_grad_clip;
				{
					double us = train_grad_unscale(
					    isfinite(last));

					if (us > 0.0)
						lm_adam_step_scaled(opt, step_lr,
						    us / (double)b);
					else
						lm_adam_zero_grad(model);
				}
				elapsed = wall_sec() - t0;
				tp.epoch = ep;
				tp.epoch_done = idx;
				tp.done_steps = done_steps;
				tp.tokens_done = tokens_done;
				if (val && eval_every > 0
				    && train_mid_val_due(idx, nw, &next_mid_pct)) {
					double v;
					int pct = (int)(100.0 * (double)idx
					    / (double)nw);

					v = lm_nll_windows(model, val, mid_wins);
					if (v >= 0.0)
						win = v;
					if (debug && !g_progress_fn) {
						snprintf(msg, sizeof(msg),
						    "mid-epoch %d%%  val NLL %.4f"
						    " (%d windows)",
						    pct, win, mid_wins);
						dbg_stamp(msg);
					}
				}
				tp.nll = last;
				tp.win_nll = win;
				tp.val_nll = val ? win : -1.0;
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
			tp.lr = train_step_lr(lr, opt->t);
			tp.adam_t = opt->t;
			tp.elapsed_sec = wall_sec() - t0;
			tp.ms_tok = (tokens_done > 0 && tp.elapsed_sec > 0.0)
			    ? 1000.0 * tp.elapsed_sec / (double)tokens_done
			    : -1.0;
			/* E16 diverge / E15 plateau on val (or train win) NLL. */
			{
				double metric = (val && win >= 0.0) ? win : last;

				if (metric != metric || metric > 1e6) {
					if (debug)
						dbg_stamp("E16 diverge: non-finite NLL");
					quit = 1;
				} else if (g_diverge_delta > 0.0 && best_val >= 0.0
				    && metric > best_val + g_diverge_delta) {
					if (debug)
						dbg_stamp("E16 diverge: NLL rose");
					quit = 1;
				} else if (metric >= 0.0
				    && (best_val < 0.0 || metric < best_val)) {
					best_val = metric;
					stall = 0;
					if (best_path[0])
						(void)lm_save_checkpoint(model, opt,
						    epoch_num, 0, best_path);
				} else if (g_early_stop_patience > 0) {
					stall++;
					if (stall >= g_early_stop_patience) {
						if (debug)
							dbg_stamp("E15 early-stop plateau");
						quit = 1;
					}
				}
			}
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
		tp.lr = train_step_lr(lr, opt->t);
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
	apply_tokens_per_step(batch_size, T);
	if (g_varlen) {
		text_stream_set_varlen(train, 1);
		if (val)
			text_stream_set_varlen(val, 1);
	}
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
			int next_mid_pct;
			int mid_wins;
			double ep_t0;

			quit = 0;
			next_mid_pct = 5;
			mid_wins = eval_windows > 0 ? eval_windows
			    : TRAIN_MID_VAL_DEFAULT_WINS;
			if ((ep > 0 || text_stream_count(train) > 0)
			    && text_stream_rewind(train) != 0)
				goto fail;
			/* Prefetch EOF flag must not carry across epochs. */
			pk.next_bb = 0;
			pk.next_done = 0;
			steps = 0;
			tokens = 0;
			n0 = 0;
			done = 0;
			/* Throughput must use this-epoch wall time (tokens reset). */
			ep_t0 = wall_sec();
			while (!done) {
				double step_lr;

				tp.epoch = ep;
				tp.epoch_done = (int)text_stream_count(train);
				{
					long est = text_stream_est_count(train);

					tp.nw = est > 0 ? (int)est : 0;
					if (tp.nw > 0 && tp.epoch_done > tp.nw)
						tp.nw = tp.epoch_done;
				}
				tp.done_steps = (int)steps;
				tp.tokens_done = (int)tokens;
				tp.nll = last;
				tp.win_nll = win;
				tp.val_nll = val ? win : -1.0;
				tp.lr = train_step_lr(lr, opt->t);
				tp.adam_t = opt->t;
				el = wall_sec() - ep_t0;
				tp.elapsed_sec = wall_sec() - t0;
				tp.ms_tok = (tokens > 0 && el > 0.0)
				    ? 1000.0 * el / (double)tokens : -1.0;
				fill_throughput(&tp, (int)tokens, el);
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
				{
					double ce_sum = 0.0;
					int ce_w = 0;

					for (k = 0; k < g_grad_accum && !done
					    && ok; k++) {
						double mb_ce = -1.0;

						/* E3: use prefetched ids_next when ready. */
						if (pk.next_bb > 0) {
							int *swap = pk.ids;
							int *swap_l = pk.lens;

							pk.ids = pk.ids_next;
							pk.ids_next = swap;
							pk.lens = pk.lens_next;
							pk.lens_next = swap_l;
							bb = pk.next_bb;
							if (pk.next_done)
								done = 1;
							pk.next_bb = 0;
						} else {
							bb = 0;
							while (bb < batch_size) {
								r = text_stream_next(
								    train,
								    &pk.ids[bb * T],
								    &n);
								if (r < 0)
									ok = 0;
								if (r != 1) {
									done = 1;
									break;
								}
								if (n < 2
								    || (!g_varlen
								    && n != T)) {
									done = 1;
									break;
								}
								pk.lens[bb] = n;
								if (n0 == 0) {
									memcpy(ids0,
									    &pk.ids[
									    bb * T],
									    (size_t)n
									    * sizeof(
									    int));
									n0 = n;
								}
								bb++;
							}
						}
						if (!ok)
							break;
						if (bb > 0 && pack_run(model, &pk,
						    bb, &mb_ce) != 0) {
							ok = 0;
							done = 1;
							break;
						}
						if (bb > 0 && mb_ce >= 0.0) {
							ce_sum += mb_ce
							    * (double)bb;
							ce_w += bb;
						}
						/*
						** E3: after GPU pack_run returns
						** (BATCH may still be draining), fill
						** the next ids buffer on CPU before
						** the forced device sync below.
						*/
						if (!done && ok && pk.ids_next) {
							int nb = 0;
							int nn;

							pk.next_bb = 0;
							while (nb < batch_size) {
								r = text_stream_next(
								    train,
								    &pk.ids_next[
								    nb * T], &nn);
								if (r < 0)
									ok = 0;
								if (r != 1
								    || nn < 2
								    || (!g_varlen
								    && nn != T)) {
									pk.next_done
									    = 1;
									break;
								}
								pk.lens_next[nb]
								    = nn;
								nb++;
							}
							pk.next_bb = nb;
						}
						b += bb;
						if (g_varlen) {
							int ti;

							for (ti = 0; ti < bb; ti++)
								tokens +=
								    pk.lens[ti];
						} else {
							tokens += (long)bb * T;
						}
					}
					if (ok && ce_w > 0)
						train_ema_ce(&last,
						    ce_sum / (double)ce_w, 1);
				}
				/* G3: single sync per microbatch. */
				if (gpu_available())
					gpu_device_sync();
				gpu_set_sync_mode(GPU_SYNC_EAGER);
				if (!ok)
					goto fail;
				if (b <= 0)
					break;
				step_lr = train_step_lr(lr, opt->t);
				opt->weight_decay = g_weight_decay;
				opt->grad_clip = g_grad_clip;
				{
					double us = train_grad_unscale(
					    isfinite(last));

					if (us > 0.0)
						lm_adam_step_scaled(opt, step_lr,
						    us / (double)b);
					else
						lm_adam_zero_grad(model);
				}
				steps++;
				el = wall_sec() - ep_t0;
				tp.epoch = ep;
				tp.epoch_done = (int)text_stream_count(train);
				{
					long est = text_stream_est_count(train);

					tp.nw = est > 0 ? (int)est : 0;
					if (tp.nw > 0 && tp.epoch_done > tp.nw)
						tp.nw = tp.epoch_done;
				}
				tp.done_steps = (int)steps;
				tp.tokens_done = (int)tokens;
				/*
				** Mid-epoch val subsample every ~5% (cheap vs train;
				** ~1–2% wall with M=32). Skip 100% — epoch-end does it.
				*/
				if (val && eval_every > 0
				    && train_mid_val_due(tp.epoch_done, tp.nw,
				    &next_mid_pct)) {
					double v;
					int pct = tp.nw > 0
					    ? (int)(100.0
					    * (double)tp.epoch_done
					    / (double)tp.nw) : 0;

					v = lm_nll_stream(model, val, mid_wins);
					if (v >= 0.0)
						win = v;
					if (debug && !g_progress_fn) {
						snprintf(msg, sizeof(msg),
						    "mid-epoch %d%%  val NLL %.4f"
						    " (%d windows)",
						    pct, win, mid_wins);
						dbg_stamp(msg);
					}
				}
				tp.nll = last;
				tp.win_nll = win;
				tp.val_nll = val ? win : -1.0;
				tp.lr = step_lr;
				tp.adam_t = opt->t;
				tp.elapsed_sec = wall_sec() - t0;
				tp.ms_tok = tokens > 0 && el > 0.0
				    ? 1000.0 * el / (double)tokens : -1.0;
				tp.event = TRAIN_EV_STEP;
				tp.msg = NULL;
				if (g_progress_fn) {
					fill_throughput(&tp, (int)tokens, el);
					emit_progress(&tp);
				} else if (debug) {
					/* Live \\r line (throttled); not every-200 stamps. */
					int win_done = tp.epoch_done;
					int win_total = tp.nw;
					int step_total;

					if (win_total > 0)
						step_total = win_total
						    / (batch_size > 0
						    ? batch_size : 1);
					else
						step_total = 0;
					if (step_total < 1 && win_total > 0)
						step_total = 1;
					fill_throughput(&tp, (int)tokens, el);
					dbg_progress(ep, epochs, (int)steps,
					    step_total > 0 ? step_total
					    : (int)steps, win_done, win_total,
					    last, win, ep_t0, (int)tokens, 0);
				}
				if (g_train_ctl && g_train_ctl->quit_save) {
					quit = 1;
					break;
				}
			}
			if (n0 == 0)
				goto fail;	/* empty stream */
			/* Keep live train CE EMA; fallback probe only if no steps. */
			if (last < 0.0 && n0 > 0)
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
			{
				long est = text_stream_est_count(train);

				/* Prefer exact count just finished this epoch. */
				tp.nw = tp.epoch_done > 0 ? tp.epoch_done
				    : (est > 0 ? (int)est : 0);
			}
			tp.lr = train_step_lr(lr, opt->t);
			tp.adam_t = opt->t;
			el = wall_sec() - ep_t0;
			tp.elapsed_sec = wall_sec() - t0;
			tp.ms_tok = tokens > 0 && el > 0.0
			    ? 1000.0 * el / (double)tokens : -1.0;
			fill_throughput(&tp, (int)tokens, el);
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
