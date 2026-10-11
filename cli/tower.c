/*
**	Ncurses training control tower / model inspect
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "tower.h"
#include "dataset_text.h"
#include "generate.h"
#include "gpu.h"
#include "lm_config.h"
#include "lm_serialize.h"
#include "lm_train.h"

#ifdef ANN_NCURSES
#include <ncurses.h>

#define TOWER_LOG_N		64
#define TOWER_LOG_W		120
#define TOWER_LOSS_N		80
#define TOWER_SAMPLE_N		512

typedef struct	tower_state {
	tower_train_cfg	cfg;
	train_control	ctl;
	train_progress	last;
	char		log[TOWER_LOG_N][TOWER_LOG_W];
	int		log_i;
	int		log_n;
	double		loss_hist[TOWER_LOSS_N];
	int		loss_n;
	char		sample[TOWER_SAMPLE_N];
	int		active;
	int		want_gen;
	int		ui_ok;
} tower_state;

static tower_state g_tw;

int tower_available(void)
{
	return isatty(STDOUT_FILENO) && isatty(STDIN_FILENO);
}

static void tower_log_line(const char *line)
{
	time_t now;
	struct tm *tm;
	char buf[32];
	char stamped[TOWER_LOG_W];

	if (!line)
		return;
	now = time(NULL);
	tm = localtime(&now);
	if (tm)
		strftime(buf, sizeof(buf), "%H:%M:%S", tm);
	else
		snprintf(buf, sizeof(buf), "??:??:??");
	snprintf(stamped, sizeof(stamped), "[%s] %s", buf, line);
	snprintf(g_tw.log[g_tw.log_i], TOWER_LOG_W, "%s", stamped);
	g_tw.log_i = (g_tw.log_i + 1) % TOWER_LOG_N;
	if (g_tw.log_n < TOWER_LOG_N)
		g_tw.log_n++;
}

static void tower_push_loss(double v)
{
	int i;

	if (v < 0.0)
		return;
	if (g_tw.loss_n < TOWER_LOSS_N) {
		g_tw.loss_hist[g_tw.loss_n++] = v;
		return;
	}
	for (i = 1; i < TOWER_LOSS_N; i++)
		g_tw.loss_hist[i - 1] = g_tw.loss_hist[i];
	g_tw.loss_hist[TOWER_LOSS_N - 1] = v;
}

static void tower_do_generate(void)
{
	const char *prompt;
	int n;

	if (!g_tw.cfg.model)
		return;
	prompt = g_tw.cfg.prompt && g_tw.cfg.prompt[0]
	    ? g_tw.cfg.prompt : "The ";
	lm_weights_ensure_host(g_tw.cfg.model);
	n = lm_generate(g_tw.cfg.model, prompt, g_tw.sample,
	    (int)sizeof(g_tw.sample), 32, 0.8, 40, 0.9);
	if (n < 0)
		snprintf(g_tw.sample, sizeof(g_tw.sample), "(generate failed)");
	tower_log_line("generated sample");
}

static void draw_bar(int y, int x, int width, double frac, const char *label)
{
	int filled;
	int i;

	if (width < 4)
		width = 4;
	if (frac < 0.0)
		frac = 0.0;
	if (frac > 1.0)
		frac = 1.0;
	filled = (int)(frac * (double)width + 0.5);
	mvprintw(y, x, "%s [", label);
	for (i = 0; i < width; i++)
		addch(i < filled ? ACS_CKBOARD : ACS_HLINE);
	printw("] %5.1f%%", 100.0 * frac);
}

static void draw_sparkline(int y, int x, int width)
{
	static const char *blocks = " .:-=+*#%@";
	int i;
	int idx;
	double mn;
	double mx;
	double v;
	int bi;

	mn = 1e300;
	mx = -1e300;
	for (i = 0; i < g_tw.loss_n; i++) {
		if (g_tw.loss_hist[i] < mn)
			mn = g_tw.loss_hist[i];
		if (g_tw.loss_hist[i] > mx)
			mx = g_tw.loss_hist[i];
	}
	mvprintw(y, x, "loss ");
	if (g_tw.loss_n == 0) {
		printw("(no samples yet)");
		return;
	}
	if (mx <= mn)
		mx = mn + 1e-6;
	for (i = 0; i < width && i < g_tw.loss_n; i++) {
		idx = g_tw.loss_n - width + i;
		if (idx < 0)
			idx = i;
		v = (g_tw.loss_hist[idx] - mn) / (mx - mn);
		bi = (int)(v * 9.0 + 0.5);
		if (bi < 0)
			bi = 0;
		if (bi > 9)
			bi = 9;
		addch((unsigned int)blocks[bi]);
	}
	printw("  last %.3f", g_tw.loss_hist[g_tw.loss_n - 1]);
}

static void tower_draw(void)
{
	int rows;
	int cols;
	int y;
	int i;
	int li;
	int params;
	size_t fr;
	size_t tot;
	const train_progress *p;
	double ep_frac;
	double win_frac;
	char eta[16];
	int h;
	int m;
	int s;
	double remain;

	getmaxyx(stdscr, rows, cols);
	erase();
	p = &g_tw.last;
	params = lm_param_count(g_tw.cfg.model);

	attron(A_BOLD);
	mvprintw(0, 0, "ANN training tower");
	attroff(A_BOLD);
	mvprintw(1, 0, "data %s  out %s",
	    g_tw.cfg.data ? g_tw.cfg.data : "-",
	    g_tw.cfg.out ? g_tw.cfg.out : "-");
	mvprintw(2, 0,
	    "d=%d heads=%d ff=%d ctx=%d blocks=%d vocab=%d  params=%d (%.2fM)  %s",
	    LM_D_MODEL, LM_N_HEADS, LM_D_FF, LM_MAX_CTX, LM_N_BLOCKS,
	    g_tw.cfg.model ? g_tw.cfg.model->vocab_size : 0,
	    params, (double)params / 1e6,
	    g_tw.cfg.model && g_tw.cfg.model->bpe ? "bpe" : "byte");

	y = 4;
	ep_frac = (p->epochs > 0)
	    ? ((double)p->epoch + (p->nw > 0
		? (double)p->epoch_done / (double)p->nw : 0.0))
		/ (double)p->epochs
	    : 0.0;
	win_frac = (p->nw > 0) ? (double)p->epoch_done / (double)p->nw : 0.0;
	draw_bar(y++, 0, 28, ep_frac, "epoch");
	draw_bar(y++, 0, 28, win_frac, "window");
	mvprintw(y++, 0,
	    "ep %d/%d  windows %d/%d  steps %d/%d  tokens %d",
	    p->epoch + 1, p->epochs > 0 ? p->epochs : 0,
	    p->epoch_done, p->nw, p->done_steps, p->total_steps,
	    p->tokens_done);
	mvprintw(y++, 0, "NLL %.4f  win %.4f  val %.4f  lr %.5g  adam_t %d",
	    p->nll, p->win_nll, p->val_nll, p->lr, p->adam_t);
	remain = -1.0;
	if (p->done_steps > 0 && p->total_steps > p->done_steps
	    && p->elapsed_sec > 0.0)
		remain = p->elapsed_sec
		    * (double)(p->total_steps - p->done_steps)
		    / (double)p->done_steps;
	if (remain >= 0.0) {
		s = (int)(remain + 0.5);
		h = s / 3600;
		s %= 3600;
		m = s / 60;
		s %= 60;
		if (h > 0)
			snprintf(eta, sizeof(eta), "%dh%02dm", h, m);
		else
			snprintf(eta, sizeof(eta), "%02d:%02d", m, s);
	} else {
		snprintf(eta, sizeof(eta), "--:--");
	}
	mvprintw(y++, 0, "speed %.2f ms/tok  %.0f t/s  ~%.2f TF  VRAM %.0f MB",
	    p->ms_tok, p->tok_sec > 0.0 ? p->tok_sec : 0.0,
	    p->tflops_est > 0.0 ? p->tflops_est : 0.0,
	    p->vram_used_mb > 0.0 ? p->vram_used_mb : 0.0);
	mvprintw(y++, 0, "elapsed %.1fs  ETA %s  %s",
	    p->elapsed_sec, eta,
	    g_tw.ctl.pause ? "PAUSED" : "running");

	fr = tot = 0;
	mvprintw(y++, 0, "gpu %s", gpu_backend_name());
	if (gpu_mem_info(&fr, &tot) == 0 && tot > 0)
		printw("  VRAM %.0f / %.0f MB free/total",
		    (double)fr / (1024.0 * 1024.0),
		    (double)tot / (1024.0 * 1024.0));

	y++;
	draw_sparkline(y++, 0, cols > 20 ? cols - 20 : 40);

	y++;
	attron(A_BOLD);
	mvprintw(y++, 0, "log");
	attroff(A_BOLD);
	for (i = 0; i < 8 && i < g_tw.log_n && y < rows - 8; i++) {
		li = (g_tw.log_i - g_tw.log_n + i + TOWER_LOG_N) % TOWER_LOG_N;
		mvprintw(y++, 0, "%.*s", cols - 1, g_tw.log[li]);
	}

	y++;
	attron(A_BOLD);
	mvprintw(y++, 0, "sample");
	attroff(A_BOLD);
	if (g_tw.sample[0])
		mvprintw(y++, 0, "%.*s", cols - 1, g_tw.sample);
	else
		mvprintw(y++, 0, "(press g to generate)");

	mvprintw(rows - 1, 0,
	    "spc pause  +/- lr  g generate  s save  q quit+save");
	refresh();
}

static void tower_handle_key(int ch)
{
	if (ch == ERR)
		return;
	if (ch == ' ') {
		g_tw.ctl.pause = !g_tw.ctl.pause;
		tower_log_line(g_tw.ctl.pause ? "paused" : "resumed");
	} else if (ch == '+' || ch == '=') {
		if (g_tw.ctl.lr <= 0.0)
			g_tw.ctl.lr = g_tw.cfg.lr > 0.0 ? g_tw.cfg.lr : 0.001;
		g_tw.ctl.lr *= 1.5;
		tower_log_line("lr * 1.5");
	} else if (ch == '-' || ch == '_') {
		if (g_tw.ctl.lr <= 0.0)
			g_tw.ctl.lr = g_tw.cfg.lr > 0.0 ? g_tw.cfg.lr : 0.001;
		g_tw.ctl.lr /= 1.5;
		if (g_tw.ctl.lr < 1e-8)
			g_tw.ctl.lr = 1e-8;
		tower_log_line("lr / 1.5");
	} else if (ch == 'g' || ch == 'G') {
		g_tw.want_gen = 1;
	} else if (ch == 's' || ch == 'S') {
		g_tw.ctl.save_now = 1;
		tower_log_line("save requested");
	} else if (ch == 'q' || ch == 'Q') {
		g_tw.ctl.quit_save = 1;
		g_tw.ctl.pause = 0;
		tower_log_line("quit+save requested");
	}
}

static void tower_progress_cb(const train_progress *p, void *userdata)
{
	int ch;
	double loss;

	(void)userdata;
	if (!p || !g_tw.ui_ok)
		return;
	g_tw.last = *p;
	if (p->event == TRAIN_EV_EPOCH || p->event == TRAIN_EV_STEP) {
		loss = p->val_nll >= 0.0 ? p->val_nll : p->win_nll;
		if (loss < 0.0)
			loss = p->nll;
		if (p->event == TRAIN_EV_EPOCH || (p->done_steps % 8) == 0)
			tower_push_loss(loss);
	}
	if (p->msg && (p->event == TRAIN_EV_EPOCH || p->event == TRAIN_EV_SAVE
	    || p->event == TRAIN_EV_DONE || p->event == TRAIN_EV_FAIL))
		tower_log_line(p->msg);

	if (g_tw.want_gen) {
		g_tw.want_gen = 0;
		tower_do_generate();
	}

	tower_draw();
	while ((ch = getch()) != ERR)
		tower_handle_key(ch);
}

int tower_train_begin(tower_train_cfg *cfg)
{
	if (!cfg || !cfg->model || !tower_available())
		return -1;
	memset(&g_tw, 0, sizeof(g_tw));
	g_tw.cfg = *cfg;
	g_tw.ctl.pause = 0;
	g_tw.ctl.quit_save = 0;
	g_tw.ctl.save_now = 0;
	g_tw.ctl.lr = cfg->lr > 0.0 ? cfg->lr : 0.001;
	g_tw.last.epochs = cfg->epochs;
	g_tw.last.lr = g_tw.ctl.lr;
	g_tw.last.batch = cfg->batch;
	g_tw.last.nll = -1.0;
	g_tw.last.win_nll = -1.0;
	g_tw.last.val_nll = -1.0;

	initscr();
	cbreak();
	noecho();
	keypad(stdscr, TRUE);
	nodelay(stdscr, TRUE);
	curs_set(0);
	g_tw.ui_ok = 1;
	g_tw.active = 1;
	tower_log_line("tower started");
	lm_train_set_control(&g_tw.ctl);
	lm_train_set_progress(tower_progress_cb, NULL);
	tower_draw();
	return 0;
}

void tower_train_end(void)
{
	if (g_tw.active) {
		lm_train_set_progress(NULL, NULL);
		lm_train_set_control(NULL);
		if (g_tw.ui_ok) {
			nodelay(stdscr, FALSE);
			endwin();
		}
	}
	memset(&g_tw, 0, sizeof(g_tw));
}

static void inspect_draw(const lm_info *info, const char *sample, int rows,
	int cols)
{
	int y;

	(void)cols;
	erase();
	attron(A_BOLD);
	mvprintw(0, 0, "ANN model inspect");
	attroff(A_BOLD);
	y = 2;
	mvprintw(y++, 0, "path   %s", info->path);
	if (info->file_bytes > 0)
		mvprintw(y++, 0, "file   %.2f MB",
		    (double)info->file_bytes / 1e6);
	mvprintw(y++, 0, "format %s  real %d",
	    info->magic_kind == LM_INFO_MAGIC_LMCV ? "LMCV" :
	    info->magic_kind == LM_INFO_MAGIC_LMCF ? "LMCF" :
	    info->magic_kind == LM_INFO_MAGIC_LMCK ? "LMCK" : "?",
	    info->real_size);
	mvprintw(y++, 0, "arch   d=%d heads=%d ff=%d ctx=%d blocks=%d",
	    info->d_model, info->n_heads, info->d_ff, info->max_ctx,
	    info->n_blocks);
	if (info->is_bpe)
		mvprintw(y++, 0, "tok    bpe vocab=%d merges=%d",
		    info->vocab_size, info->n_merges);
	else
		mvprintw(y++, 0, "tok    byte vocab=%d", info->vocab_size);
	mvprintw(y++, 0, "params %d (%.3f M)  weights %.2f MB",
	    info->params, (double)info->params / 1e6,
	    (double)info->weight_bytes / 1e6);
	if (info->has_checkpoint)
		mvprintw(y++, 0, "ckpt   epoch=%d step=%d adam_t=%d",
		    info->epoch, info->step, info->adam_t);
	else
		mvprintw(y++, 0, "ckpt   weights only");
	mvprintw(y++, 0, "gpu    %s", info->gpu_name ? info->gpu_name : "?");
	if (info->gpu_total > 0)
		printw("  VRAM %.0f / %.0f MB",
		    (double)info->gpu_free / (1024.0 * 1024.0),
		    (double)info->gpu_total / (1024.0 * 1024.0));
	if (info->prompt_nll >= 0.0)
		mvprintw(y++, 0, "prompt NLL %.4f  PPL %.4f",
		    info->prompt_nll, info->prompt_ppl);
	if (info->data_win_nll >= 0.0)
		mvprintw(y++, 0, "data   win NLL %.4f", info->data_win_nll);
	y++;
	attron(A_BOLD);
	mvprintw(y++, 0, "sample");
	attroff(A_BOLD);
	mvprintw(y++, 0, "%s", sample && sample[0] ? sample : "(empty)");
	mvprintw(rows - 1, 0, "g regenerate   q quit");
	refresh();
}

int tower_inspect(const tower_inspect_cfg *cfg)
{
	lm *model;
	lm_adam *opt;
	lm_info info;
	text_dataset *ds;
	char sample[TOWER_SAMPLE_N];
	const char *prompt;
	int epoch;
	int step;
	int ch;
	int rows;
	int cols;
	int ids[LM_MAX_CTX];
	int n;

	if (!cfg || !cfg->model_path || !tower_available())
		return -1;
	opt = NULL;
	epoch = 0;
	step = 0;
	model = lm_load_checkpoint(cfg->model_path, &opt, &epoch, &step);
	if (!model)
		model = lm_load_any(cfg->model_path);
	if (!model)
		return -1;
	lm_info_from_model(&info, cfg->model_path, model, opt, epoch, step);
	prompt = cfg->prompt && cfg->prompt[0] ? cfg->prompt : "The ";
	n = lm_encode(model, prompt, ids, LM_MAX_CTX);
	if (n >= 2) {
		info.prompt_nll = lm_nll(model, ids, n);
		info.prompt_ppl = lm_perplexity(model, ids, n);
	}
	ds = NULL;
	if (cfg->data) {
		int win;

		win = cfg->window > 1 ? cfg->window : 64;
		if (win > LM_MAX_CTX)
			win = LM_MAX_CTX;
		ds = text_dataset_load_tok(cfg->data, win, model->bpe);
		if (ds)
			info.data_win_nll = lm_nll_windows(model, ds,
			    cfg->eval_windows);
	}
	sample[0] = '\0';
	lm_generate(model, prompt, sample, (int)sizeof(sample), 32, 0.8, 40,
	    0.9);

	initscr();
	cbreak();
	noecho();
	keypad(stdscr, TRUE);
	curs_set(0);
	for (;;) {
		getmaxyx(stdscr, rows, cols);
		inspect_draw(&info, sample, rows, cols);
		ch = getch();
		if (ch == 'q' || ch == 'Q')
			break;
		if (ch == 'g' || ch == 'G') {
			lm_generate(model, prompt, sample, (int)sizeof(sample),
			    32, 0.8, 40, 0.9);
		}
	}
	endwin();
	text_dataset_free(&ds);
	lm_adam_free(&opt);
	lm_free(&model);
	return 0;
}

#else /* !ANN_NCURSES */

int tower_available(void)
{
	return 0;
}

int tower_train_begin(tower_train_cfg *cfg)
{
	(void)cfg;
	fprintf(stderr, "tower: rebuild with libncurses-dev (ANN_NCURSES)\n");
	return -1;
}

void tower_train_end(void)
{
}

int tower_inspect(const tower_inspect_cfg *cfg)
{
	(void)cfg;
	fprintf(stderr, "tower: rebuild with libncurses-dev (ANN_NCURSES)\n");
	return -1;
}

#endif
