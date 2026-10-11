/*
**	Ncurses AnnI chat window
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chat_ui.h"
#include "chat_session.h"
#include "lm_config.h"
#include "lm_info.h"
#include "tower.h"

#ifdef ANN_NCURSES
#include <ncurses.h>

#define CHAT_COMP_N	1024
#define CHAT_REPLY_N	8192
#define CHAT_LINE_MAX	512
#define CHAT_LINES_MAX	256

typedef struct	chat_line {
	char	text[CHAT_LINE_MAX];
	int	is_user;	/* 1 user, 0 assistant, 2 system/status */
} chat_line;

typedef struct	chat_ui {
	chat_ui_cfg	cfg;
	chat_session	sess;
	lm_info		info;
	chat_line	lines[CHAT_LINES_MAX];
	int		n_lines;
	int		scroll;		/* lines scrolled up from bottom */
	char		comp[CHAT_COMP_N];
	int		comp_len;
	int		generating;
	char		status[96];
} chat_ui;

static void ui_push_line(chat_ui *ui, const char *text, int is_user, int cols)
{
	const char *p;
	int width;
	int n;

	if (!ui || !text || cols < 8)
		return;
	width = cols - 2;
	if (width > CHAT_LINE_MAX - 1)
		width = CHAT_LINE_MAX - 1;
	p = text;
	while (*p) {
		if (ui->n_lines >= CHAT_LINES_MAX) {
			memmove(ui->lines, ui->lines + 1,
			    (size_t)(CHAT_LINES_MAX - 1) * sizeof(chat_line));
			ui->n_lines = CHAT_LINES_MAX - 1;
		}
		n = 0;
		while (p[n] && p[n] != '\n' && n < width)
			n++;
		memcpy(ui->lines[ui->n_lines].text, p, (size_t)n);
		ui->lines[ui->n_lines].text[n] = '\0';
		ui->lines[ui->n_lines].is_user = is_user;
		ui->n_lines++;
		p += n;
		if (*p == '\n')
			p++;
	}
	ui->scroll = 0;
}

static void chat_draw(chat_ui *ui)
{
	int rows;
	int cols;
	int y;
	int header_h;
	int footer_h;
	int body_h;
	int start;
	int i;
	int max_scroll;
	int ctx;
	const char *path;

	getmaxyx(stdscr, rows, cols);
	if (rows < 8 || cols < 20)
		return;
	erase();
	header_h = 2;
	footer_h = 2;
	body_h = rows - header_h - footer_h;
	if (body_h < 1)
		body_h = 1;

	path = ui->cfg.model_path ? ui->cfg.model_path : "?";
	attron(A_BOLD);
	mvprintw(0, 0, "AnnI chat  %s", path);
	attroff(A_BOLD);
	ctx = chat_session_ctx_used(&ui->sess);
	mvprintw(1, 0, "temp %.2g  top-p %.2g  tokens %d  ctx %d/%d  V=%d  %.1fM",
	    ui->cfg.temperature, ui->cfg.top_p, ui->cfg.tokens,
	    ctx, LM_MAX_CTX, ui->info.vocab_size,
	    (double)ui->info.params / 1e6);
	if (ui->status[0]) {
		attron(A_DIM);
		mvprintw(1, cols > 40 ? cols - 24 : cols / 2, "%s", ui->status);
		attroff(A_DIM);
	}

	max_scroll = ui->n_lines - body_h;
	if (max_scroll < 0)
		max_scroll = 0;
	if (ui->scroll > max_scroll)
		ui->scroll = max_scroll;
	if (ui->scroll < 0)
		ui->scroll = 0;
	start = ui->n_lines - body_h - ui->scroll;
	if (start < 0)
		start = 0;
	y = header_h;
	for (i = start; i < ui->n_lines && y < header_h + body_h; i++, y++) {
		if (ui->lines[i].is_user == 1)
			attron(A_BOLD);
		else if (ui->lines[i].is_user == 2)
			attron(A_DIM);
		mvprintw(y, 0, "%.*s", cols - 1, ui->lines[i].text);
		attroff(A_BOLD);
		attroff(A_DIM);
	}

	mvhline(rows - 2, 0, ACS_HLINE, cols);
	mvprintw(rows - 2, 0, "> %.*s", cols - 3, ui->comp);
	mvprintw(rows - 1, 0,
	    "Enter send  PgUp/PgDn scroll  Esc clear/quit  q quit  Ctrl-L redraw");
	move(rows - 2, 2 + ui->comp_len);
	refresh();
}

static int chat_send(chat_ui *ui, int cols)
{
	char reply[CHAT_REPLY_N];
	char labeled[CHAT_REPLY_N + 32];

	if (ui->comp_len <= 0)
		return 0;
	ui->comp[ui->comp_len] = '\0';
	snprintf(labeled, sizeof(labeled), "User: %s", ui->comp);
	ui_push_line(ui, labeled, 1, cols);
	snprintf(ui->status, sizeof(ui->status), "generating…");
	chat_draw(ui);
	ui->generating = 1;
	if (chat_session_turn(&ui->sess, ui->comp, reply, (int)sizeof(reply))
	    != 0) {
		ui_push_line(ui, "(generate failed)", 2, cols);
		ui->status[0] = '\0';
		ui->generating = 0;
		ui->comp_len = 0;
		ui->comp[0] = '\0';
		return -1;
	}
	snprintf(labeled, sizeof(labeled), "Assistant: %s", reply);
	ui_push_line(ui, labeled, 0, cols);
	ui->status[0] = '\0';
	ui->generating = 0;
	ui->comp_len = 0;
	ui->comp[0] = '\0';
	return 0;
}

int chat_ui_run(const chat_ui_cfg *cfg)
{
	chat_ui ui;
	int ch;
	int rows;
	int cols;

	if (!cfg || !cfg->model || !tower_available())
		return -1;
	memset(&ui, 0, sizeof(ui));
	ui.cfg = *cfg;
	if (chat_session_init(&ui.sess, cfg->model, cfg->tokens,
		cfg->temperature, cfg->top_k, cfg->top_p) != 0)
		return -1;
	lm_info_from_model(&ui.info, cfg->model_path ? cfg->model_path : "",
	    cfg->model, NULL, 0, 0);

	initscr();
	cbreak();
	noecho();
	keypad(stdscr, TRUE);
	curs_set(1);
	getmaxyx(stdscr, rows, cols);
	(void)rows;
	ui_push_line(&ui, "AnnI chat — type a message and press Enter.", 2,
	    cols);

	for (;;) {
		getmaxyx(stdscr, rows, cols);
		(void)rows;
		chat_draw(&ui);
		ch = getch();
		if (ch == KEY_RESIZE)
			continue;
		if (ch == 12)	/* Ctrl-L */
			continue;
		if (ch == KEY_PPAGE) {
			ui.scroll += 4;
			continue;
		}
		if (ch == KEY_NPAGE) {
			ui.scroll -= 4;
			if (ui.scroll < 0)
				ui.scroll = 0;
			continue;
		}
		if (ch == 27) {	/* Esc */
			if (ui.comp_len > 0) {
				ui.comp_len = 0;
				ui.comp[0] = '\0';
			} else
				break;
			continue;
		}
		if ((ch == 'q' || ch == 'Q') && ui.comp_len == 0)
			break;
		if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
			if (ui.comp_len > 0) {
				ui.comp_len--;
				ui.comp[ui.comp_len] = '\0';
			}
			continue;
		}
		if (ch == '\n' || ch == KEY_ENTER || ch == '\r') {
			chat_send(&ui, cols);
			continue;
		}
		if (ch >= 32 && ch < 127 && ui.comp_len < CHAT_COMP_N - 1) {
			ui.comp[ui.comp_len++] = (char)ch;
			ui.comp[ui.comp_len] = '\0';
		}
	}

	endwin();
	chat_session_free(&ui.sess);
	return 0;
}

#else /* !ANN_NCURSES */

int chat_ui_run(const chat_ui_cfg *cfg)
{
	(void)cfg;
	fprintf(stderr, "chat: rebuild with libncurses-dev (ANN_NCURSES)\n");
	return -1;
}

#endif
