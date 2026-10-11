/*
**	Ncurses messenger-style chat UI (AnnI chat)
*/
#ifndef CHAT_UI_H__
#define CHAT_UI_H__

#include "lm.h"

typedef struct	chat_ui_cfg {
	lm		*model;
	const char	*model_path;	/* display only */
	int		tokens;
	double		temperature;
	int		top_k;
	double		top_p;
} chat_ui_cfg;

/*
** Full-screen chat; blocks until quit. Requires TTY + ANN_NCURSES
** (tower_available). Returns 0 on success, -1 if unavailable/fail.
*/
int	chat_ui_run(const chat_ui_cfg *cfg);

#endif
