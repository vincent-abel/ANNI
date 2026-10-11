/*
**	Ncurses training control tower / model inspect
*/
#ifndef TOWER_H__
#define TOWER_H__

#include "lm.h"
#include "lm_adam.h"
#include "lm_info.h"
#include "train_progress.h"

/* 1 if built with ANN_NCURSES and a TTY is available for the UI. */
int	tower_available(void);

typedef struct	tower_train_cfg {
	lm		*model;
	lm_adam		*opt;
	const char	*data;
	const char	*out;
	const char	*prompt;	/* generate prompt; default "The " */
	double		lr;
	int		epochs;
	int		batch;
	int		window;
	int		epoch_base;
} tower_train_cfg;

/*
** Init ncurses, attach progress/control hooks. Call before lm_train_*.
** Returns 0 on success, -1 if ncurses unavailable.
*/
int	tower_train_begin(tower_train_cfg *cfg);
/* Tear down UI and clear hooks. Safe if begin was never called. */
void	tower_train_end(void);

typedef struct	tower_inspect_cfg {
	const char	*model_path;
	const char	*prompt;
	const char	*data;		/* optional corpus for window NLL */
	int		eval_windows;
	int		window;
} tower_inspect_cfg;

/* Full-screen inspect; blocks until q. Returns 0 on success. */
int	tower_inspect(const tower_inspect_cfg *cfg);

#endif
