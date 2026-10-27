/*
**	Optional progress / control hooks for LM corpus training
*/
#ifndef TRAIN_PROGRESS_H__
#define TRAIN_PROGRESS_H__

typedef enum	train_event {
	TRAIN_EV_STEP = 0,
	TRAIN_EV_EPOCH,
	TRAIN_EV_SAVE,
	TRAIN_EV_DONE,
	TRAIN_EV_FAIL
} train_event;

typedef struct	train_progress {
	train_event	event;
	int		epoch;
	int		epochs;
	int		epoch_done;
	int		nw;
	int		done_steps;
	int		total_steps;
	long		tokens_done;
	double		nll;
	double		win_nll;
	double		val_nll;
	double		ms_tok;
	double		tok_sec;	/* E10: tokens / wall second */
	double		tflops_est;	/* E10: rough train FLOP/s / 1e12 */
	double		vram_used_mb;	/* E10: (total-free) MB; <0 if n/a */
	double		elapsed_sec;
	double		lr;
	int		batch;
	int		grad_accum;
	const char	*ckpt_path;
	int		adam_t;
	const char	*msg;
} train_progress;

/*
** Cooperative controls. pause: spin between microbatches.
** quit_save: finish the current step, save if possible, return.
** save_now: write a checkpoint at the next microbatch boundary.
** lr: mutable learning rate (<=0 means keep the train-loop argument).
*/
typedef struct	train_control {
	volatile int	pause;
	volatile int	quit_save;
	volatile int	save_now;
	volatile double	lr;
} train_control;

typedef void	(*train_progress_fn)(const train_progress *p, void *userdata);

void	lm_train_set_progress(train_progress_fn fn, void *userdata);
void	lm_train_set_control(train_control *ctl);	/* NULL = ignore */

#endif
