/*
**	Mini-batch corpus training for the LM
*/
#ifndef LM_TRAIN_H__
#define LM_TRAIN_H__

#include "dataset_stream.h"
#include "dataset_text.h"
#include "lm.h"
#include "lm_adam.h"
#include "train_progress.h"

/*
** Token-parallel packing: batch_size = B parallel sequences per packed
** forward/backward ((B*T) x D GEMMs); batch_size <= 0 (LM_BATCH_AUTO) picks B
** from free VRAM (lm_batch_auto, capped at LM_BATCH_AUTO_MAX). Windows per
** Adam step = B * grad_accum; grads are scaled by 1/windows.
*/
#define LM_BATCH_AUTO		0
#define LM_BATCH_AUTO_MAX	1024	/* R3: fill most of free VRAM on 16GB+ cards */
void	lm_train_set_grad_accum(int k);		/* default 1 */
void	lm_train_set_vram_frac(double f);	/* default 0.85 */
/* E2: warmup steps, then cosine|linear|none over decay_steps (0 = flat). */
void	lm_train_set_lr_schedule(int warmup_steps, int decay_steps,
		const char *kind);
void	lm_train_set_grad_clip(double max_norm);	/* 0 = off */
void	lm_train_set_weight_decay(double wd);		/* AdamW-style; 0 = off */
/* E9: if >0, derive grad_accum from B*T so each Adam ≈ this many tokens. */
void	lm_train_set_tokens_per_step(int n);
/* E15: stop after `patience` evals without val improvement (0 = off). */
void	lm_train_set_early_stop(int patience);
/* E16: abort if train/val NLL rises by more than delta vs best, or non-finite. */
void	lm_train_set_diverge_delta(double delta);

/*
** ckpt_path NULL = no mid-train saves. Otherwise overwrite that path after each
** epoch with epoch_base+1..epoch_base+epochs.
** eval_every: 0 = skip window NLL; 1 = every epoch; N = every N epochs.
** eval_windows: 0 = all windows; M>0 = first M windows only.
*/
double	lm_train_corpus(lm *model, text_dataset *ds, lm_adam *opt,
		int epochs, int batch_size, double lr, int debug,
		const char *ckpt_path, int epoch_base,
		int eval_every, int eval_windows);
/*
** Same as lm_train_corpus, but the per-epoch "window NLL" is measured on `val`
** (held-out) when it is non-NULL instead of on the training windows.
*/
double	lm_train_corpus_val(lm *model, text_dataset *ds, lm_adam *opt,
		int epochs, int batch_size, double lr, int debug,
		const char *ckpt_path, int epoch_base,
		int eval_every, int eval_windows, const text_dataset *val);
/*
** Streaming variant: windows come from `train` (rewound each epoch), so the
** corpus is never held in memory. `val` (may be NULL) is a non-shuffled stream
** used for the per-epoch held-out NLL on up to eval_windows windows (0 = all).
** Returns the NLL of the first window of the last epoch, or <0 on failure.
*/
double	lm_train_stream(lm *model, text_stream *train, lm_adam *opt,
		int epochs, int batch_size, double lr, int debug,
		const char *ckpt_path, int epoch_base,
		int eval_every, int eval_windows, text_stream *val);
/* Mean NLL over a stream (rewinds it first); max_windows<=0 means all. */
double	lm_nll_stream(const lm *model, text_stream *st, int max_windows);
/* Mean NLL over training windows; max_windows<=0 means all. */
double	lm_nll_windows(const lm *model, const text_dataset *ds,
		int max_windows);

#endif
