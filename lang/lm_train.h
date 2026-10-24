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
/* R9/S11: block activation checkpointing (recompute in bwd). */
void	lm_train_set_act_checkpoint(int on);
/* E8: emit/pad short EOF windows and mask pad positions in CE (+ fused attn). */
void	lm_train_set_varlen(int on);
/* S4/R12: capture/replay packed fwd+bwd when shapes are steady (0 = off). */
void	lm_train_set_cuda_graphs(int on);
/* S5a/S5b: GPU_PREC_*; FP16 enables dynamic loss scaling. */
void	lm_train_set_compute_prec(int prec);
void	lm_train_set_fp8(int on);
void	lm_train_set_fp8_dgrad(int on);
double	lm_train_loss_scale(void);	/* current scale (1.0 if not FP16) */
/* S7: Flash-style attn (no persistent B·H·T·T weights). */
void	lm_train_set_flash_attn(int on);
/* S13: optional CPU offload of Adam m/v between steps. */
void	lm_train_set_adam_offload(int on);

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

/*
** lm bench step: one timed full train step on a fixed packed batch.
** ms covers pack_run (fwd, fused CE, bwd) + Adam, device-synced at both ends.
** Phase times are filled only under profile (else, and on graph replay, -1).
** ce is the microbatch mean CE when sampled, else -1.
*/
typedef struct	lm_step_time {
	double	ms;
	double	fwd_ms;
	double	loss_ms;
	double	bwd_ms;
	double	adam_ms;
	double	ce;
} lm_step_time;

/*
** Run warmup + steps packed train steps on ids[*B * T] (the same batch every
** step) through the production pack_run path. *B shrinks if the packed working
** set does not fit. Fills out[0..steps-1]; returns 0, or -1 on failure.
*/
int	lm_train_bench_packed(lm *model, lm_adam *opt, const int *ids, int *B,
		int T, double lr, int warmup, int steps, int profile,
		lm_step_time *out);

#endif
