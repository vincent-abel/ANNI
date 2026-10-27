/*
**	Wall-clock LM forward microbenchmark
*/
#ifndef LM_BENCH_H__
#define LM_BENCH_H__

#include "lm.h"

/* Run iters forwards of length n; return mean milliseconds per forward. */
double	lm_bench_forward_ms(lm *model, int n, int iters);
/*
** Run n_steps train microsteps (forward+loss_bwd+backward+Adam) on synthetic ids.
** Returns mean milliseconds per step, or -1 on failure.
*/
double	lm_bench_train_ms(lm *model, lm_adam *opt, int n, int n_steps);

/*
** lm bench harness (docs/bench-harness.md): times the production packed
** train step, GEMM ceilings and decode, and appends one JSON line per result.
*/
#define LM_BENCH_FP32	0	/* FP32 masters, TF32 GEMMs on CUDA */
#define LM_BENCH_BF16	1
#define LM_BENCH_FP8	2	/* BF16 + FP8 forward/dgrad GEMMs */

typedef struct	lm_bench_cfg {
	int		B;		/* windows per step */
	int		T;		/* window length */
	int		vocab;		/* head rows (16384 = BPE target) */
	int		prec;		/* LM_BENCH_* */
	int		graphs;		/* CUDA Graph replay */
	int		ckpt;		/* activation checkpointing */
	int		offload;	/* Adam m/v CPU offload */
	int		warmup;
	int		steps;
	int		profile;	/* per-phase times (graph replay skips them) */
	int		ceiling;	/* step: measure a GEMM ceiling for MFU */
	int		allow_busy;	/* run even if another process holds VRAM */
	int		seed;
	double		lr;
	const char	*json;		/* append results here (NULL = none) */
	const char	*model_path;	/* decode */
	const char	*prompt;	/* decode */
	int		tokens;		/* decode: new tokens */
	int		runs;		/* decode: repeats */
} lm_bench_cfg;

void	lm_bench_cfg_defaults(lm_bench_cfg *cfg);
/* Model FLOPs per trained token (fwd+bwd): matmuls 6*(P_blocks + D*V) + causal attn 6*L*T*D. */
double	lm_bench_flops_per_token(int T, int vocab);
/* Extra hardware FLOPs per token from activation-checkpoint recompute. */
double	lm_bench_recompute_flops_per_token(int T);
int	lm_bench_step(const lm_bench_cfg *cfg);
int	lm_bench_gemm(const lm_bench_cfg *cfg);
int	lm_bench_decode(const lm_bench_cfg *cfg);
/* Cartesian sweep: comma lists of T, B and prec names ("fp32,bf16,fp8"); graphs "0,1". */
int	lm_bench_sweep(const lm_bench_cfg *cfg, const char *windows,
		const char *batches, const char *precs, const char *graphs);

#endif
