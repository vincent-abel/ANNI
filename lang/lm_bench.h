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

#endif
