/*
**	Wall-clock LM forward microbenchmark
*/
#ifndef LM_BENCH_H__
#define LM_BENCH_H__

#include "lm.h"

/* Run iters forwards of length n; return mean milliseconds per forward. */
double	lm_bench_forward_ms(lm *model, int n, int iters);

#endif
