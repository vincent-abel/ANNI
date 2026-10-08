/*
**	Wall-clock LM forward microbenchmark
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lm_bench.h"
#include "tokenize.h"

double lm_bench_forward_ms(lm *model, int n, int iters)
{
	lm_cache *c;
	int *ids;
	int i;
	clock_t t0;
	clock_t t1;
	double ms;

	if (!model || n <= 0 || iters <= 0)
		return -1.0;
	if (n > LM_MAX_CTX)
		n = LM_MAX_CTX;
	ids = calloc((size_t)n, sizeof(int));
	c = lm_cache_create(n);
	if (!ids || !c) {
		free(ids);
		lm_cache_free(&c);
		return -1.0;
	}
	for (i = 0; i < n; i++)
		ids[i] = (i * 17 + 3) % model->vocab_size;
	/* Warmup + allocate persistent transformer cache. */
	if (lm_forward_cache(model, ids, n, c) != 0) {
		free(ids);
		lm_cache_free(&c);
		return -1.0;
	}
	t0 = clock();
	for (i = 0; i < iters; i++) {
		if (lm_forward_cache(model, ids, n, c) != 0) {
			free(ids);
			lm_cache_free(&c);
			return -1.0;
		}
	}
	t1 = clock();
	ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC
	    / (double)iters;
	free(ids);
	lm_cache_free(&c);
	return ms;
}
