/*
**	Wall-clock LM forward / train microbenchmarks
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _WIN32
#include "ann_posix.h"
#else
#include <sys/time.h>
#endif
#include "gpu.h"
#include "gradutil.h"
#include "lm_adam.h"
#include "lm_bench.h"
#include "tokenize.h"

/* clock() freezes while waiting on CUDA — use wall time. */
static double wall_sec(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

double lm_bench_forward_ms(lm *model, int n, int iters)
{
	lm_cache *c;
	int *ids;
	int i;
	double t0;
	double t1;
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
	t0 = wall_sec();
	for (i = 0; i < iters; i++) {
		if (lm_forward_cache(model, ids, n, c) != 0) {
			free(ids);
			lm_cache_free(&c);
			return -1.0;
		}
	}
	t1 = wall_sec();
	ms = 1000.0 * (t1 - t0) / (double)iters;
	free(ids);
	lm_cache_free(&c);
	return ms;
}

double lm_bench_train_ms(lm *model, lm_adam *opt, int n, int n_steps)
{
	lm_cache *c;
	mat *dhidden;
	int *ids;
	int i;
	double t0;
	double t1;
	double ms;

	if (!model || !opt || n < 2 || n_steps <= 0)
		return -1.0;
	if (n > LM_MAX_CTX)
		n = LM_MAX_CTX;
	ids = calloc((size_t)n, sizeof(int));
	c = lm_cache_create(n);
	dhidden = mat_create(n, LM_D_MODEL);
	if (!ids || !c || !dhidden) {
		free(ids);
		lm_cache_free(&c);
		mat_free(&dhidden);
		return -1.0;
	}
	for (i = 0; i < n; i++)
		ids[i] = (i * 17 + 3) % model->vocab_size;
	mat_try_pin(dhidden);
	if (gpu_available()) {
		lm_weights_to_device(model);
		lm_adam_invalidate(opt);
		gpu_set_sync_mode(GPU_SYNC_BATCH);
	}
	/* Warmup one full train step. */
	lm_adam_zero_grad(model);
	if (lm_forward_cache(model, ids, n, c) != 0
	    || lm_loss_backward(model, c, ids, n, dhidden, NULL) != 0
	    || lm_backward(model, dhidden, c) != 0) {
		gpu_set_sync_mode(GPU_SYNC_EAGER);
		free(ids);
		lm_cache_free(&c);
		mat_free(&dhidden);
		return -1.0;
	}
	if (gpu_available())
		gpu_device_sync();
	lm_adam_step(opt, 0.01);
	t0 = wall_sec();
	for (i = 0; i < n_steps; i++) {
		lm_adam_zero_grad(model);
		if (gpu_available())
			gpu_set_sync_mode(GPU_SYNC_BATCH);
		if (lm_forward_cache(model, ids, n, c) != 0
		    || lm_loss_backward(model, c, ids, n, dhidden, NULL) != 0
		    || lm_backward(model, dhidden, c) != 0) {
			gpu_set_sync_mode(GPU_SYNC_EAGER);
			free(ids);
			lm_cache_free(&c);
			mat_free(&dhidden);
			return -1.0;
		}
		if (gpu_available()) {
			gpu_device_sync();
			gpu_set_sync_mode(GPU_SYNC_EAGER);
		}
		lm_adam_step(opt, 0.01);
	}
	t1 = wall_sec();
	ms = 1000.0 * (t1 - t0) / (double)n_steps;
	gpu_set_sync_mode(GPU_SYNC_EAGER);
	lm_weights_ensure_host(model);	/* A1: Adam leaves weights on device */
	free(ids);
	lm_cache_free(&c);
	mat_free(&dhidden);
	return ms;
}
