/*
**	GPU regression checks on the small test LM (make test-gpu, needs CUDA).
**	Each check is a bug that make test (CPU stub) cannot see:
**	  1. forward determinism and no cross-batch state (stale CE targets)
**	  2. first Adam step applies the weight gradients (flat alias setup)
**	  3. bias / LN vectors trained and refreshed on the device
**	  4. a freed model leaves no pending CUDA error (aliased dW frees)
**	  5. BF16 gradients track FP32 from step 1 (cuBLASLt timing loop)
*/
#define _POSIX_C_SOURCE 200112L
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "attn.h"
#include "gpu.h"
#include "gradutil.h"
#include "lm.h"
#include "lm_adam.h"

#define B 4
#define T 32
#define ACC 2
#define STEPS 4
#define NT 64
/* Pseudo precision for rig_open: BF16 GEMMs with the FP8 fwd/dgrad stack. */
#define PREC_FP8 99

static void fill_ids(int *ids, unsigned x)
{
	int j;

	for (j = 0; j < B * T; j++) {
		x = x * 1103515245u + 12345u;
		ids[j] = (int)((x >> 16) % 256u);
	}
}

static double abs_sum_vec(const vec *v)
{
	int i;
	double s = 0.0;

	for (i = 0; i < v->n; i++)
		s += fabs((double)v->data[i]);
	return s;
}

typedef struct {
	lm		*m;
	lm_adam		*opt;
	lm_cache	*c;
	mat		*dh;
} rig;

static void rig_open(rig *r, int prec)
{
	srand(1234);
	r->m = lm_create();
	r->opt = lm_adam_create(r->m);
	assert(r->m && r->opt);
	r->opt->beta2 = 0.95;
	r->opt->grad_clip = 1.0;
	r->opt->weight_decay = 0.1;
	gpu_set_fp8(prec == PREC_FP8);
	gpu_set_compute_prec(prec == PREC_FP8 ? GPU_PREC_BF16 : prec);
	attn_set_flash(1);
	lm_weights_to_device(r->m);
	r->c = lm_cache_create_packed(B, T);
	r->dh = mat_create(B * T, LM_D_MODEL);
	assert(r->c && r->dh);
	gpu_set_sync_mode(GPU_SYNC_BATCH);
}

static void rig_close(rig *r)
{
	gpu_set_sync_mode(GPU_SYNC_EAGER);
	gpu_set_fp8(0);
	lm_weights_ensure_host(r->m);
	lm_adam_free(&r->opt);
	lm_cache_free(&r->c);
	mat_free(&r->dh);
	lm_free(&r->m);
}

/* Forward + loss (+ backward) of one microbatch; returns mean CE. */
static double microbatch(rig *r, unsigned seed, int bwd)
{
	int ids[B * T];
	double ce = -1.0;

	fill_ids(ids, seed);
	assert(lm_forward_cache_packed(r->m, ids, B, T, r->c) == 0);
	assert(lm_loss_backward_packed(r->m, r->c, ids, B, T, r->dh, &ce) == 0);
	if (bwd)
		assert(lm_backward_packed(r->m, r->dh, r->c) == 0);
	gpu_device_sync();
	return ce;
}

static int collect(lm *m, mat **g, char names[][16])
{
	int n = 0;
	int i;

	strcpy(names[n], "dw_head");
	g[n++] = m->dw_head;
	strcpy(names[n], "dtable");
	g[n++] = m->embed->dtable;
	for (i = 0; i < m->tr->n_blocks; i++) {
		tblock *b = &m->tr->blocks[i];

		sprintf(names[n], "b%d.dwq", i);
		g[n++] = b->attn->dwq;
		sprintf(names[n], "b%d.dwk", i);
		g[n++] = b->attn->dwk;
		sprintf(names[n], "b%d.dwv", i);
		g[n++] = b->attn->dwv;
		sprintf(names[n], "b%d.dwo", i);
		g[n++] = b->attn->dwo;
		sprintf(names[n], "b%d.dw1", i);
		g[n++] = b->ffn->dw1;
		sprintf(names[n], "b%d.dw2", i);
		g[n++] = b->ffn->dw2;
	}
	return n;
}

/* STEPS Adam steps; snapshots[s][t] = host copy of grad tensor t before step s. */
static void train(int prec, float **snaps, double *loss, int *nt, int *sizes,
	char names[][16])
{
	rig r;
	mat *g[NT];
	int s;
	int a;
	int i;

	rig_open(&r, prec);
	*nt = collect(r.m, g, names);
	for (i = 0; i < *nt; i++)
		sizes[i] = g[i]->rows * g[i]->cols;
	for (s = 0; s < STEPS; s++) {
		double ce_sum = 0.0;

		lm_adam_zero_grad(r.m);
		vec_zero(r.m->db_head);
		for (a = 0; a < ACC; a++)
			ce_sum += microbatch(&r, 777u + 1000u * s + 10u * a, 1);
		loss[s] = ce_sum / ACC;
		for (i = 0; i < *nt; i++) {
			size_t nb = (size_t)sizes[i] * sizeof(float);

			snaps[s * NT + i] = malloc(nb);
			assert(snaps[s * NT + i]);
			/* dtable is device-authoritative too: pull every tensor. */
			assert(mat_pull_host(g[i]) == 0);
			memcpy(snaps[s * NT + i], g[i]->data, nb);
		}
		lm_adam_step_scaled(r.opt, 1e-3, 1.0 / (double)(B * (T - 1)));
	}
	rig_close(&r);
}

/* 1. Same model, same batch -> same loss; a new batch must not inherit the
** previous batch's targets; a fresh model sees the same first loss. */
static void test_forward_no_cross_batch_state(void)
{
	rig r;
	double a;
	double b;
	double c;
	double d;

	rig_open(&r, GPU_PREC_FP32);
	a = microbatch(&r, 777u, 0);
	b = microbatch(&r, 777u, 0);
	c = microbatch(&r, 999u, 0);
	rig_close(&r);
	rig_open(&r, GPU_PREC_FP32);
	d = microbatch(&r, 999u, 0);
	rig_close(&r);
	printf("  forward: %.6f %.6f | other batch %.6f fresh %.6f\n", a, b, c, d);
	assert(fabs(a - b) < 1e-5);
	assert(fabs(c - a) > 1e-3);		/* targets were stale if equal */
	assert(fabs(c - d) < 1e-5);
}

/* 2 + 3. After one Adam step: matrices moved by more than weight decay, FFN
** biases and LN vectors moved, and the next forward sees the new vectors. */
static void test_first_step_updates_everything(void)
{
	rig r;
	float *w0;
	double dw;
	double b1_0;
	double b1_1;
	double g0;
	double g1;
	double l_before;
	double l_after;
	double l_refresh;
	tblock *blk;

	rig_open(&r, GPU_PREC_FP32);
	blk = &r.m->tr->blocks[0];
	{
		size_t nb = (size_t)r.m->w_head->rows * r.m->w_head->cols
		    * sizeof(float);

		w0 = malloc(nb);
		assert(w0);
		mat_pull_host(r.m->w_head);
		memcpy(w0, r.m->w_head->data, nb);
	}
	b1_0 = abs_sum_vec(blk->ffn->b1);
	g0 = abs_sum_vec(blk->ln1->gamma);
	lm_adam_zero_grad(r.m);
	vec_zero(r.m->db_head);
	l_before = microbatch(&r, 777u, 1);
	lm_adam_step_scaled(r.opt, 1e-3, 1.0 / (double)(B * (T - 1)));
	{
		size_t i;
		size_t n = (size_t)r.m->w_head->rows * r.m->w_head->cols;

		mat_pull_host(r.m->w_head);
		dw = 0.0;
		for (i = 0; i < n; i++)
			dw += fabs((double)r.m->w_head->data[i] - (double)w0[i]);
		free(w0);
	}
	b1_1 = abs_sum_vec(blk->ffn->b1);
	g1 = abs_sum_vec(blk->ln1->gamma);
	printf("  step1: |dW_head|_1 %.4f, b1 %.4f -> %.4f, gamma %.4f -> %.4f\n",
	    dw, b1_0, b1_1, g0, g1);
	/*
	** Adam's first step moves every weight by about lr (sign-like), so
	** |dW|_1 is about lr * n = 16; weight decay alone gives 1e-4 * |W|_1,
	** under 0.05. A dropped first-step gradient fails this.
	*/
	assert(dw > 1.0);
	assert(b1_1 > 0.01);
	assert(fabs(g1 - g0) > 1e-3);
	/* The next forward must use the updated vectors: forcing a refresh of
	** every cached device vector must not change its loss. */
	lm_adam_zero_grad(r.m);
	l_after = microbatch(&r, 777u, 0);
	gpu_half_wt_invalidate(NULL);
	l_refresh = microbatch(&r, 777u, 0);
	printf("  loss before %.6f after %.6f refreshed %.6f\n", l_before, l_after,
	    l_refresh);
	assert(fabs(l_after - l_refresh) < 1e-5);
	assert(l_after < l_before);
	rig_close(&r);
}

/* 4. Train, free, and make sure no CUDA error is left pending. */
static void test_free_leaves_no_cuda_error(void)
{
	rig r;
	const char *err;

	(void)gpu_debug_last_error();
	rig_open(&r, GPU_PREC_FP32);
	lm_adam_zero_grad(r.m);
	microbatch(&r, 777u, 1);
	lm_adam_step_scaled(r.opt, 1e-3, 1.0 / (double)(B * (T - 1)));
	microbatch(&r, 787u, 1);
	rig_close(&r);
	err = gpu_debug_last_error();
	printf("  pending CUDA error after free: '%s'\n", err);
	assert(err[0] == '\0');
}

/*
** 5. Reduced-precision training tracks FP32: loss trajectory within
** loss_tol and every weight gradient within rel_tol relative error and
** ratio_tol in norm, from step 1 on (step 1 catches the GEMM-level bugs:
** timing loop, scale saturation; later steps the trajectory).
*/
static void check_tracks_fp32(const char *tag, int prec, double loss_tol,
	double rel_tol, double ratio_tol)
{
	static float *s32[STEPS * NT];
	static float *s16[STEPS * NT];
	double l32[STEPS];
	double l16[STEPS];
	int sizes[NT];
	char names[NT][16];
	int nt;
	int s;
	int i;
	int j;
	int bad = 0;

	train(GPU_PREC_FP32, s32, l32, &nt, sizes, names);
	train(prec, s16, l16, &nt, sizes, names);
	for (s = 0; s < STEPS; s++) {
		printf("  step %d loss fp32 %.5f %s %.5f\n", s + 1, l32[s], tag,
		    l16[s]);
		if (fabs(l32[s] - l16[s]) > loss_tol)
			bad = 1;
		for (i = 0; i < nt; i++) {
			double num = 0.0;
			double den = 0.0;
			double n16 = 0.0;
			double rel;
			double ratio;

			for (j = 0; j < sizes[i]; j++) {
				double d = s32[s * NT + i][j] - s16[s * NT + i][j];

				num += d * d;
				den += (double)s32[s * NT + i][j] * s32[s * NT + i][j];
				n16 += (double)s16[s * NT + i][j] * s16[s * NT + i][j];
			}
			rel = sqrt(num / (den + 1e-30));
			ratio = sqrt(n16 / (den + 1e-30));
			if (rel > rel_tol || ratio > 1.0 + ratio_tol
			    || ratio < 1.0 - ratio_tol) {
				printf("    %s: rel_err %.3f ratio %.3f\n", names[i], rel,
				    ratio);
				bad = 1;
			}
			free(s32[s * NT + i]);
			free(s16[s * NT + i]);
		}
	}
	assert(!bad);
}

int main(void)
{
	if (!gpu_available()) {
		printf("test_gpu: no GPU backend (%s), skipping\n",
		    gpu_backend_name());
		return 0;
	}
	setvbuf(stdout, NULL, _IONBF, 0);
	/* Load-independent cuBLASLt algo choice: BF16 rounding then repeats. */
	setenv("ANN_LT_HEURISTIC", "1", 1);
	printf("test_gpu on %s\n", gpu_backend_name());
	test_forward_no_cross_batch_state();
	test_first_step_updates_everything();
	test_free_leaves_no_cuda_error();
	printf("bf16 vs fp32\n");
	check_tracks_fp32("bf16", GPU_PREC_BF16, 2e-3, 0.2, 0.1);
	printf("fp8 (E4M3 fwd, E5M2 dgrad) vs fp32\n");
	check_tracks_fp32("fp8", PREC_FP8, 1e-2, 0.4, 0.2);
	printf("test_gpu: all passed\n");
	return 0;
}
