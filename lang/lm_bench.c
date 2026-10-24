/*
**	Wall-clock LM forward / train microbenchmarks and the lm bench harness
*/
/* popen / getpid for the GPU idle check and the git stamp. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include "ann_posix.h"
#include <process.h>
#define popen	_popen
#define pclose	_pclose
#define getpid	_getpid
#else
#include <sys/time.h>
#include <unistd.h>
#endif
#include "attn.h"
#include "generate.h"
#include "gpu.h"
#include "gradutil.h"
#include "lm_adam.h"
#include "lm_bench.h"
#include "lm_serialize.h"
#include "lm_train.h"
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

/* ---- lm bench harness --------------------------------------------------- */

static const char *const g_prec_name[] = { "fp32", "bf16", "fp8" };

/* Below this many timed steps the loss check only requires finite CE. */
#define LM_BENCH_LOSS_MIN_STEPS	16

void lm_bench_cfg_defaults(lm_bench_cfg *cfg)
{
	memset(cfg, 0, sizeof(*cfg));
	cfg->B = 32;
	cfg->T = LM_MAX_CTX < 256 ? LM_MAX_CTX : 256;
	cfg->vocab = 16384;
	cfg->prec = LM_BENCH_BF16;
	cfg->graphs = 1;
	cfg->warmup = 5;
	cfg->steps = 30;
	cfg->ceiling = 1;
	cfg->seed = 1;
	cfg->lr = 3e-4;
	cfg->json = "bench/results.jsonl";
	cfg->prompt = "The ";
	cfg->tokens = 128;
	cfg->runs = 3;
}

static double block_matmul_params(void)
{
	return 4.0 * LM_D_MODEL * LM_D_MODEL + 2.0 * LM_D_MODEL * LM_D_FF;
}

double lm_bench_flops_per_token(int T, int vocab)
{
	double p;

	/* Embedding lookup is a gather (no FLOPs); the head is a D x V GEMM. */
	p = (double)LM_N_BLOCKS * block_matmul_params()
	    + (double)LM_D_MODEL * (double)vocab;
	/* Causal QK^T + AV: 2*T*D fwd per layer per token on average, x3. */
	return 6.0 * p + 6.0 * (double)LM_N_BLOCKS * (double)T * LM_D_MODEL;
}

double lm_bench_recompute_flops_per_token(int T)
{
	/* One extra block forward per token. */
	return 2.0 * (double)LM_N_BLOCKS * block_matmul_params()
	    + 2.0 * (double)LM_N_BLOCKS * (double)T * LM_D_MODEL;
}

static int cmp_double(const void *a, const void *b)
{
	double x = *(const double *)a;
	double y = *(const double *)b;

	return (x > y) - (x < y);
}

typedef struct	bench_stats {
	double	p10;
	double	p50;
	double	p90;
	double	mean;
	double	cv;
} bench_stats;

/* Nearest-rank percentiles; sorts v in place. */
static void stats_of(double *v, int n, bench_stats *st)
{
	double sum;
	double ss;
	int i;

	memset(st, 0, sizeof(*st));
	if (n <= 0)
		return;
	qsort(v, (size_t)n, sizeof(double), cmp_double);
	st->p10 = v[(n - 1) / 10];
	st->p50 = v[(n - 1) / 2];
	st->p90 = v[(9 * (n - 1) + 9) / 10];
	sum = 0.0;
	for (i = 0; i < n; i++)
		sum += v[i];
	st->mean = sum / (double)n;
	ss = 0.0;
	for (i = 0; i < n; i++)
		ss += (v[i] - st->mean) * (v[i] - st->mean);
	st->cv = (n > 1 && st->mean > 0.0)
	    ? sqrt(ss / (double)(n - 1)) / st->mean : 0.0;
}

/* First line of a shell command's stdout into buf (trimmed); 0 on success. */
static int cmd_line(const char *cmd, char *buf, size_t cap)
{
	FILE *fp;
	size_t n;

	buf[0] = '\0';
	fp = popen(cmd, "r");
	if (!fp)
		return -1;
	if (!fgets(buf, (int)cap, fp))
		buf[0] = '\0';
	pclose(fp);
	n = strlen(buf);
	while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'
	    || buf[n - 1] == ' '))
		buf[--n] = '\0';
	return buf[0] ? 0 : -1;
}

/* Strip characters that would break a JSON string. */
static void json_safe(char *s)
{
	for (; *s; s++)
		if (*s == '"' || *s == '\\' || (unsigned char)*s < 32)
			*s = '_';
}

typedef struct	bench_env {
	char	git[48];
	int	dirty;
	char	gpu[96];
	char	driver[32];
	double	vram_total_mb;
	double	vram_used_mb;	/* by everything else, before we allocate */
	int	busy;
} bench_env;

/*
** GPU name/driver via nvidia-smi; "busy" when another process holds more
** than 512 MB. WSL does not list compute apps, so used VRAM before any
** allocation of ours is the fallback signal (llama-server shows up there).
*/
static void bench_env_probe(bench_env *e)
{
	char line[256];
	size_t fr;
	size_t tot;
	FILE *fp;
	long pid;
	double mb;
	double other;

	memset(e, 0, sizeof(*e));
	if (cmd_line("git rev-parse --short HEAD 2>/dev/null", e->git,
	    sizeof(e->git)) != 0)
		snprintf(e->git, sizeof(e->git), "unknown");
	e->dirty = cmd_line("git status --porcelain --untracked-files=no "
	    "2>/dev/null", line, sizeof(line)) == 0;
	json_safe(e->git);
	snprintf(e->gpu, sizeof(e->gpu), "%s", gpu_available() ? "cuda" : "cpu");
	if (!gpu_available())
		return;
	if (cmd_line("nvidia-smi --query-gpu=name --format=csv,noheader "
	    "2>/dev/null", line, sizeof(line)) == 0)
		snprintf(e->gpu, sizeof(e->gpu), "%.95s", line);
	if (cmd_line("nvidia-smi --query-gpu=driver_version "
	    "--format=csv,noheader 2>/dev/null", line, sizeof(line)) == 0)
		snprintf(e->driver, sizeof(e->driver), "%.31s", line);
	json_safe(e->gpu);
	json_safe(e->driver);
	fr = 0;
	tot = 0;
	if (gpu_mem_info(&fr, &tot) == 0 && tot > 0) {
		e->vram_total_mb = (double)tot / 1048576.0;
		e->vram_used_mb = (double)(tot - fr) / 1048576.0;
	}
	other = 0.0;
	fp = popen("nvidia-smi --query-compute-apps=pid,used_memory "
	    "--format=csv,noheader,nounits 2>/dev/null", "r");
	if (fp) {
		while (fgets(line, (int)sizeof(line), fp)) {
			if (sscanf(line, "%ld, %lf", &pid, &mb) == 2
			    && pid != (long)getpid())
				other += mb;
		}
		pclose(fp);
	}
	/* Our own CUDA context is ~300-500 MB at this point. */
	e->busy = other > 512.0 || e->vram_used_mb > 1024.0;
}

static int bench_env_ok(const lm_bench_cfg *cfg, const bench_env *e)
{
	if (!e->busy)
		return 1;
	fprintf(stderr, "lm bench: GPU busy (%.0f MB of %.0f MB already used "
	    "by other processes)\n", e->vram_used_mb, e->vram_total_mb);
	if (cfg->allow_busy) {
		fprintf(stderr, "lm bench: --allow-busy: numbers are recorded "
		    "with busy=true\n");
		return 1;
	}
	fprintf(stderr, "lm bench: refusing; free the GPU or pass "
	    "--allow-busy\n");
	return 0;
}

static FILE *json_open(const lm_bench_cfg *cfg)
{
	FILE *fp;

	if (!cfg->json || !cfg->json[0])
		return NULL;
	fp = fopen(cfg->json, "a");
	if (!fp && strncmp(cfg->json, "bench/", 6) == 0) {
		if (system("mkdir -p bench") == 0)
			fp = fopen(cfg->json, "a");
	}
	if (!fp)
		fprintf(stderr, "lm bench: cannot append %s\n", cfg->json);
	return fp;
}

static void json_head(FILE *fp, const char *mode, const bench_env *e)
{
	fprintf(fp, "{\"mode\":\"%s\",\"git\":\"%s\",\"dirty\":%s,"
	    "\"backend\":\"%s\",\"gpu\":\"%s\",\"driver\":\"%s\","
	    "\"vram_mb\":%.0f,\"busy\":%s,\"time\":%ld,"
	    "\"model\":{\"D\":%d,\"L\":%d,\"H\":%d,\"FF\":%d}",
	    mode, e->git, e->dirty ? "true" : "false", gpu_backend_name(),
	    e->gpu, e->driver, e->vram_total_mb, e->busy ? "true" : "false",
	    (long)time(NULL), LM_D_MODEL, LM_N_BLOCKS, LM_N_HEADS, LM_D_FF);
}

static void bench_apply(const lm_bench_cfg *cfg)
{
	lm_train_set_compute_prec(cfg->prec == LM_BENCH_FP32
	    ? GPU_PREC_FP32 : GPU_PREC_BF16);
	lm_train_set_fp8(cfg->prec == LM_BENCH_FP8);
	if (cfg->prec != LM_BENCH_FP8)
		lm_train_set_fp8_dgrad(0);
	/* Auto attention: TC scores when they fit, else tiled flash. */
	lm_train_set_flash_attn(1);
	lm_train_set_act_checkpoint(cfg->ckpt);
	lm_train_set_adam_offload(cfg->offload);
	lm_train_set_cuda_graphs(cfg->graphs);
	lm_train_set_varlen(0);
	lm_train_set_grad_clip(0.0);
	lm_train_set_weight_decay(0.0);
}

/* Best TF/s of an (M x K) * (K x N) GEMM at the current compute precision. */
static double gemm_tf(int M, int K, int N, int warmup, int iters)
{
	mat *a;
	mat *b;
	mat *c;
	double *ms;
	double t0;
	double best;
	int i;
	int n;

	a = mat_create(M, K);
	b = mat_create(K, N);
	c = mat_create(M, N);
	ms = calloc((size_t)iters, sizeof(double));
	best = -1.0;
	if (!a || !b || !c || !ms)
		goto out;
	n = M * K;
	for (i = 0; i < n; i++)
		a->data[i] = (ann_real)(((i * 7 + 3) % 17) - 8) * (ann_real)0.01;
	n = K * N;
	for (i = 0; i < n; i++)
		b->data[i] = (ann_real)(((i * 5 + 1) % 13) - 6) * (ann_real)0.01;
	mat_try_pin(a);
	mat_try_pin(b);
	mat_try_pin(c);
	if (gpu_available()) {
		mat_to_device(a);
		mat_to_device(b);
		gpu_set_sync_mode(GPU_SYNC_BATCH);
	}
	for (i = 0; i < warmup + iters; i++) {
		gpu_device_sync();
		t0 = wall_sec();
		if (mat_mul(c, a, b) != 0)
			goto out;
		gpu_device_sync();
		if (i >= warmup)
			ms[i - warmup] = 1000.0 * (wall_sec() - t0);
	}
	qsort(ms, (size_t)iters, sizeof(double), cmp_double);
	if (ms[0] > 0.0)
		best = 2.0 * (double)M * (double)K * (double)N
		    / (ms[0] * 1e-3) / 1e12;
out:
	gpu_set_sync_mode(GPU_SYNC_EAGER);
	free(ms);
	mat_free(&a);
	mat_free(&b);
	mat_free(&c);
	return best;
}

/*
** Ceiling for MFU: the model's FFN-up GEMM with enough rows to saturate the
** device (independent of the step's B*T, so MFU < 100% means headroom).
*/
static double step_ceiling(void)
{
	return gemm_tf(gpu_available() ? 16384 : 2048, LM_D_MODEL, LM_D_FF,
	    2, 5);
}

/* One step point; prints a row and appends JSON. Returns 0, 1 = loss check failed, -1 = error. */
static int bench_step_point(const lm_bench_cfg *cfg, const bench_env *env,
	FILE *js, double ceiling)
{
	lm *model;
	lm_adam *opt;
	lm_step_time *st;
	double *v;
	int *ids;
	int B;
	int i;
	int n;
	int rc;
	double ce0;
	double ce1;
	double tok_s;
	double tf_model;
	double tf_hw;
	double mfu;
	double vram_peak;
	size_t fr;
	size_t tot;
	bench_stats ms;
	bench_stats ph[4];
	int loss_ok;

	srand((unsigned int)cfg->seed);
	bench_apply(cfg);
	model = lm_create_vocab(cfg->vocab);
	opt = model ? lm_adam_create(model) : NULL;
	B = cfg->B;
	n = B * cfg->T;
	ids = malloc((size_t)n * sizeof(int));
	st = calloc((size_t)cfg->steps, sizeof(lm_step_time));
	v = calloc((size_t)cfg->steps, sizeof(double));
	rc = -1;
	if (!model || !opt || !ids || !st || !v)
		goto out;
	for (i = 0; i < n; i++)
		ids[i] = VOCAB_SPECIALS + rand() % (model->vocab_size
		    - VOCAB_SPECIALS);
	if (lm_train_bench_packed(model, opt, ids, &B, cfg->T, cfg->lr,
	    cfg->warmup, cfg->steps, cfg->profile, st) != 0) {
		fprintf(stderr, "lm bench step: B=%d T=%d %s failed\n",
		    cfg->B, cfg->T, g_prec_name[cfg->prec]);
		goto out;
	}
	vram_peak = -1.0;
	fr = 0;
	tot = 0;
	if (gpu_available() && gpu_mem_info(&fr, &tot) == 0 && tot > 0)
		vram_peak = (double)(tot - fr) / 1048576.0;
	for (i = 0; i < cfg->steps; i++)
		v[i] = st[i].ms;
	stats_of(v, cfg->steps, &ms);
	memset(ph, 0, sizeof(ph));
	if (cfg->profile) {
		int k;

		for (k = 0; k < 4; k++) {
			int m = 0;

			for (i = 0; i < cfg->steps; i++) {
				double x = k == 0 ? st[i].fwd_ms
				    : k == 1 ? st[i].loss_ms
				    : k == 2 ? st[i].bwd_ms : st[i].adam_ms;

				if (x >= 0.0)
					v[m++] = x;
			}
			if (m > 0)
				stats_of(v, m, &ph[k]);
			else
				ph[k].p50 = -1.0;
		}
	}
	ce0 = -1.0;
	ce1 = -1.0;
	for (i = 0; i < cfg->steps; i++) {
		if (st[i].ce >= 0.0 && isfinite(st[i].ce)) {
			if (ce0 < 0.0)
				ce0 = st[i].ce;
			ce1 = st[i].ce;
		}
	}
	/*
	** Same batch every step: CE must be finite and fall. Adam can lift it
	** for the first few steps at D=1024, so short runs only check finite.
	*/
	loss_ok = ce0 >= 0.0 && ce1 >= 0.0
	    && (cfg->steps < LM_BENCH_LOSS_MIN_STEPS || ce1 < ce0);
	tok_s = ms.p50 > 0.0 ? (double)B * cfg->T / (ms.p50 * 1e-3) : 0.0;
	tf_model = tok_s * lm_bench_flops_per_token(cfg->T, model->vocab_size)
	    / 1e12;
	tf_hw = tf_model + (cfg->ckpt ? tok_s
	    * lm_bench_recompute_flops_per_token(cfg->T) / 1e12 : 0.0);
	mfu = ceiling > 0.0 ? tf_model / ceiling : -1.0;
	printf("step %-4s B=%-4d T=%-3d graphs=%d ckpt=%d off=%d  "
	    "p50 %8.2f ms (p10 %.2f p90 %.2f cv %.1f%%)  %9.0f tok/s  "
	    "%7.3f TF", g_prec_name[cfg->prec], B, cfg->T, cfg->graphs,
	    cfg->ckpt, cfg->offload, ms.p50, ms.p10, ms.p90, 100.0 * ms.cv,
	    tok_s, tf_model);
	if (mfu >= 0.0)
		printf("  MFU %4.1f%%", 100.0 * mfu);
	if (vram_peak >= 0.0)
		printf("  VRAM %.0f MB", vram_peak);
	printf("  loss %.3f->%.3f %s\n", ce0, ce1, loss_ok ? "ok" : "FAIL");
	if (cfg->profile)
		printf("     phases p50: fwd %.2f  loss %.2f  bwd %.2f  "
		    "adam %.2f ms%s\n", ph[0].p50, ph[1].p50, ph[2].p50,
		    ph[3].p50, ph[0].p50 < 0.0 ? " (graph replay: no phases)"
		    : "");
	if (ms.cv > 0.05)
		fprintf(stderr, "lm bench: warning: cv %.1f%% > 5%% (noisy run)\n",
		    100.0 * ms.cv);
	if (js) {
		json_head(js, "step", env);
		fprintf(js, ",\"cfg\":{\"V\":%d,\"B\":%d,\"B_req\":%d,\"T\":%d,"
		    "\"prec\":\"%s\",\"attn\":\"auto\",\"graphs\":%d,"
		    "\"ckpt\":%d,\"offload\":%d,\"warmup\":%d,\"steps\":%d,"
		    "\"lr\":%g},", model->vocab_size, B, cfg->B, cfg->T,
		    g_prec_name[cfg->prec], cfg->graphs, cfg->ckpt,
		    cfg->offload, cfg->warmup, cfg->steps, cfg->lr);
		fprintf(js, "\"ms\":{\"p10\":%.4f,\"p50\":%.4f,\"p90\":%.4f,"
		    "\"mean\":%.4f,\"cv\":%.4f},\"tok_s\":%.1f,"
		    "\"tf_model\":%.4f,\"tf_hw\":%.4f,\"ceiling_tf\":%.4f,"
		    "\"mfu\":%.4f,\"vram_peak_mb\":%.0f,"
		    "\"loss\":[%.5f,%.5f],\"loss_ok\":%s",
		    ms.p10, ms.p50, ms.p90, ms.mean, ms.cv, tok_s, tf_model,
		    tf_hw, ceiling, mfu, vram_peak, ce0, ce1,
		    loss_ok ? "true" : "false");
		if (cfg->profile)
			fprintf(js, ",\"phase_ms\":{\"fwd\":%.4f,\"loss\":%.4f,"
			    "\"bwd\":%.4f,\"adam\":%.4f}", ph[0].p50,
			    ph[1].p50, ph[2].p50, ph[3].p50);
		fprintf(js, "}\n");
		fflush(js);
	}
	rc = loss_ok ? 0 : 1;
out:
	free(v);
	free(st);
	free(ids);
	lm_adam_free(&opt);
	lm_free(&model);
	return rc;
}

int lm_bench_step(const lm_bench_cfg *cfg)
{
	bench_env env;
	FILE *js;
	double ceiling;
	int rc;

	if (!cfg || cfg->B < 1 || cfg->T < 2 || cfg->T > LM_MAX_CTX
	    || cfg->steps < 1 || cfg->vocab < LM_BYTE_VOCAB + VOCAB_SPECIALS) {
		fprintf(stderr, "lm bench step: need B>=1, 2<=T<=%d, steps>=1, "
		    "vocab>=%d\n", LM_MAX_CTX, LM_BYTE_VOCAB + VOCAB_SPECIALS);
		return 1;
	}
	bench_env_probe(&env);
	if (!bench_env_ok(cfg, &env))
		return 2;
	bench_apply(cfg);
	ceiling = cfg->ceiling ? step_ceiling() : -1.0;
	js = json_open(cfg);
	rc = bench_step_point(cfg, &env, js, ceiling);
	if (js)
		fclose(js);
	return rc == 0 ? 0 : 1;
}

int lm_bench_gemm(const lm_bench_cfg *cfg)
{
	static const char *const names[] = { "attn_proj", "ffn_up",
	    "ffn_down", "head" };
	bench_env env;
	FILE *js;
	int K[4];
	int N[4];
	int M;
	int i;
	double tf;

	bench_env_probe(&env);
	if (!bench_env_ok(cfg, &env))
		return 2;
	bench_apply(cfg);
	M = cfg->B * cfg->T;
	K[0] = LM_D_MODEL;
	N[0] = LM_D_MODEL;
	K[1] = LM_D_MODEL;
	N[1] = LM_D_FF;
	K[2] = LM_D_FF;
	N[2] = LM_D_MODEL;
	K[3] = LM_D_MODEL;
	N[3] = cfg->vocab;
	js = json_open(cfg);
	for (i = 0; i < 4; i++) {
		tf = gemm_tf(M, K[i], N[i], 3, cfg->steps < 5 ? 5 : cfg->steps);
		printf("gemm %-4s %-9s M=%-6d K=%-5d N=%-6d  best %7.2f TF\n",
		    g_prec_name[cfg->prec], names[i], M, K[i], N[i], tf);
		if (js) {
			json_head(js, "gemm", &env);
			fprintf(js, ",\"cfg\":{\"prec\":\"%s\",\"shape\":\"%s\","
			    "\"M\":%d,\"K\":%d,\"N\":%d},\"tf_best\":%.4f}\n",
			    g_prec_name[cfg->prec], names[i], M, K[i], N[i], tf);
		}
	}
	if (js)
		fclose(js);
	return 0;
}

/* FNV-1a, to compare greedy outputs across commits without storing text. */
static unsigned long text_hash(const char *s)
{
	unsigned long h = 2166136261UL;

	for (; *s; s++) {
		h ^= (unsigned char)*s;
		h = (h * 16777619UL) & 0xffffffffUL;
	}
	return h;
}

int lm_bench_decode(const lm_bench_cfg *cfg)
{
	bench_env env;
	FILE *js;
	lm *model;
	char *out;
	double *t1;
	double *tn;
	double t0;
	bench_stats s1;
	bench_stats sn;
	unsigned long hash;
	int ids[LM_MAX_CTX];
	int n_prompt;
	int nnew;
	int cap;
	int r;
	int rc;

	if (!cfg || !cfg->model_path || cfg->runs < 1) {
		fprintf(stderr, "lm bench decode: need --model PATH\n");
		return 1;
	}
	bench_env_probe(&env);
	if (!bench_env_ok(cfg, &env))
		return 2;
	model = lm_load_any(cfg->model_path);
	if (!model) {
		fprintf(stderr, "lm bench decode: cannot load %s\n",
		    cfg->model_path);
		return 1;
	}
	n_prompt = lm_encode(model, cfg->prompt, ids, LM_MAX_CTX);
	nnew = cfg->tokens;
	if (n_prompt < 1 || n_prompt + 2 > LM_MAX_CTX) {
		fprintf(stderr, "lm bench decode: prompt must be 1..%d tokens\n",
		    LM_MAX_CTX - 2);
		lm_free(&model);
		return 1;
	}
	if (nnew < 2)
		nnew = 2;
	if (n_prompt + nnew > LM_MAX_CTX)
		nnew = LM_MAX_CTX - n_prompt;
	cap = 16 * (LM_MAX_CTX + 8) + (int)strlen(cfg->prompt);
	out = malloc((size_t)cap);
	t1 = calloc((size_t)cfg->runs, sizeof(double));
	tn = calloc((size_t)cfg->runs, sizeof(double));
	rc = 1;
	hash = 0;
	if (!out || !t1 || !tn)
		goto done;
	/* Warm: device weights, KV buffers, decode graph. */
	if (lm_generate(model, cfg->prompt, out, cap, 2, 0.0, 0, 1.0) < 0)
		goto done;
	for (r = 0; r < cfg->runs; r++) {
		t0 = wall_sec();
		if (lm_generate(model, cfg->prompt, out, cap, 1, 0.0, 0, 1.0)
		    < 0)
			goto done;
		t1[r] = 1000.0 * (wall_sec() - t0);
		t0 = wall_sec();
		if (lm_generate(model, cfg->prompt, out, cap, nnew, 0.0, 0,
		    1.0) < 0)
			goto done;
		tn[r] = 1000.0 * (wall_sec() - t0);
		if (r == 0)
			hash = text_hash(out);
		else if (text_hash(out) != hash) {
			fprintf(stderr, "lm bench decode: greedy output differs "
			    "between runs\n");
			goto done;
		}
	}
	stats_of(t1, cfg->runs, &s1);
	stats_of(tn, cfg->runs, &sn);
	{
		double steady = sn.p50 > s1.p50
		    ? (double)(nnew - 1) / ((sn.p50 - s1.p50) * 1e-3) : -1.0;

		printf("decode prompt=%d new=%d  first-token %.2f ms  "
		    "total %.2f ms  steady %.1f tok/s  hash %08lx\n", n_prompt,
		    nnew, s1.p50, sn.p50, steady, hash);
		js = json_open(cfg);
		if (js) {
			json_head(js, "decode", &env);
			fprintf(js, ",\"cfg\":{\"prompt_tokens\":%d,\"new\":%d,"
			    "\"runs\":%d,\"V\":%d},\"ttft_ms\":%.4f,"
			    "\"total_ms\":%.4f,\"steady_tok_s\":%.2f,"
			    "\"hash\":\"%08lx\"}\n", n_prompt, nnew, cfg->runs,
			    model->vocab_size, s1.p50, sn.p50, steady, hash);
			fclose(js);
		}
	}
	rc = 0;
done:
	free(out);
	free(t1);
	free(tn);
	lm_free(&model);
	return rc;
}

/* Parse a comma list of ints into v (max cap); returns count. */
static int parse_ints(const char *s, int *v, int cap)
{
	int n = 0;
	char *end;
	long x;

	while (s && *s && n < cap) {
		x = strtol(s, &end, 10);
		if (end == s)
			break;
		v[n++] = (int)x;
		s = (*end == ',') ? end + 1 : end;
	}
	return n;
}

static int parse_precs(const char *s, int *v, int cap)
{
	int n = 0;
	int i;
	size_t len;

	while (s && *s && n < cap) {
		len = strcspn(s, ",");
		for (i = 0; i < 3; i++) {
			if (strlen(g_prec_name[i]) == len
			    && strncmp(s, g_prec_name[i], len) == 0) {
				v[n++] = i;
				break;
			}
		}
		if (i == 3)
			return -1;
		s += len;
		if (*s == ',')
			s++;
	}
	return n;
}

int lm_bench_sweep(const lm_bench_cfg *cfg, const char *windows,
	const char *batches, const char *precs, const char *graphs)
{
	lm_bench_cfg c;
	bench_env env;
	FILE *js;
	int Ts[8];
	int Bs[16];
	int Ps[3];
	int Gs[2];
	int nT;
	int nB;
	int nP;
	int nG;
	int a;
	int b;
	int p;
	int g;
	int bad;
	int r;
	double ceil_tf[3];

	nT = parse_ints(windows, Ts, 8);
	nB = parse_ints(batches, Bs, 16);
	nP = parse_precs(precs, Ps, 3);
	nG = parse_ints(graphs, Gs, 2);
	if (nT < 1 || nB < 1 || nP < 1 || nG < 1) {
		fprintf(stderr, "lm bench sweep: bad --windows/--batches/"
		    "--precs/--graphs list\n");
		return 1;
	}
	bench_env_probe(&env);
	if (!bench_env_ok(cfg, &env))
		return 2;
	for (p = 0; p < 3; p++)
		ceil_tf[p] = -1.0;
	js = json_open(cfg);
	bad = 0;
	for (p = 0; p < nP; p++) {
		c = *cfg;
		c.prec = Ps[p];
		if (c.ceiling) {
			bench_apply(&c);
			ceil_tf[Ps[p]] = step_ceiling();
		}
		for (a = 0; a < nT; a++)
			for (b = 0; b < nB; b++)
				for (g = 0; g < nG; g++) {
					c.T = Ts[a];
					c.B = Bs[b];
					c.graphs = Gs[g];
					if (c.T < 2 || c.T > LM_MAX_CTX
					    || c.B < 1)
						continue;
					r = bench_step_point(&c, &env, js,
					    ceil_tf[c.prec]);
					if (r != 0)
						bad++;
				}
	}
	if (js)
		fclose(js);
	if (bad)
		fprintf(stderr, "lm bench sweep: %d point(s) failed or failed "
		    "the loss check\n", bad);
	return bad ? 1 : 0;
}
