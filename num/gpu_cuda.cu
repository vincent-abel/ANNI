/*
**	Optional CUDA backend (build with: make cuda)
*/
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#if CUDART_VERSION >= 12000
#include <cuda_fp8.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include "ann_posix.h"
#else
#include <execinfo.h>
#include <pthread.h>
#endif
#ifdef ANN_CUBLAS
#include <cublas_v2.h>
#include <cublasLt.h>
#endif
extern "C" {
#include "gpu.h"
#include "mat.h"
}

static int g_sync_mode;
static void devvec_forget(const void *host);
static void qkv_bf16_forget(const void *p);
static int twin_emit_on(void);
static int g_compute_prec;
static int g_fp8_fwd;
static int g_fp8_dgrad;
static cudaStream_t g_copy_stream;
/*
** Compute stream: every kernel, cuBLAS call and async copy of the training
** path goes here. A blocking stream (cudaStreamCreate) orders itself with the
** legacy default stream exactly as stream 0 did, so the synchronous
** cudaMemcpy calls keep their meaning; unlike stream 0 it can be captured
** into a CUDA graph (cudaStreamBeginCapture refuses the legacy stream).
*/
static cudaStream_t g_st;

/*
** CUDA-graph bookkeeping for every device allocation in this file.
** A captured graph holds raw device addresses: any cudaFree may hand one of
** them to a new owner, so each free bumps g_dev_free_gen and a graph older
** than the current generation is never replayed. cudaMalloc / cudaFree are
** illegal inside a capture (they invalidate it); they are refused there and
** the capture is marked bad, so a path that was not warmed up before the
** capture costs a re-capture instead of a broken graph. A path that takes
** another branch when an allocation fails is harmless for the same reason:
** the bad capture is thrown away, never replayed.
*/
static unsigned long g_dev_free_gen;
static int g_capture_bad;
static int g_graph_mode;	/* gpu_graph_set_mode: lm_graph_step is in use */

static int capturing_now(void)
{
	cudaStreamCaptureStatus cs;

	if (!g_st || cudaStreamIsCapturing(g_st, &cs) != cudaSuccess)
		return 0;
	/* Invalidated still counts: the host must keep off the device. */
	return cs != cudaStreamCaptureStatusNone;
}

/* ANN_GRAPH_DEBUG=1: say why a capture was thrown away. */
static int graph_debug_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_GRAPH_DEBUG");

		on = (e && e[0] == '1') ? 1 : 0;
	}
	return on;
}

static void capture_refuse(const char *what, int line)
{
	if (graph_debug_on() && !g_capture_bad) {
		fprintf(stderr, "[graph] capture refused: %s at gpu_cuda.cu:%d\n",
		    what, line);
#ifndef _WIN32
		{
			void *bt[24];

			backtrace_symbols_fd(bt, backtrace(bt, 24), 2);
		}
#endif
	}
	g_capture_bad = 1;
}

static cudaError_t ann_dev_malloc(void **p, size_t n, int line)
{
	if (capturing_now()) {
		capture_refuse("cudaMalloc", line);
		*p = NULL;
		return cudaErrorStreamCaptureUnsupported;
	}
	return cudaMalloc(p, n);
}

static cudaError_t ann_dev_free(void *p, int line)
{
	if (!p)
		return cudaSuccess;
	if (capturing_now()) {
		capture_refuse("cudaFree", line);
		return cudaErrorStreamCaptureUnsupported;
	}
	g_dev_free_gen++;
	return cudaFree(p);
}
#define cudaMalloc(p, n)	ann_dev_malloc((void **)(p), (n), __LINE__)
#define cudaFree(p)		ann_dev_free((void *)(p), __LINE__)
static cudaGraph_t g_graph;
static cudaGraphExec_t g_graph_exec;
static int g_graph_ok;
static unsigned long g_graph_gen;	/* g_dev_free_gen when captured */

extern "C" void gpu_set_compute_prec(int prec)
{
	if (prec != GPU_PREC_BF16)
		prec = GPU_PREC_FP32;
	g_compute_prec = prec;
}

extern "C" void gpu_set_fp8(int on)
{
	/* On implies dgrad; off switches both (an FP8 dgrad left on after an
	** FP8 run silently quantized the gradients of the next BF16 run). */
	g_fp8_fwd = on ? 1 : 0;
	g_fp8_dgrad = on ? 1 : 0;
}

extern "C" int gpu_fp8(void)
{
	return g_fp8_fwd;
}

extern "C" void gpu_set_fp8_dgrad(int on)
{
	g_fp8_dgrad = on ? 1 : 0;
}

extern "C" void gpu_set_sync_mode(int mode)
{
	g_sync_mode = mode;
}

extern "C" int gpu_sync_mode(void)
{
	return g_sync_mode;
}

#ifdef ANN_CUBLAS
static cublasHandle_t g_cublas;
static cublasLtHandle_t g_cublaslt;
static int g_cublas_ok;
static void *g_lt_workspace;
static size_t g_lt_workspace_bytes;
static pthread_mutex_t g_gemm_mu = PTHREAD_MUTEX_INITIALIZER;
#define LT_WS_BYTES	(128ull * 1024ull * 1024ull)
#define LT_ALGO_CACHE	64
typedef struct {
	int		M, N, K;
	int		opa, opb;
	int		bf16;
	int		fp8;	/* separate from BF16 — never reuse BF16 algos */
	int		epi;	/* cublasLtEpilogue_t; 0 = default */
	int		obf;	/* C / D in BF16 instead of FP32 */
	int		lda, ldb, ldc;
	int		valid;
	int		handles;
	cublasLtMatmulAlgo_t algo;
	cublasLtMatmulDesc_t desc;
	cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
	const float	*bias;
	size_t		ws;
} lt_algo_entry;
static lt_algo_entry g_lt_algo[LT_ALGO_CACHE];
/* Device bias scratch for Lt epilogues (host vec biases are not aliased). */
static float *g_lt_bias_dev;
static size_t g_lt_bias_cap;
static void *g_lt_bias16;	/* BF16 copy for BF16-output epilogues */
static size_t g_lt_bias16_cap;
/* Parallel QKV: three handles/streams so Wq/Wk/Wv GEMMs overlap. */
static cudaStream_t g_qkv_stream[3];
static cublasHandle_t g_qkv_blas[3];
static cudaEvent_t g_qkv_ready;
static int g_qkv_par_ok;

static int cublas_ensure(void)
{
	if (g_cublas_ok)
		return 0;
	if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS)
		return -1;
	if (cublasLtCreate(&g_cublaslt) != CUBLAS_STATUS_SUCCESS) {
		cublasDestroy(g_cublas);
		return -1;
	}
	/* R10: TF32 tensor cores on Ampere+ when the cuBLAS header knows it. */
#ifdef CUBLAS_TF32_TENSOR_OP_MATH
	(void)cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
#elif defined(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
	(void)cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
#endif
	(void)gpu_available();	/* creates g_st */
	if (cublasSetStream(g_cublas, g_st) != CUBLAS_STATUS_SUCCESS) {
		cublasLtDestroy(g_cublaslt);
		cublasDestroy(g_cublas);
		return -1;
	}
	if (cudaMalloc(&g_lt_workspace, LT_WS_BYTES) == cudaSuccess)
		g_lt_workspace_bytes = LT_WS_BYTES;
	else {
		g_lt_workspace = NULL;
		g_lt_workspace_bytes = 0;
	}
	g_cublas_ok = 1;
	return 0;
}

static int qkv_par_ensure(void)
{
	int i;

	if (g_qkv_par_ok)
		return 0;
	if (cublas_ensure() != 0)
		return -1;
	if (cudaEventCreateWithFlags(&g_qkv_ready, cudaEventDisableTiming)
	    != cudaSuccess)
		return -1;
	for (i = 0; i < 3; i++) {
		if (cudaStreamCreateWithFlags(&g_qkv_stream[i],
		    cudaStreamNonBlocking) != cudaSuccess)
			return -1;
		if (cublasCreate(&g_qkv_blas[i]) != CUBLAS_STATUS_SUCCESS)
			return -1;
		if (cublasSetStream(g_qkv_blas[i], g_qkv_stream[i])
		    != CUBLAS_STATUS_SUCCESS)
			return -1;
#ifdef CUBLAS_TF32_TENSOR_OP_MATH
		(void)cublasSetMathMode(g_qkv_blas[i],
		    CUBLAS_TF32_TENSOR_OP_MATH);
#endif
	}
	g_qkv_par_ok = 1;
	return 0;
}

static int cublas_gemm_ex_bf16_h(cublasHandle_t h, int N, int M, int K,
	const float *alpha, const void *B, int ldb, const void *A, int lda,
	const float *beta, float *C, int ldc)
{
	return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B,
	    CUDA_R_16BF, ldb, A, CUDA_R_16BF, lda, beta, C, CUDA_R_32F, ldc,
	    CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
	    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

/*
** ANN_LT_SPLIT1=0: time only the heuristic's own candidates. By default
** each split-K candidate is also timed with split-K 1 (B6): the W2 forward
** at 4096 x 1024 x 4096 got split-K 2, whose reduction + bias + residual
** runs as a separate cuBLASLt pass (66 us per layer), and the heuristic
** list held no unsplit version of the same tile to compare against.
** ANN_LT_LOG=1 prints the shapes where such a variant won.
*/
static int lt_split1_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_LT_SPLIT1");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

static int lt_heuristic_only(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_LT_HEURISTIC");

		on = (e && e[0] == '1') ? 1 : 0;
	}
	return on;
}

/*
** cuBLASLt BF16 GEMM with cached heuristic. Falls back to GemmEx.
** Column-major dims match existing GemmEx callers (N,M,K with opa/opb).
** epi/bias_dev: optional Lt epilogue (BIAS / RELU_BIAS); bias length = N.
*/
static int cublas_lt_gemm_half(cublasOperation_t opa, cublasOperation_t opb,
	int N, int M, int K, const float *alpha, const void *B, int ldb,
	const void *A, int lda, const float *beta, float *C, int ldc, int bf16,
	cublasLtEpilogue_t epi, const float *bias_dev, float *Dout, int out_bf16)
{
	float *D = Dout ? Dout : C;	/* D = alpha A B + beta C, out of place when Dout */
	cublasLtMatmulDesc_t op;
	cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
	cublasLtMatmulPreference_t pref;
	cublasLtMatmulHeuristicResult_t heur;
	cublasStatus_t st;
	cudaDataType_t dt;
	int returned = 0;
	int i;
	int slot = -1;
	int epi_i;
	lt_algo_entry *e;
	size_t ws_need;

	if (cublas_ensure() != 0)
		return -1;
	if (epi != CUBLASLT_EPILOGUE_DEFAULT && !bias_dev
	    && (epi & CUBLASLT_EPILOGUE_BIAS))
		return -1;
	epi_i = (int)epi;
	if (!bf16)
		return -1;	/* FP16 was removed */
	dt = CUDA_R_16BF;
	for (i = 0; i < LT_ALGO_CACHE; i++) {
		e = &g_lt_algo[i];
		if (e->valid && e->M == M && e->N == N && e->K == K
		    && e->opa == (int)opa && e->opb == (int)opb
		    && e->bf16 == bf16 && e->fp8 == 0 && e->epi == epi_i
		    && e->obf == out_bf16) {
			slot = i;
			break;
		}
		if (slot < 0 && !e->valid)
			slot = i;
	}
	if (slot < 0)
		slot = 0;
	e = &g_lt_algo[slot];

	/* Reuse Lt handles. Creating them per GEMM is slower than a short GEMM. */
	if (e->handles && e->valid && e->M == M && e->N == N && e->K == K
	    && e->opa == (int)opa && e->opb == (int)opb && e->bf16 == bf16
	    && e->fp8 == 0 && e->epi == epi_i && e->obf == out_bf16
	    && e->lda == lda && e->ldb == ldb && e->ldc == ldc) {
		if (bias_dev != e->bias) {
			(void)cublasLtMatmulDescSetAttribute(e->desc,
			    CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_dev,
			    sizeof(bias_dev));
			e->bias = bias_dev;
		}
		st = cublasLtMatmul(g_cublaslt, e->desc, alpha, B, e->layoutA,
		    A, e->layoutB, beta, C, e->layoutC, D, e->layoutC,
		    &e->algo, g_lt_workspace,
		    e->ws > 0 ? e->ws : g_lt_workspace_bytes, g_st);
		return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;
	}
	if (e->handles) {
		cublasLtMatrixLayoutDestroy(e->layoutC);
		cublasLtMatrixLayoutDestroy(e->layoutB);
		cublasLtMatrixLayoutDestroy(e->layoutA);
		cublasLtMatmulDescDestroy(e->desc);
		e->handles = 0;
		e->valid = 0;
	}

	/*
	** Match cublasGemmEx callers: m=N,n=M,k=K with first matrix = B (opa)
	** and second = A (opb). Lt: D = α op(A) op(B) + β C using those roles.
	** FAST_16BF unlocks BF16 tensor-core algos; plain COMPUTE_32F can pick
	** a slow SIMT path on some Blackwell shapes.
	*/
	{
		cublasComputeType_t cterm = CUBLAS_COMPUTE_32F_FAST_16BF;

		if (cublasLtMatmulDescCreate(&op, cterm, CUDA_R_32F)
		    != CUBLAS_STATUS_SUCCESS
		    && cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F,
		    CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
			return -1;
	}
	(void)cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
	    &opa, sizeof(opa));
	(void)cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
	    &opb, sizeof(opb));
	if (epi != CUBLASLT_EPILOGUE_DEFAULT) {
		/* cuBLASLt wants the bias in the output type: BF16 with a BF16 D. */
		cudaDataType_t bdt = out_bf16 ? dt : CUDA_R_32F;

		if (cublasLtMatmulDescSetAttribute(op,
		    CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi))
		    != CUBLAS_STATUS_SUCCESS) {
			cublasLtMatmulDescDestroy(op);
			return -1;
		}
		if (bias_dev) {
			(void)cublasLtMatmulDescSetAttribute(op,
			    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt,
			    sizeof(bdt));
			if (cublasLtMatmulDescSetAttribute(op,
			    CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_dev,
			    sizeof(bias_dev)) != CUBLAS_STATUS_SUCCESS) {
				cublasLtMatmulDescDestroy(op);
				return -1;
			}
		}
	}
	/* layoutA = GemmEx's B (first matrix), layoutB = GemmEx's A. */
	if (cublasLtMatrixLayoutCreate(&layoutA, dt,
	    opa == CUBLAS_OP_N ? N : K, opa == CUBLAS_OP_N ? K : N, ldb)
	    != CUBLAS_STATUS_SUCCESS) {
		cublasLtMatmulDescDestroy(op);
		return -1;
	}
	if (cublasLtMatrixLayoutCreate(&layoutB, dt,
	    opb == CUBLAS_OP_N ? K : M, opb == CUBLAS_OP_N ? M : K, lda)
	    != CUBLAS_STATUS_SUCCESS) {
		cublasLtMatrixLayoutDestroy(layoutA);
		cublasLtMatmulDescDestroy(op);
		return -1;
	}
	if (cublasLtMatrixLayoutCreate(&layoutC, out_bf16 ? dt : CUDA_R_32F, N, M, ldc)
	    != CUBLAS_STATUS_SUCCESS) {
		cublasLtMatrixLayoutDestroy(layoutB);
		cublasLtMatrixLayoutDestroy(layoutA);
		cublasLtMatmulDescDestroy(op);
		return -1;
	}
	if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
		cublasLtMatrixLayoutDestroy(layoutC);
		cublasLtMatrixLayoutDestroy(layoutB);
		cublasLtMatrixLayoutDestroy(layoutA);
		cublasLtMatmulDescDestroy(op);
		return -1;
	}
	ws_need = g_lt_workspace_bytes;
	(void)cublasLtMatmulPreferenceSetAttribute(pref,
	    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_need, sizeof(ws_need));

	if (!e->valid || e->M != M || e->N != N || e->K != K
	    || e->opa != (int)opa || e->opb != (int)opb || e->bf16 != bf16
	    || e->fp8 != 0 || e->epi != epi_i || e->obf != out_bf16) {
		cublasLtMatmulHeuristicResult_t heurs[16];
		cublasLtMatmulHeuristicResult_t alt_best;
		int alt_won = 0;
		float heur_ms = 1e30f;
		int hi;
		int best_i;
		float best_ms;
		cudaEvent_t ev0;
		cudaEvent_t ev1;
		float *time_D;
		int time_D_owned;

		st = cublasLtMatmulAlgoGetHeuristic(g_cublaslt, op, layoutA,
		    layoutB, layoutC, layoutC, pref, 16, heurs, &returned);
		if (st != CUBLAS_STATUS_SUCCESS || returned < 1) {
			cublasLtMatmulPreferenceDestroy(pref);
			cublasLtMatrixLayoutDestroy(layoutC);
			cublasLtMatrixLayoutDestroy(layoutB);
			cublasLtMatrixLayoutDestroy(layoutA);
			cublasLtMatmulDescDestroy(op);
			return -1;
		}
		/*
		** Time each workspace-fitting candidate once. Heuristic order
		** is often wrong on sm_120; pick the measured winner.
		**
		** The timing runs are real GEMMs. With beta == 0 they are
		** idempotent, but an accumulate call (beta == 1: dW += A^T dY)
		** would add the product once per candidate into C. Time those
		** out of place into a scratch D (Lt allows D != C), and skip
		** the timing when that scratch cannot be allocated.
		*/
		best_i = -1;
		best_ms = 1e30f;
		time_D = D;
		time_D_owned = 0;
		if (*beta != 0.0f && D == C) {
			size_t d_bytes = (size_t)M * (size_t)N * (out_bf16 ? 2 : sizeof(float));

			if (cudaMalloc(&time_D, d_bytes) == cudaSuccess)
				time_D_owned = 1;
			else
				time_D = NULL;
		}
		/*
		** ANN_LT_HEURISTIC=1 skips the timing and takes the first
		** fitting candidate: different algos round differently, so
		** tests that compare runs want a load-independent choice.
		*/
		if (time_D && !lt_heuristic_only()
		    && cudaEventCreate(&ev0) == cudaSuccess
		    && cudaEventCreate(&ev1) == cudaSuccess) {
			for (hi = 0; hi < returned; hi++) {
				float ms;
				int r;

				if (heurs[hi].workspaceSize > g_lt_workspace_bytes)
					continue;
				(void)cudaEventRecord(ev0, g_st);
				r = cublasLtMatmul(g_cublaslt, op, alpha, B,
				    layoutA, A, layoutB, beta, C, layoutC,
				    time_D, layoutC, &heurs[hi].algo,
				    g_lt_workspace,
				    heurs[hi].workspaceSize > 0
				    ? heurs[hi].workspaceSize
				    : g_lt_workspace_bytes, g_st);
				(void)cudaEventRecord(ev1, g_st);
				if (r != CUBLAS_STATUS_SUCCESS)
					continue;
				(void)cudaEventSynchronize(ev1);
				if (cudaEventElapsedTime(&ms, ev0, ev1)
				    != cudaSuccess)
					continue;
				if (ms < best_ms) {
					best_ms = ms;
					best_i = hi;
				}
			}
			heur_ms = best_ms;
			/*
			** B6: the split-K 1 version of every split-K candidate
			** (same tile, stages, swizzle), validated by AlgoCheck.
			*/
			for (hi = 0; hi < returned && best_i >= 0 && lt_split1_on(); hi++) {
				cublasLtMatmulHeuristicResult_t alt;
				int32_t sk = 1;
				uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
				size_t wr = 0;
				float ms;
				int r;

				if (heurs[hi].workspaceSize > g_lt_workspace_bytes
				    || cublasLtMatmulAlgoConfigGetAttribute(&heurs[hi].algo,
				    CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof(sk), &wr)
				    != CUBLAS_STATUS_SUCCESS || sk <= 1)
					continue;
				alt = heurs[hi];
				sk = 1;
				if (cublasLtMatmulAlgoConfigSetAttribute(&alt.algo,
				    CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof(sk))
				    != CUBLAS_STATUS_SUCCESS
				    || cublasLtMatmulAlgoConfigSetAttribute(&alt.algo,
				    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red))
				    != CUBLAS_STATUS_SUCCESS
				    || cublasLtMatmulAlgoCheck(g_cublaslt, op, layoutA,
				    layoutB, layoutC, layoutC, &alt.algo, &alt)
				    != CUBLAS_STATUS_SUCCESS
				    || alt.workspaceSize > g_lt_workspace_bytes) {
					(void)cudaGetLastError();
					continue;
				}
				(void)cudaEventRecord(ev0, g_st);
				r = cublasLtMatmul(g_cublaslt, op, alpha, B,
				    layoutA, A, layoutB, beta, C, layoutC,
				    time_D, layoutC, &alt.algo, g_lt_workspace,
				    alt.workspaceSize > 0 ? alt.workspaceSize
				    : g_lt_workspace_bytes, g_st);
				(void)cudaEventRecord(ev1, g_st);
				if (r != CUBLAS_STATUS_SUCCESS)
					continue;
				(void)cudaEventSynchronize(ev1);
				if (cudaEventElapsedTime(&ms, ev0, ev1) != cudaSuccess)
					continue;
				if (ms < best_ms) {
					best_ms = ms;
					alt_best = alt;
					alt_won = 1;
				}
			}
			cudaEventDestroy(ev0);
			cudaEventDestroy(ev1);
		}
		if (time_D_owned)
			cudaFree(time_D);
		if (best_i < 0) {
			for (hi = 0; hi < returned; hi++) {
				if (heurs[hi].workspaceSize
				    <= g_lt_workspace_bytes) {
					best_i = hi;
					break;
				}
			}
		}
		if (best_i < 0) {
			cublasLtMatmulPreferenceDestroy(pref);
			cublasLtMatrixLayoutDestroy(layoutC);
			cublasLtMatrixLayoutDestroy(layoutB);
			cublasLtMatrixLayoutDestroy(layoutA);
			cublasLtMatmulDescDestroy(op);
			return -1;
		}
		heur = heurs[best_i];
		if (alt_won) {
			static int log_on = -1;

			if (log_on < 0) {
				const char *le = getenv("ANN_LT_LOG");

				log_on = (le && le[0] == '1') ? 1 : 0;
			}
			if (log_on)
				fprintf(stderr, "[lt] %d x %d x %d epi %d: split-K 1 variant "
				    "%.3f ms vs %.3f ms\n", M, N, K, epi_i, best_ms, heur_ms);
			heur = alt_best;
		}
		e->algo = heur.algo;
		e->ws = heur.workspaceSize;
		e->M = M;
		e->N = N;
		e->K = K;
		e->opa = (int)opa;
		e->opb = (int)opb;
		e->bf16 = bf16;
		e->fp8 = 0;
		e->epi = epi_i;
		e->obf = out_bf16;
		e->lda = lda;
		e->ldb = ldb;
		e->ldc = ldc;
		e->desc = op;
		e->layoutA = layoutA;
		e->layoutB = layoutB;
		e->layoutC = layoutC;
		e->bias = bias_dev;
		e->handles = 1;
		e->valid = 1;
	}
	st = cublasLtMatmul(g_cublaslt, op, alpha, B, layoutA, A, layoutB, beta,
	    C, layoutC, D, layoutC, &e->algo,
	    g_lt_workspace, e->ws > 0 ? e->ws : g_lt_workspace_bytes, g_st);
	cublasLtMatmulPreferenceDestroy(pref);
	return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

#if CUDART_VERSION >= 12000 && defined(ANN_CUBLAS)
/*
** FP8 mixed stack: E4M3 forward, E4M3×E5M2 dgrad, BF16 wgrad, FP32 accum.
** Note: on sm_89/sm_120 Tensor Cores, cuBLASLt FP8 ONLY supports (OP_T, OP_N)
** and rejects E5M2×E5M2. So:
**   - Forward (C = A * B): B [K, N] is cached transposed as [N, K] E4M3 so
**     cuBLASLt sees layoutA=[K,N] col-major with OP_T and layoutB=A [K,M] OP_N.
**   - Dgrad (dX = dY * W^T): W [N, K] is cached as [N, K] E4M3 (layoutA=[K,N]
**     col-major with OP_T) and dY [M, K] is quantized to E5M2 (OP_N).
*/
static void *g_fp8_A;
static size_t g_fp8_A_cap;
static float *g_fp8_amax_buf;
/* LN one-pass writes E4M3 here; the next GEMM on that pointer skips quant. */
static void *g_fp8_hot_buf;
static size_t g_fp8_hot_cap;
static float *g_fp8_hot_amax;
static const ann_real *g_fp8_hot_src;
static int g_fp8_hot_n;
static int g_fp8_dgrad_fail;		/* soft-disable after repeated Lt misses */
#define FP8_WT_SLOTS	320
typedef struct {
	const ann_real	*fp32;
	void		*fp8;
	float		*scale;
	size_t		n;
	int		rows;
	int		cols;
	int		transposed;
	int		valid;
	int		e5m2;
} fp8_wt_slot;
static fp8_wt_slot g_fp8_wt[FP8_WT_SLOTS];
#define FP8_ALGO_CACHE	64
typedef struct {
	int		M, N, K;
	int		opa, opb;
	int		dt_a, dt_b;
	int		lda, ldb, ldc;
	int		epi;
	int		valid;
	int		handles;
	cublasLtMatmulAlgo_t algo;
	cublasLtMatmulDesc_t desc;
	cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
	const float	*scale_a, *scale_b;
	size_t		ws;
} fp8_algo_entry;
static fp8_algo_entry g_fp8_algo[FP8_ALGO_CACHE];

__global__ static void f32_amax_block_kernel(float *amax, const float *src, int n)
{
	__shared__ float smax[32];
	int tid = threadIdx.x;
	int idx = (blockIdx.x * blockDim.x + tid) * 4;
	int stride = gridDim.x * blockDim.x * 4;
	float vmax = 0.0f;

	for (; idx + 3 < n; idx += stride) {
		float4 v = *(const float4 *)(src + idx);
		vmax = fmaxf(vmax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
		    fmaxf(fabsf(v.z), fabsf(v.w))));
	}
	for (; idx < n; idx++)
		vmax = fmaxf(vmax, fabsf(src[idx]));
	for (int mask = 16; mask > 0; mask >>= 1)
		vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, mask));
	if ((tid & 31) == 0)
		smax[tid >> 5] = vmax;
	__syncthreads();
	if (tid < 32) {
		int nw = (blockDim.x + 31) >> 5;
		vmax = (tid < nw) ? smax[tid] : 0.0f;
		for (int mask = 16; mask > 0; mask >>= 1)
			vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, mask));
		if (tid == 0 && vmax > 0.0f)
			atomicMax((int *)amax, __float_as_int(vmax));
	}
}

__global__ static void f32_to_fp8_vec_kernel(__nv_fp8_storage_t *dst,
	const float *src, int n, const float *scale_dev, int e5m2)
{
	int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	float scale;
	float inv;

	if (base >= n)
		return;
	scale = scale_dev ? scale_dev[0] : 1.f;
	inv = (scale > 0.f) ? (1.f / scale) : 0.f;
	if (base + 3 < n && (((uintptr_t)(src + base) | (uintptr_t)(dst + base)) & 3u) == 0) {
		float4 v = *(const float4 *)(src + base);
		__nv_fp8_interpretation_t fmt = e5m2 ? __NV_E5M2 : __NV_E4M3;
		float2 lo = make_float2(v.x * inv, v.y * inv);
		float2 hi = make_float2(v.z * inv, v.w * inv);
		uint16_t p0 = __nv_cvt_float2_to_fp8x2(lo, __NV_SATFINITE, fmt);
		uint16_t p1 = __nv_cvt_float2_to_fp8x2(hi, __NV_SATFINITE, fmt);
		*(uint32_t *)(dst + base) = (uint32_t)p0 | ((uint32_t)p1 << 16);
		return;
	}
	for (int k = 0; k < 4 && base + k < n; k++) {
		dst[base + k] = __nv_cvt_float_to_fp8(src[base + k] * inv,
		    __NV_SATFINITE, e5m2 ? __NV_E5M2 : __NV_E4M3);
	}
}

/* Fused quantize + block-reduced amax for next delayed scale. */
__global__ static void f32_to_fp8_delayed_kernel(__nv_fp8_storage_t *dst,
	float *amax, const float *src, int n, const float *scale_dev, int e5m2)
{
	__shared__ float smax[32];
	int tid = threadIdx.x;
	int base = (blockIdx.x * blockDim.x + tid) * 4;
	float scale = scale_dev ? scale_dev[0] : 1.f;
	float inv = (scale > 0.f) ? (1.f / scale) : 0.f;
	float vmax = 0.0f;

	if (base + 3 < n && (((uintptr_t)(src + base) | (uintptr_t)(dst + base)) & 3u) == 0) {
		float4 v = *(const float4 *)(src + base);
		__nv_fp8_interpretation_t fmt = e5m2 ? __NV_E5M2 : __NV_E4M3;
		vmax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
		    fmaxf(fabsf(v.z), fabsf(v.w)));
		float2 lo = make_float2(v.x * inv, v.y * inv);
		float2 hi = make_float2(v.z * inv, v.w * inv);
		uint16_t p0 = __nv_cvt_float2_to_fp8x2(lo, __NV_SATFINITE, fmt);
		uint16_t p1 = __nv_cvt_float2_to_fp8x2(hi, __NV_SATFINITE, fmt);
		*(uint32_t *)(dst + base) = (uint32_t)p0 | ((uint32_t)p1 << 16);
	} else if (base < n) {
		for (int k = 0; k < 4 && base + k < n; k++) {
			float x = src[base + k];
			vmax = fmaxf(vmax, fabsf(x));
			dst[base + k] = __nv_cvt_float_to_fp8(x * inv,
			    __NV_SATFINITE, e5m2 ? __NV_E5M2 : __NV_E4M3);
		}
	}
	for (int mask = 16; mask > 0; mask >>= 1)
		vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, mask));
	if ((tid & 31) == 0)
		smax[tid >> 5] = vmax;
	__syncthreads();
	if (tid < 32) {
		int nw = (blockDim.x + 31) >> 5;
		vmax = (tid < nw) ? smax[tid] : 0.0f;
		for (int mask = 16; mask > 0; mask >>= 1)
			vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, mask));
		if (tid == 0 && vmax > 0.0f)
			atomicMax((int *)amax, __float_as_int(vmax));
	}
}

/* 32x32 shared-memory tile transpose + FP8 quantize: src [rows, cols] -> dst [cols, rows]. */
__global__ static void f32_transpose_to_fp8_kernel(__nv_fp8_storage_t *dst,
	const float *src, int rows, int cols, const float *scale_dev, int e5m2)
{
	__shared__ __nv_fp8_storage_t tile[32][33];
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	float scale = scale_dev ? scale_dev[0] : 1.f;
	float inv = (scale > 0.f) ? (1.f / scale) : 0.f;
	__nv_fp8_interpretation_t fmt = e5m2 ? __NV_E5M2 : __NV_E4M3;

	for (int j = 0; j < 32; j += 8) {
		if (x < cols && (y + j) < rows) {
			float v = src[(size_t)(y + j) * (size_t)cols + (size_t)x];
			tile[threadIdx.y + j][threadIdx.x] =
			    __nv_cvt_float_to_fp8(v * inv, __NV_SATFINITE, fmt);
		}
	}
	__syncthreads();
	x = blockIdx.y * 32 + threadIdx.x;
	y = blockIdx.x * 32 + threadIdx.y;
	for (int j = 0; j < 32; j += 8) {
		if (x < rows && (y + j) < cols) {
			dst[(size_t)(y + j) * (size_t)rows + (size_t)x] =
			    tile[threadIdx.x][threadIdx.y + j];
		}
	}
}

/* amax → scale = max(amax/max_fp8, 1e-12). E4M3:448 E5M2:57344. */
__global__ static void amax_to_scale_kernel(float *scale, const float *amax,
	float max_fp8)
{
	float a = amax ? amax[0] : 0.f;

	if (!(a > 0.f) || !isfinite(a))
		a = 1e-8f;
	a = a / max_fp8;
	if (a < 1e-12f)
		a = 1e-12f;
	scale[0] = a;
}

static int ensure_fp8_buf(void **buf, size_t *cap, size_t bytes)
{
	if (*cap >= bytes)
		return 0;
	if (*buf)
		cudaFree(*buf);
	*buf = NULL;
	*cap = 0;
	if (cudaMalloc(buf, bytes) != cudaSuccess) {
		*buf = NULL;
		return -1;
	}
	*cap = bytes;
	return 0;
}

static int fp8_scales_ensure(void)
{
	if (!g_fp8_amax_buf
	    && cudaMalloc((void **)&g_fp8_amax_buf, sizeof(float)) != cudaSuccess)
		return -1;
	if (!g_fp8_hot_amax
	    && cudaMalloc((void **)&g_fp8_hot_amax, sizeof(float)) != cudaSuccess)
		return -1;
	return 0;
}

/*
** Delayed scaling is per tensor: the scale a tensor is quantized with comes
** from its own amax of the previous microbatch. One shared scale for every
** activation (or every gradient) of the network quantized each tensor with
** the amax of whichever tensor came before it: a tensor larger than its
** predecessor saturated at 448 (E4M3) and a smaller one lost its mantissa.
** Slots are keyed on the FP32 source address (stable per layer buffer).
*/
#define FP8_ACT_SLOTS	512
typedef struct {
	const void	*key;
	int		e5m2;
	float		*scale;		/* [0] current, [1] next (wgrad swap) */
	int		cur;
} fp8_act_slot;
static fp8_act_slot g_fp8_act[FP8_ACT_SLOTS];

static fp8_act_slot *fp8_act_scale(const void *key, int e5m2)
{
	int i;
	int free_slot = -1;
	fp8_act_slot *e;
	float one[2] = { 1.f, 1.f };

	if (!key)
		return NULL;
	for (i = 0; i < FP8_ACT_SLOTS; i++) {
		e = &g_fp8_act[i];
		if (e->key == key && e->e5m2 == e5m2)
			return e;
		if (free_slot < 0 && !e->key)
			free_slot = i;
	}
	if (free_slot < 0)
		return NULL;
	e = &g_fp8_act[free_slot];
	if (!e->scale
	    && cudaMalloc((void **)&e->scale, 2 * sizeof(float)) != cudaSuccess) {
		e->scale = NULL;
		return NULL;
	}
	if (cudaMemcpy(e->scale, one, sizeof(one), cudaMemcpyHostToDevice)
	    != cudaSuccess)
		return NULL;
	e->key = key;
	e->e5m2 = e5m2;
	e->cur = 0;
	return e;
}

static void fp8_act_forget(const void *key)
{
	int i;

	for (i = 0; i < FP8_ACT_SLOTS; i++)
		if (g_fp8_act[i].key == key)
			g_fp8_act[i].key = NULL;
}

#define FP8_SCALE_CUR(e)	((e)->scale + (e)->cur)
#define FP8_SCALE_NEXT(e)	((e)->scale + ((e)->cur ^ 1))

/* The scale pointer of the LN output being quantized by the one-pass kernel. */
static float *g_fp8_hot_scale;

/* LN calls this before the one-pass kernel. Returns the E4M3 destination. */
static int fp8_hot_prepare(int n, const ann_real *y, __nv_fp8_storage_t **buf,
	float **scale, float **amax)
{
	fp8_act_slot *as;

	if (!g_fp8_fwd || n <= 0 || !y)
		return -1;
	if (fp8_scales_ensure() != 0)
		return -1;
	if (ensure_fp8_buf(&g_fp8_hot_buf, &g_fp8_hot_cap, (size_t)n) != 0)
		return -1;
	as = fp8_act_scale(y, 0);
	if (!as)
		return -1;
	if (cudaMemsetAsync(g_fp8_hot_amax, 0, sizeof(float), g_st) != cudaSuccess)
		return -1;
	g_fp8_hot_scale = FP8_SCALE_CUR(as);
	*buf = (__nv_fp8_storage_t *)g_fp8_hot_buf;
	*scale = g_fp8_hot_scale;
	*amax = g_fp8_hot_amax;
	return 0;
}

static void fp8_hot_commit(const ann_real *src, int n)
{
	/* Scale stays the pointer LN just quantized with. The consumer GEMM
	** reads it, then fp8_hot_finish overwrites it for the next LN.
	** One pointer so a CUDA graph replay stays consistent. */
	g_fp8_hot_src = src;
	g_fp8_hot_n = n;
}

static void fp8_hot_finish(void)
{
	if (!g_fp8_hot_scale || !g_fp8_hot_amax)
		return;
	amax_to_scale_kernel<<<1, 1, 0, g_st>>>(g_fp8_hot_scale, g_fp8_hot_amax, 448.f);
	g_fp8_hot_src = NULL;
	g_fp8_hot_n = 0;
}

/* Non-zero when this activation was just written by one-pass LN. */
static int fp8_hot_take(const ann_real *src, int n, void **buf, float **scale)
{
	if (!src || src != g_fp8_hot_src || n != g_fp8_hot_n || !g_fp8_hot_buf)
		return -1;
	*buf = g_fp8_hot_buf;
	*scale = g_fp8_hot_scale;
	return 0;
}

static float fp8_max_finite(int e5m2)
{
	return e5m2 ? 57344.f : 448.f;
}

/* Immediate scale: amax→scale then quant (weights / transposed weights). */
static int fp8_quant_now(void *dst_fp8, float *scale_dev, const float *src,
	int n, int e5m2)
{
	int threads = 256;
	int blocks;

	if (!dst_fp8 || !scale_dev || !src || n <= 0)
		return -1;
	if (fp8_scales_ensure() != 0)
		return -1;
	cudaMemsetAsync(g_fp8_amax_buf, 0, sizeof(float), g_st);
	blocks = (n + threads * 4 - 1) / (threads * 4);
	if (blocks > 1024)
		blocks = 1024;
	if (blocks < 1)
		blocks = 1;
	f32_amax_block_kernel<<<blocks, threads, 0, g_st>>>(g_fp8_amax_buf, src, n);
	amax_to_scale_kernel<<<1, 1, 0, g_st>>>(scale_dev, g_fp8_amax_buf,
	    fp8_max_finite(e5m2));
	blocks = (n + threads * 4 - 1) / (threads * 4);
	if (blocks < 1)
		blocks = 1;
	f32_to_fp8_vec_kernel<<<blocks, threads, 0, g_st>>>(
	    (__nv_fp8_storage_t *)dst_fp8, src, n, scale_dev, e5m2);
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static int fp8_quant_transpose_now(void *dst_fp8, float *scale_dev,
	const float *src, int rows, int cols, int e5m2)
{
	int threads = 256;
	int blocks;
	int n = rows * cols;
	dim3 grid((cols + 31) / 32, (rows + 31) / 32, 1);
	dim3 block(32, 8, 1);

	if (!dst_fp8 || !scale_dev || !src || rows <= 0 || cols <= 0)
		return -1;
	if (fp8_scales_ensure() != 0)
		return -1;
	cudaMemsetAsync(g_fp8_amax_buf, 0, sizeof(float), g_st);
	blocks = (n + threads * 4 - 1) / (threads * 4);
	if (blocks > 1024)
		blocks = 1024;
	if (blocks < 1)
		blocks = 1;
	f32_amax_block_kernel<<<blocks, threads, 0, g_st>>>(g_fp8_amax_buf, src, n);
	amax_to_scale_kernel<<<1, 1, 0, g_st>>>(scale_dev, g_fp8_amax_buf,
	    fp8_max_finite(e5m2));
	f32_transpose_to_fp8_kernel<<<grid, block, 0, g_st>>>(
	    (__nv_fp8_storage_t *)dst_fp8, src, rows, cols, scale_dev, e5m2);
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/*
** Delayed act scale: quantize with scale_dev and leave amax in
** g_fp8_amax_buf. Caller runs fp8_act_scale_commit AFTER the GEMM so the
** same pointer is what a CUDA graph replays.
*/
static int fp8_quant_tensor(void *dst_fp8, float *scale_dev, const float *src,
	int n, int e5m2)
{
	int threads = 256;
	int blocks;

	if (!dst_fp8 || !scale_dev || !src || n <= 0)
		return -1;
	if (fp8_scales_ensure() != 0)
		return -1;
	cudaMemsetAsync(g_fp8_amax_buf, 0, sizeof(float), g_st);
	blocks = (n + threads * 4 - 1) / (threads * 4);
	if (blocks < 1)
		blocks = 1;
	f32_to_fp8_delayed_kernel<<<blocks, threads, 0, g_st>>>(
	    (__nv_fp8_storage_t *)dst_fp8, g_fp8_amax_buf, src, n, scale_dev,
	    e5m2);
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static void fp8_act_scale_commit(float *scale, int e5m2)
{
	if (!scale || !g_fp8_amax_buf)
		return;
	amax_to_scale_kernel<<<1, 1, 0, g_st>>>(scale, g_fp8_amax_buf,
	    fp8_max_finite(e5m2));
}

static void fp8_wt_invalidate(const ann_real *fp32)
{
	int i;

	for (i = 0; i < FP8_WT_SLOTS; i++) {
		if (!fp32 || g_fp8_wt[i].fp32 == fp32)
			g_fp8_wt[i].valid = 0;
	}
}

static int fp8_wt_get_ex(const ann_real *fp32, int rows, int cols,
	int transposed, int e5m2, void **out_fp8, float **out_scale)
{
	fp8_wt_slot *e;
	size_t n = (size_t)rows * (size_t)cols;
	int i;
	int free_slot;

	if (!fp32 || rows <= 0 || cols <= 0 || !out_fp8 || !out_scale)
		return -1;
	free_slot = -1;
	e = NULL;
	for (i = 0; i < FP8_WT_SLOTS; i++) {
		if (g_fp8_wt[i].fp32 == fp32
		    && g_fp8_wt[i].transposed == transposed
		    && g_fp8_wt[i].e5m2 == e5m2) {
			e = &g_fp8_wt[i];
			break;
		}
		if (free_slot < 0 && !g_fp8_wt[i].fp32)
			free_slot = i;
	}
	if (!e) {
		if (free_slot < 0)
			free_slot = 0;
		e = &g_fp8_wt[free_slot];
		if (e->fp8)
			cudaFree(e->fp8);
		if (e->scale)
			cudaFree(e->scale);
		e->fp8 = NULL;
		e->scale = NULL;
		e->fp32 = fp32;
		e->n = n;
		e->rows = rows;
		e->cols = cols;
		e->transposed = transposed;
		e->e5m2 = e5m2;
		e->valid = 0;
		if (cudaMalloc(&e->fp8, n) != cudaSuccess
		    || cudaMalloc((void **)&e->scale, sizeof(float))
		    != cudaSuccess) {
			if (e->fp8)
				cudaFree(e->fp8);
			if (e->scale)
				cudaFree(e->scale);
			e->fp8 = NULL;
			e->scale = NULL;
			e->fp32 = NULL;
			return -1;
		}
	}
	if (!e->valid || e->n != n || e->rows != rows || e->cols != cols) {
		if (e->n != n) {
			cudaFree(e->fp8);
			e->fp8 = NULL;
			if (cudaMalloc(&e->fp8, n) != cudaSuccess) {
				e->fp32 = NULL;
				return -1;
			}
			e->n = n;
		}
		e->rows = rows;
		e->cols = cols;
		if (transposed) {
			if (fp8_quant_transpose_now(e->fp8, e->scale, fp32,
			    rows, cols, e5m2) != 0)
				return -1;
		} else {
			if (fp8_quant_now(e->fp8, e->scale, fp32, (int)n, e5m2)
			    != 0)
				return -1;
		}
		e->valid = 1;
	}
	*out_fp8 = e->fp8;
	*out_scale = e->scale;
	return 0;
}

/*
** cuBLASLt FP8×FP8 → FP32. GemmEx argument order.
** Notice: on sm_89/sm_120, cuBLASLt requires opa == CUBLAS_OP_T and
** opb == CUBLAS_OP_N, and does not allow E5M2×E5M2.
*/
static int cublas_lt_gemm_fp8_ex(cublasOperation_t opa, cublasOperation_t opb,
	int N, int M, int K, const float *alpha, const void *B, int ldb,
	const float *scale_B, cudaDataType_t dt_B, const void *A, int lda,
	const float *scale_A, cudaDataType_t dt_A, const float *beta, float *C,
	int ldc)
{
	cublasLtMatmulDesc_t op;
	cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
	cublasLtMatmulPreference_t pref;
	cublasLtMatmulHeuristicResult_t heurs[8];
	cublasStatus_t st;
	int8_t fast_accum = 1;
	int returned = 0;
	int i;
	int slot = -1;
	fp8_algo_entry *e;
	size_t ws_need;

	if (cublas_ensure() != 0)
		return -1;
	for (i = 0; i < FP8_ALGO_CACHE; i++) {
		e = &g_fp8_algo[i];
		if (e->valid && e->M == M && e->N == N && e->K == K
		    && e->opa == (int)opa && e->opb == (int)opb
		    && e->dt_a == (int)dt_B && e->dt_b == (int)dt_A) {
			slot = i;
			break;
		}
		if (slot < 0 && !e->valid)
			slot = i;
	}
	if (slot < 0)
		slot = 0;
	e = &g_fp8_algo[slot];

	/*
	** Descriptor create/destroy is slower than the FP8 GEMM. Keep the
	** handles on the shape slot and only refresh scale pointers.
	*/
	if (e->handles && e->valid && e->M == M && e->N == N && e->K == K
	    && e->opa == (int)opa && e->opb == (int)opb
	    && e->dt_a == (int)dt_B && e->dt_b == (int)dt_A
	    && e->lda == lda && e->ldb == ldb && e->ldc == ldc) {
		if (e->scale_b != scale_B) {
			(void)cublasLtMatmulDescSetAttribute(e->desc,
			    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_B,
			    sizeof(scale_B));
			e->scale_b = scale_B;
		}
		if (e->scale_a != scale_A) {
			(void)cublasLtMatmulDescSetAttribute(e->desc,
			    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_A,
			    sizeof(scale_A));
			e->scale_a = scale_A;
		}
		st = cublasLtMatmul(g_cublaslt, e->desc, alpha, B, e->layoutA,
		    A, e->layoutB, beta, C, e->layoutC, C, e->layoutC,
		    &e->algo, g_lt_workspace,
		    e->ws > 0 ? e->ws : g_lt_workspace_bytes, g_st);
		return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;
	}
	if (e->handles) {
		cublasLtMatrixLayoutDestroy(e->layoutC);
		cublasLtMatrixLayoutDestroy(e->layoutB);
		cublasLtMatrixLayoutDestroy(e->layoutA);
		cublasLtMatmulDescDestroy(e->desc);
		e->handles = 0;
		e->valid = 0;
	}
	if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)
	    != CUBLAS_STATUS_SUCCESS)
		return -1;
	(void)cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
	    &opa, sizeof(opa));
	(void)cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
	    &opb, sizeof(opb));
	(void)cublasLtMatmulDescSetAttribute(op,
	    CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum));
	(void)cublasLtMatmulDescSetAttribute(op,
	    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_B, sizeof(scale_B));
	(void)cublasLtMatmulDescSetAttribute(op,
	    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_A, sizeof(scale_A));
	if (cublasLtMatrixLayoutCreate(&layoutA, dt_B,
	    opa == CUBLAS_OP_N ? N : K, opa == CUBLAS_OP_N ? K : N, ldb)
	    != CUBLAS_STATUS_SUCCESS) {
		cublasLtMatmulDescDestroy(op);
		return -1;
	}
	if (cublasLtMatrixLayoutCreate(&layoutB, dt_A,
	    opb == CUBLAS_OP_N ? K : M, opb == CUBLAS_OP_N ? M : K, lda)
	    != CUBLAS_STATUS_SUCCESS) {
		cublasLtMatrixLayoutDestroy(layoutA);
		cublasLtMatmulDescDestroy(op);
		return -1;
	}
	if (cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, N, M, ldc)
	    != CUBLAS_STATUS_SUCCESS) {
		cublasLtMatrixLayoutDestroy(layoutB);
		cublasLtMatrixLayoutDestroy(layoutA);
		cublasLtMatmulDescDestroy(op);
		return -1;
	}
	if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
		cublasLtMatrixLayoutDestroy(layoutC);
		cublasLtMatrixLayoutDestroy(layoutB);
		cublasLtMatrixLayoutDestroy(layoutA);
		cublasLtMatmulDescDestroy(op);
		return -1;
	}
	ws_need = g_lt_workspace_bytes;
	(void)cublasLtMatmulPreferenceSetAttribute(pref,
	    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_need, sizeof(ws_need));
	if (!e->valid || e->M != M || e->N != N || e->K != K
	    || e->opa != (int)opa || e->opb != (int)opb
	    || e->dt_a != (int)dt_B || e->dt_b != (int)dt_A) {
		int hi, best_i = -1;

		st = cublasLtMatmulAlgoGetHeuristic(g_cublaslt, op, layoutA,
		    layoutB, layoutC, layoutC, pref, 8, heurs, &returned);
		if (st != CUBLAS_STATUS_SUCCESS || returned < 1) {
			cublasLtMatmulPreferenceDestroy(pref);
			cublasLtMatrixLayoutDestroy(layoutC);
			cublasLtMatrixLayoutDestroy(layoutB);
			cublasLtMatrixLayoutDestroy(layoutA);
			cublasLtMatmulDescDestroy(op);
			return -1;
		}
		if (returned > 1 && !lt_heuristic_only()) {
			cudaStreamCaptureStatus cs =
			    cudaStreamCaptureStatusNone;
			cudaEvent_t ev0, ev1;
			float best_ms = 1e30f;
			float *time_D = C;
			int time_D_owned = 0;

			/* Same rule as the BF16 path: beta != 0 times out of place. */
			if (*beta != 0.0f) {
				if (cudaMalloc(&time_D, (size_t)M * (size_t)N
				    * sizeof(float)) == cudaSuccess)
					time_D_owned = 1;
				else
					time_D = NULL;
			}
			if (time_D && cudaStreamIsCapturing(g_st, &cs) == cudaSuccess
			    && cs == cudaStreamCaptureStatusNone
			    && cudaEventCreate(&ev0) == cudaSuccess
			    && cudaEventCreate(&ev1) == cudaSuccess) {
				for (hi = 0; hi < returned; hi++) {
					float ms = 0.0f;
					cublasStatus_t r;

					if (heurs[hi].workspaceSize
					    > g_lt_workspace_bytes)
						continue;
					(void)cudaEventRecord(ev0, g_st);
					r = cublasLtMatmul(g_cublaslt, op,
					    alpha, B, layoutA, A, layoutB, beta,
					    C, layoutC, time_D, layoutC,
					    &heurs[hi].algo, g_lt_workspace,
					    heurs[hi].workspaceSize > 0
					    ? heurs[hi].workspaceSize
					    : g_lt_workspace_bytes, g_st);
					(void)cudaEventRecord(ev1, g_st);
					if (r != CUBLAS_STATUS_SUCCESS)
						continue;
					(void)cudaEventSynchronize(ev1);
					if (cudaEventElapsedTime(&ms, ev0, ev1)
					    == cudaSuccess && ms < best_ms) {
						best_ms = ms;
						best_i = hi;
					}
				}
				cudaEventDestroy(ev0);
				cudaEventDestroy(ev1);
			}
			if (time_D_owned)
				cudaFree(time_D);
		}
		if (best_i < 0) {
			for (hi = 0; hi < returned; hi++) {
				if (heurs[hi].workspaceSize
				    <= g_lt_workspace_bytes) {
					best_i = hi;
					break;
				}
			}
		}
		if (best_i < 0) {
			cublasLtMatmulPreferenceDestroy(pref);
			cublasLtMatrixLayoutDestroy(layoutC);
			cublasLtMatrixLayoutDestroy(layoutB);
			cublasLtMatrixLayoutDestroy(layoutA);
			cublasLtMatmulDescDestroy(op);
			return -1;
		}
		e->algo = heurs[best_i].algo;
		e->ws = heurs[best_i].workspaceSize;
		e->M = M;
		e->N = N;
		e->K = K;
		e->opa = (int)opa;
		e->opb = (int)opb;
		e->dt_a = (int)dt_B;
		e->dt_b = (int)dt_A;
		e->lda = lda;
		e->ldb = ldb;
		e->ldc = ldc;
		e->desc = op;
		e->layoutA = layoutA;
		e->layoutB = layoutB;
		e->layoutC = layoutC;
		e->scale_a = scale_A;
		e->scale_b = scale_B;
		e->handles = 1;
		e->valid = 1;
	}
	st = cublasLtMatmul(g_cublaslt, op, alpha, B, layoutA, A, layoutB, beta,
	    C, layoutC, C, layoutC, &e->algo,
	    g_lt_workspace, e->ws > 0 ? e->ws : g_lt_workspace_bytes, g_st);
	cublasLtMatmulPreferenceDestroy(pref);
	/* desc and layouts stay on the shape slot. */
	return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}
#endif /* CUDART>=12000 FP8 */

__global__ static void f32_to_bf16_kernel(__nv_bfloat16 *dst, const float *src,
	int n)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

	if (i + 3 < n) {
		float4 v = *(const float4 *)(src + i);

		dst[i] = __float2bfloat16(v.x);
		dst[i + 1] = __float2bfloat16(v.y);
		dst[i + 2] = __float2bfloat16(v.z);
		dst[i + 3] = __float2bfloat16(v.w);
	} else {
		for (; i < n; i++)
			dst[i] = __float2bfloat16(src[i]);
	}
}

static void *g_half_A;
static void *g_half_B;
static size_t g_half_A_cap;
static size_t g_half_B_cap;

/* Persistent FP16/BF16 twins for weight mats (stable across gemms in a step). */
#define HALF_WT_SLOTS	128
typedef struct {
	const ann_real	*fp32;
	void		*half;
	size_t		n;
	int		valid;
	int		bf16;
} half_wt;
static half_wt g_half_wt[HALF_WT_SLOTS];
/* Bumped when any cached weight/bias twin must be rebuilt (Adam scatter). */
static unsigned g_wt_epoch;
/* Flat Adam BF16 twin of d_param (served by half_wt_get for aliased mats). */
static void *g_adam_bf16;
static const ann_real *g_adam_fp32;
static int g_adam_flat_n;

extern "C" void gpu_adam_set_bf16_flat(void *bf16, const ann_real *fp32, int n)
{
	g_adam_bf16 = bf16;
	g_adam_fp32 = fp32;
	g_adam_flat_n = (bf16 && fp32 && n > 0) ? n : 0;
}
/*
** Short-lived BF16 twins for the forward producer→consumer hop (LN→GEMM,
** FFN hidden→W2). Two slots so LN and FFN hidden do not clobber each other
** inside one block. Not a full per-layer residency (that doubles act VRAM).
*/
#define ACT_BF16_SLOTS	64	/* ln1, ln2, hidden, concat of every layer */
typedef struct {
	const ann_real	*fp32;
	void		*half;
	size_t		cap;
	int		n;
	int		live;
	int		bf16_only;	/* the GEMM wrote the twin itself; the FP32 buffer holds nothing */
} act_bf16_slot;
static act_bf16_slot g_act_bf16[ACT_BF16_SLOTS];
static int g_act_bf16_rr;
/*
** One-shot BF16 twin of the last gradient converted by a weight-gradient
** GEMM (dW += A^T G): the data-gradient GEMM that follows (dX = G W^T)
** converts the same G again. The slot is consumed by that one call and
** cleared, so it can never outlive a rewrite of G.
*/
static int ensure_half_buf(void **buf, size_t *cap, size_t bytes);
static int half_wt_convert(void *dst, const ann_real *src, int n, int bf16);

#define GRAD_TWIN_SLOTS	9	/* 8 round robin (dq, dk, dv, the LN-backward outputs, dh...) + the in-place head delta */
#define GRAD_TWIN_RR	8	/* the round-robin slots */
#define GRAD_TWIN_BIG_N	((size_t)16 * 1024 * 1024)	/* a twin above this (the head's delta) is refused */
/*
** The head's BF16 delta written in place over its own FP32 logits rows by
** the fused softmax-CE kernel: half points into the logits buffer (borrowed,
** never allocated or freed through this slot) and ld is 2 V.
*/
#define GRAD_TWIN_INPLACE	8
typedef struct {
	void		*half;
	size_t		cap;
	const ann_real	*src;
	size_t		n;
	int		ld;	/* BF16 row stride in elements; 0 = dense (the GEMM's own) */
} grad_twin_slot;
static grad_twin_slot g_grad_twin[GRAD_TWIN_SLOTS];
/* Bit 1: the head wgrad ran on the in-place twin; bit 2: the dgrad did. */
static int g_ce_inplace_used;
static int g_grad_twin_rr;
static long g_grad_twin_hit;
static long g_grad_twin_miss;

/*
** Drop the pending twins of src other than keep: a new twin of the same
** buffer supersedes them. The in-place residual gradient (block.c B4)
** rewrites one buffer several times per block, and grad_twin_take serves
** the first slot that matches, which could be an older, unconsumed twin.
*/
static void grad_twin_drop_src(const ann_real *src, const grad_twin_slot *keep)
{
	int i;

	for (i = 0; i < GRAD_TWIN_SLOTS; i++)
		if (i != GRAD_TWIN_INPLACE && &g_grad_twin[i] != keep
		    && g_grad_twin[i].src == src) {
			g_grad_twin[i].src = NULL;
			g_grad_twin[i].n = 0;
		}
}

/* Convert G into a one-shot twin slot (round robin); NULL on failure. */
static void *grad_twin_put(const ann_real *src, size_t n)
{
	grad_twin_slot *e;

	/*
	** The head's delta (rows x vocab, 134 MB in BF16) would size every
	** round-robin slot to it: it gets no twin here (the fused CE kernel
	** writes it in place over the logits, GRAD_TWIN_INPLACE), which keeps
	** the full run under the 85% VRAM line.
	*/
	if (n > GRAD_TWIN_BIG_N)
		return NULL;
	e = &g_grad_twin[g_grad_twin_rr % GRAD_TWIN_RR];
	g_grad_twin_rr++;
	if (ensure_half_buf(&e->half, &e->cap, n * 2) != 0
	    || half_wt_convert(e->half, src, (int)n, 1) != 0) {
		e->src = NULL;
		e->n = 0;
		return NULL;
	}
	grad_twin_drop_src(src, e);
	e->src = src;
	e->n = n;
	e->ld = 0;
	return e->half;
}

/*
** Take (and clear) the twin of src if one is pending; *ld = its BF16 row
** stride, or dense_ld when the twin is packed.
*/
static void *grad_twin_take(const ann_real *src, size_t n, int dense_ld,
	int *ld)
{
	int i;

	for (i = 0; i < GRAD_TWIN_SLOTS; i++) {
		if (g_grad_twin[i].src == src && g_grad_twin[i].n == n) {
			g_grad_twin[i].src = NULL;
			g_grad_twin[i].n = 0;
			*ld = g_grad_twin[i].ld ? g_grad_twin[i].ld : dense_ld;
			return g_grad_twin[i].half;
		}
	}
	return NULL;
}

/*
** Reserve a twin slot for src without converting: a kernel fills it (the
** fused ReLU backward writes the masked BF16 dh straight into it). No size
** cap: one 32 MB slot for dh at 4096 x 4096.
*/
static void *grad_twin_reserve(const ann_real *src, size_t n)
{
	grad_twin_slot *e = &g_grad_twin[g_grad_twin_rr % GRAD_TWIN_RR];

	g_grad_twin_rr++;
	if (ensure_half_buf(&e->half, &e->cap, n * 2) != 0) {
		e->src = NULL;
		e->n = 0;
		return NULL;
	}
	grad_twin_drop_src(src, e);
	e->src = src;
	e->n = n;
	e->ld = 0;
	return e->half;
}

/* The pending twin of src without clearing it (the wgrad reads it first). */
static void *grad_twin_peek(const ann_real *src, size_t n, int dense_ld,
	int *ld)
{
	int i;

	for (i = 0; i < GRAD_TWIN_SLOTS; i++)
		if (g_grad_twin[i].src == src && g_grad_twin[i].n == n) {
			*ld = g_grad_twin[i].ld ? g_grad_twin[i].ld : dense_ld;
			return g_grad_twin[i].half;
		}
	return NULL;
}

/*
** B3: the BF16 [dQ|dK|dV] rows ([N][3 D], row stride 3 D) of the last FA2
** attention backward, keyed on its FP32 dq buffer. The backward kernel
** writes the dK / dV column blocks, a strided conversion the dQ block; the
** QKV projection backward (gpu_attn_qkv_bwd) consumes them in one dgrad
** GEMM with K = 3 D and one strided-batched wgrad. One shot, like the
** gradient twins: live from that backward to that projection only.
*/
typedef struct {
	void		*buf;
	size_t		cap;
	const ann_real	*key;
	size_t		n;	/* N x D: the size of one block */
	int		live;
} qkv_d16_state;
static qkv_d16_state g_qkv_d16;

/*
** Drop every pending gradient twin: called when a microbatch's backward is
** done, so a twin left by a producer without a GEMM consumer (the first
** block's LayerNorm backward) can never be served for whatever the next
** microbatch writes into that buffer.
*/
extern "C" void gpu_grad_twins_clear(void)
{
	int i;

	for (i = 0; i < GRAD_TWIN_SLOTS; i++) {
		g_grad_twin[i].src = NULL;
		g_grad_twin[i].n = 0;
	}
	g_qkv_d16.key = NULL;
	g_qkv_d16.live = 0;
	/*
	** Graph mode: every microbatch then reserves the same slots in the
	** same order, so a slot sized by the uncaptured microbatch is never
	** grown inside a capture.
	*/
	if (g_graph_mode)
		g_grad_twin_rr = 0;
}

static void grad_twin_forget(const void *p)
{
	int i;

	for (i = 0; i < GRAD_TWIN_SLOTS; i++)
		if (g_grad_twin[i].src == (const ann_real *)p) {
			g_grad_twin[i].src = NULL;
			g_grad_twin[i].n = 0;
		}
}

static int ensure_half_buf(void **buf, size_t *cap, size_t bytes)
{
	if (*cap >= bytes)
		return 0;
	if (*buf)
		cudaFree(*buf);
	*buf = NULL;
	*cap = 0;
	if (cudaMalloc(buf, bytes) != cudaSuccess) {
		*buf = NULL;
		return -1;
	}
	*cap = bytes;
	return 0;
}

/* Grow scratch once to peak FFN shape so OpenMP never races cudaFree. */
static void half_scratch_warm(void)
{
	static int warmed;

	if (warmed)
		return;
	(void)ensure_half_buf(&g_half_A, &g_half_A_cap,
	    (size_t)8192 * 4096 * 2);
	(void)ensure_half_buf(&g_half_B, &g_half_B_cap,
	    (size_t)4096 * 4096 * 2);
	warmed = 1;
}

static void half_wt_invalidate(const ann_real *fp32)
{
	int i;

	if (!fp32)
		return;
	for (i = 0; i < HALF_WT_SLOTS; i++) {
		if (g_half_wt[i].fp32 == fp32)
			g_half_wt[i].valid = 0;
	}
}

extern "C" void gpu_half_wt_invalidate(const ann_real *fp32)
{
	int i;

	g_wt_epoch++;
#if CUDART_VERSION >= 12000 && defined(ANN_CUBLAS)
	fp8_wt_invalidate(fp32);
#endif
	if (!fp32) {
		for (i = 0; i < HALF_WT_SLOTS; i++)
			g_half_wt[i].valid = 0;
		return;
	}
	half_wt_invalidate(fp32);
}

static int op_begin_nested(int op);
static void op_end(int i);
#define OP_CONVERT_ID	17	/* == OP_CONVERT; enum is defined further down */
#define OP_ATT_PACK_ID	18
#define OP_ATT_GEMM_ID	19
#define OP_ATT_SMX_ID	20
/* Nested sub-step timer (inside attn_fwd / attn_bwd scopes). */
struct OpNest {
	int	i;
	OpNest(int op) { i = op_begin_nested(op); }
	~OpNest() { op_end(i); }
};

static int half_wt_convert(void *dst, const ann_real *src, int n, int bf16)
{
	int threads = 256;
	int blocks;
	int prof = op_begin_nested(OP_CONVERT_ID);
	int rc;

	if (bf16) {
		/* Vectorized kernel: 4 elems/thread. */
		blocks = (n + threads * 4 - 1) / (threads * 4);
		if (blocks < 1)
			blocks = 1;
		f32_to_bf16_kernel<<<blocks, threads, 0, g_st>>>(
		    (__nv_bfloat16 *)dst, src, n);
		rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
		op_end(prof);
		return rc;
	}
	/* FP16 was removed: only the BF16 conversion exists. */
	op_end(prof);
	return -1;
}

extern "C" int gpu_adam_seed_bf16_flat(void *bf16, const ann_real *fp32, int n)
{
	if (!bf16 || !fp32 || n <= 0)
		return -1;
	return half_wt_convert(bf16, fp32, n, 1);
}

/* The slot for fp32 (existing, else a free one, else round robin) with a buffer of n elements; NULL on failure. */
static act_bf16_slot *act_bf16_slot_for(const ann_real *fp32, int n)
{
	act_bf16_slot *e;
	int i;

	if (!fp32 || n <= 0 || g_compute_prec != GPU_PREC_BF16)
		return NULL;
	e = NULL;
	for (i = 0; i < ACT_BF16_SLOTS; i++) {
		if (g_act_bf16[i].fp32 == fp32) {
			e = &g_act_bf16[i];
			break;
		}
	}
	if (!e) {
		/*
		** New tensor: a free slot (its buffer freed by gpu_free) whose
		** buffer fits, else any free slot, else round robin. Pointer
		** keys keep each training tensor in its own slot, so this
		** only runs for eval caches; slots over twice the size shrink
		** so eval-sized buffers do not pin 32 MB each forever.
		*/
		int best = -1;

		for (i = 0; i < ACT_BF16_SLOTS; i++) {
			if (g_act_bf16[i].fp32)
				continue;
			if (g_act_bf16[i].cap >= (size_t)n * 2
			    && (best < 0 || g_act_bf16[i].cap
			    < g_act_bf16[best].cap))
				best = i;
			else if (best < 0 || (g_act_bf16[best].cap < (size_t)n * 2
			    && g_act_bf16[i].cap > g_act_bf16[best].cap))
				best = i;
		}
		if (best >= 0)
			e = &g_act_bf16[best];
		else {
			e = &g_act_bf16[g_act_bf16_rr % ACT_BF16_SLOTS];
			g_act_bf16_rr++;
		}
		if (e->half && e->cap > (size_t)n * 4) {
			cudaFree(e->half);
			e->half = NULL;
			e->cap = 0;
		}
	}
	if (ensure_half_buf(&e->half, &e->cap, (size_t)n * 2) != 0) {
		e->live = 0;
		e->fp32 = NULL;
		return NULL;
	}
	e->fp32 = fp32;
	e->n = n;
	e->live = 0;
	e->bf16_only = 0;
	return e;
}

static void act_bf16_publish(const ann_real *fp32, int n)
{
	act_bf16_slot *e = act_bf16_slot_for(fp32, n);

	if (!e)
		return;
	if (half_wt_convert(e->half, fp32, n, 1) != 0) {
		e->live = 0;
		return;
	}
	e->live = 1;
}

/*
** A twin buffer for fp32 that a GEMM writes directly (BF16 output): no
** conversion, and the FP32 buffer stays unwritten (bf16_only). NULL when
** no slot; the caller marks it live once the GEMM has run.
*/
static act_bf16_slot *act_bf16_reserve(const ann_real *fp32, int n)
{
	act_bf16_slot *e = act_bf16_slot_for(fp32, n);

	if (e)
		e->bf16_only = 1;
	return e;
}

extern "C" int gpu_act_bf16_only(const mat *m)
{
	int i;

	if (!m || !m->d_data)
		return 0;
	for (i = 0; i < ACT_BF16_SLOTS; i++)
		if (g_act_bf16[i].fp32 == m->d_data && g_act_bf16[i].live)
			return g_act_bf16[i].bf16_only;
	return 0;
}

/*
** ANN_TWIN_CHECK=1: reconvert a cached twin and compare it with the cache
** before serving it (debug only; synchronises). Reports stale twins.
*/
static int twin_check_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_TWIN_CHECK");

		on = (e && e[0] == '1') ? 1 : 0;
	}
	return on;
}

static void twin_check(const char *what, const void *twin,
	const ann_real *fp32, size_t n, int bf16)
{
	void *fresh = NULL;
	unsigned char *a;
	unsigned char *b;
	size_t bytes = n * 2;
	size_t i;
	size_t bad = 0;

	if (!twin_check_on() || !twin || !fp32 || n == 0)
		return;
	if (cudaMalloc(&fresh, bytes) != cudaSuccess)
		return;
	if (half_wt_convert(fresh, fp32, (int)n, bf16) != 0) {
		cudaFree(fresh);
		return;
	}
	a = (unsigned char *)malloc(bytes);
	b = (unsigned char *)malloc(bytes);
	if (a && b && cudaMemcpy(a, twin, bytes, cudaMemcpyDeviceToHost)
	    == cudaSuccess
	    && cudaMemcpy(b, fresh, bytes, cudaMemcpyDeviceToHost)
	    == cudaSuccess) {
		for (i = 0; i < bytes; i++)
			bad += a[i] != b[i];
		if (bad)
			fprintf(stderr, "[twin] STALE %s fp32=%p n=%zu bytes_diff=%zu\n",
			    what, (const void *)fp32, n, bad);
	}
	free(a);
	free(b);
	cudaFree(fresh);
}

static const void *act_bf16_get(const ann_real *fp32, int n)
{
	int i;

	if (!fp32 || n <= 0)
		return NULL;
	for (i = 0; i < ACT_BF16_SLOTS; i++) {
		if (g_act_bf16[i].live && g_act_bf16[i].fp32 == fp32
		    && g_act_bf16[i].n == n) {
			if (!g_act_bf16[i].bf16_only)
				twin_check("act", g_act_bf16[i].half, fp32, (size_t)n, 1);
			return g_act_bf16[i].half;
		}
	}
	return NULL;
}

/*
** Cached half for a device FP32 weight buffer. Activations should keep using
** the scratch g_half_A path (they change every gemm).
*/
static void *half_wt_get(const ann_real *fp32, size_t n, int bf16)
{
	int i;
	int free_slot = -1;
	half_wt *e;

	if (!fp32 || n == 0)
		return NULL;
	/* Flat Adam twin: no per-mat convert after the fused Adam write. */
	if (bf16 && g_adam_bf16 && g_adam_fp32 && g_adam_flat_n > 0
	    && fp32 >= g_adam_fp32
	    && fp32 + n <= g_adam_fp32 + g_adam_flat_n) {
		twin_check("flat", (char *)g_adam_bf16
		    + (size_t)(fp32 - g_adam_fp32) * 2u, fp32, n, 1);
		return (char *)g_adam_bf16
		    + (size_t)(fp32 - g_adam_fp32) * 2u;
	}
	/*
	** With the flat registered every trained weight is inside it. Any
	** other pointer here is an activation (per-head K/V, scores...):
	** it changes every microbatch, so it must not get a persistent twin.
	*/
	if (bf16 && g_adam_flat_n > 0)
		return NULL;
	for (i = 0; i < HALF_WT_SLOTS; i++) {
		if (g_half_wt[i].fp32 == fp32) {
			e = &g_half_wt[i];
			if (e->n < n || e->bf16 != bf16) {
				if (e->half)
					cudaFree(e->half);
				e->half = NULL;
				e->n = 0;
				e->valid = 0;
				if (cudaMalloc(&e->half, n * 2) != cudaSuccess) {
					e->half = NULL;
					return NULL;
				}
				e->n = n;
				e->bf16 = bf16;
			}
			if (!e->valid) {
				if (half_wt_convert(e->half, fp32, (int)n, bf16)
				    != 0)
					return NULL;
				e->valid = 1;
			} else
				twin_check("wt", e->half, fp32, n, bf16);
			return e->half;
		}
		if (free_slot < 0 && !g_half_wt[i].fp32)
			free_slot = i;
	}
	if (free_slot < 0)
		return NULL;
	e = &g_half_wt[free_slot];
	e->fp32 = fp32;
	e->n = n;
	e->bf16 = bf16;
	e->valid = 0;
	e->half = NULL;
	if (cudaMalloc(&e->half, n * 2) != cudaSuccess) {
		e->fp32 = NULL;
		e->half = NULL;
		e->n = 0;
		return NULL;
	}
	if (half_wt_convert(e->half, fp32, (int)n, bf16) != 0) {
		cudaFree(e->half);
		e->fp32 = NULL;
		e->half = NULL;
		e->n = 0;
		return NULL;
	}
	e->valid = 1;
	return e->half;
}

static int cublas_gemm_ex_bf16(cublasOperation_t opa, cublasOperation_t opb,
	int N, int M, int K, const float *alpha, const void *B, int ldb,
	const void *A, int lda, const float *beta, float *C, int ldc)
{
	if (cublas_lt_gemm_half(opa, opb, N, M, K, alpha, B, ldb, A, lda, beta,
	    C, ldc, 1, CUBLASLT_EPILOGUE_DEFAULT, NULL, NULL, 0) == 0)
		return 0;
	/* FAST_16BF: plain COMPUTE_32F can pick SIMT on Blackwell. */
	return cublasGemmEx(g_cublas, opa, opb, N, M, K, alpha, B, CUDA_R_16BF,
	    ldb, A, CUDA_R_16BF, lda, beta, C, CUDA_R_32F, ldc,
	    CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
	    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

/*
** Row-major C = A B. If hA_pre is non-NULL it is already FP16/BF16 of A
** (nA elems); skip the A convert (QKV reuses one X convert for three GEMMs).
*/
static int cublas_gemm_rm_ha(ann_real *dC, const ann_real *dA, const ann_real *dB,
	int M, int N, int K, const void *hA_pre)
{
	const ann_real alpha = 1.0f;
	const ann_real beta = 0.0f;
	size_t nA;
	size_t nB;
	int use_bf16;
	void *hA_use;
	void *hB_use;
	int rc;

	if (cublas_ensure() != 0)
		return -1;
	pthread_mutex_lock(&g_gemm_mu);
	half_scratch_warm();
	use_bf16 = (g_compute_prec == GPU_PREC_BF16);
	if (g_compute_prec == GPU_PREC_FP32) {
		rc = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
		    &alpha, dB, N, dA, K, &beta, dC, N)
		    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
		pthread_mutex_unlock(&g_gemm_mu);
		return rc;
	}
	nA = (size_t)M * (size_t)K;
	nB = (size_t)K * (size_t)N;
#if CUDART_VERSION >= 12000
	/*
	** FP8 forward: E4M3×E4M3, FP32 accumulate. Cached transposed FP8
	** weights [N, K] row-major (= [K, N] col-major, OP_T); delayed scale
	** on activations.
	*/
	if (g_fp8_fwd) {
		void *wt_fp8 = NULL;
		float *wt_scale = NULL;
		void *act8 = NULL;
		float *act_scale = NULL;
		int hot = 0;
		fp8_act_slot *as = fp8_act_scale(dA, 0);

		hot = (fp8_hot_take(dA, (int)nA, &act8, &act_scale) == 0);
		if (!hot && as)
			act_scale = FP8_SCALE_CUR(as);
		if (act_scale && (K & 15) == 0 && (N & 15) == 0 && (M & 15) == 0
		    && fp8_scales_ensure() == 0
		    && fp8_wt_get_ex(dB, K, N, 1, 0, &wt_fp8, &wt_scale) == 0
		    && (hot || (ensure_fp8_buf(&g_fp8_A, &g_fp8_A_cap, nA) == 0
		    && fp8_quant_tensor(g_fp8_A, act_scale, dA, (int)nA, 0)
		    == 0))
		    && cublas_lt_gemm_fp8_ex(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
		    &alpha, wt_fp8, K, wt_scale, CUDA_R_8F_E4M3,
		    hot ? act8 : g_fp8_A, K, act_scale,
		    CUDA_R_8F_E4M3, &beta, dC, N) == 0) {
			if (hot)
				fp8_hot_finish();
			else
				fp8_act_scale_commit(act_scale, 0);
			pthread_mutex_unlock(&g_gemm_mu);
			return 0;
		}
		if (hot) {
			g_fp8_hot_src = NULL;
			g_fp8_hot_n = 0;
		}
		/* Fall through to BF16 if FP8 algo unavailable. */
	}
#endif
	hA_use = (void *)hA_pre;
	if (!hA_use && use_bf16)
		hA_use = (void *)act_bf16_get(dA, (int)nA);
	if (!hA_use) {
		if (ensure_half_buf(&g_half_A, &g_half_A_cap, nA * 2) != 0) {
			pthread_mutex_unlock(&g_gemm_mu);
			return -1;
		}
		if (half_wt_convert(g_half_A, dA, (int)nA, use_bf16) != 0) {
			pthread_mutex_unlock(&g_gemm_mu);
			return -1;
		}
		hA_use = g_half_A;
	}
	hB_use = half_wt_get(dB, nB, use_bf16);
	if (!hB_use) {
		if (ensure_half_buf(&g_half_B, &g_half_B_cap, nB * 2) != 0) {
			pthread_mutex_unlock(&g_gemm_mu);
			return -1;
		}
		if (half_wt_convert(g_half_B, dB, (int)nB, use_bf16) != 0) {
			pthread_mutex_unlock(&g_gemm_mu);
			return -1;
		}
		hB_use = g_half_B;
	}
	rc = cublas_gemm_ex_bf16(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
	    &alpha, hB_use, N, hA_use, K, &beta, dC, N);
	pthread_mutex_unlock(&g_gemm_mu);
	return rc;
}

static int cublas_gemm_rm(ann_real *dC, const ann_real *dA, const ann_real *dB,
	int M, int N, int K)
{
	return cublas_gemm_rm_ha(dC, dA, dB, M, N, K, NULL);
}

static void *bias_cache_get(const void *host, size_t bytes);
static int ew_is_dev(const void *p);
/* The device bias the epilogue reads: g_lt_bias_dev or a bias_cache copy. */
static const float *g_lt_bias_cur;

/* Upload host bias into persistent device scratch (length N). */
static int lt_bias_upload(const ann_real *bias_host, int N)
{
	size_t nb;

	if (!bias_host || N <= 0)
		return -1;
	nb = (size_t)N * sizeof(float);
	/*
	** Graph mode: the parameter's own persistent copy, refreshed once
	** per optimizer step like every other bias (ew_acquire_param). The
	** synchronous upload below cannot be captured, and a graph would
	** read the scratch with whatever bias was uploaded last.
	*/
	if (g_graph_mode && !ew_is_dev(bias_host)) {
		g_lt_bias_cur = (const float *)bias_cache_get(bias_host, nb);
		if (g_lt_bias_cur)
			return 0;
		if (capturing_now()) {
			capture_refuse("epilogue bias outside the bias cache", __LINE__);
			return -1;
		}
	}
	g_lt_bias_cur = g_lt_bias_dev;
	if (g_lt_bias_cap < nb) {
		if (g_lt_bias_dev)
			cudaFree(g_lt_bias_dev);
		g_lt_bias_dev = NULL;
		g_lt_bias_cap = 0;
		if (cudaMalloc(&g_lt_bias_dev, nb) != cudaSuccess)
			return -1;
		g_lt_bias_cap = nb;
	}
	g_lt_bias_cur = g_lt_bias_dev;
	return cudaMemcpy(g_lt_bias_dev, bias_host, nb, cudaMemcpyHostToDevice)
	    == cudaSuccess ? 0 : -1;
}

static const void *lt_bias_cached(const ann_real *host, int N, int bf16);

/*
** Row-major C = epi(A B + bias). epi is BIAS or RELU_BIAS. Bias length N.
*/
/*
** Row-major D = epi(A B + bias) (+ res when res is given: beta = 1 with
** C = res and D = dC out of place, which fuses a residual add).
*/
static int cublas_gemm_rm_epi_res(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int M, int N, int K, const ann_real *bias_host,
	cublasLtEpilogue_t epi, const ann_real *res, int out_bf16)
{
	const ann_real alpha = 1.0f;
	const ann_real beta = res ? 1.0f : 0.0f;
	size_t nA;
	size_t nB;
	int use_bf16;
	void *hA_use;
	void *hB_use;
	const void *bias_dev = NULL;
	int rc;

	if (cublas_ensure() != 0)
		return -1;
	if (!bias_host && epi != CUBLASLT_EPILOGUE_DEFAULT)
		return -1;
	if (bias_host)
		bias_dev = lt_bias_cached(bias_host, N, out_bf16);
	if (bias_host && !bias_dev && lt_bias_upload(bias_host, N) != 0)
		return -1;
	pthread_mutex_lock(&g_gemm_mu);
	if (bias_host && out_bf16 && !bias_dev) {
		/* The epilogue bias in BF16, as the BF16 output requires. */
		if (ensure_half_buf(&g_lt_bias16, &g_lt_bias16_cap, (size_t)N * 2) != 0
		    || half_wt_convert(g_lt_bias16, g_lt_bias_cur, N, 1) != 0) {
			pthread_mutex_unlock(&g_gemm_mu);
			return -1;
		}
	}
	half_scratch_warm();
	use_bf16 = (g_compute_prec == GPU_PREC_BF16);
	if (!use_bf16) {
		pthread_mutex_unlock(&g_gemm_mu);
		return -1;
	}
	nA = (size_t)M * (size_t)K;
	nB = (size_t)K * (size_t)N;
	hA_use = (void *)act_bf16_get(dA, (int)nA);
	if (!hA_use) {
		if (ensure_half_buf(&g_half_A, &g_half_A_cap, nA * 2) != 0) {
			pthread_mutex_unlock(&g_gemm_mu);
			return -1;
		}
		if (half_wt_convert(g_half_A, dA, (int)nA, 1) != 0) {
			pthread_mutex_unlock(&g_gemm_mu);
			return -1;
		}
		hA_use = g_half_A;
	}
	hB_use = half_wt_get(dB, nB, 1);
	if (!hB_use) {
		if (ensure_half_buf(&g_half_B, &g_half_B_cap, nB * 2) != 0) {
			pthread_mutex_unlock(&g_gemm_mu);
			return -1;
		}
		if (half_wt_convert(g_half_B, dB, (int)nB, 1) != 0) {
			pthread_mutex_unlock(&g_gemm_mu);
			return -1;
		}
		hB_use = g_half_B;
	}
	rc = cublas_lt_gemm_half(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
	    hB_use, N, hA_use, K, &beta, res ? (ann_real *)res : dC, N, 1, epi,
	    !bias_host ? NULL : bias_dev ? (const float *)bias_dev
	    : (out_bf16 ? (const float *)g_lt_bias16 : g_lt_bias_cur),
	    res ? dC : NULL, out_bf16);
	pthread_mutex_unlock(&g_gemm_mu);
	return rc;
}

static int cublas_gemm_rm_epi(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int M, int N, int K, const ann_real *bias_host,
	cublasLtEpilogue_t epi)
{
	return cublas_gemm_rm_epi_res(dC, dA, dB, M, N, K, bias_host, epi, NULL, 0);
}

static int cublas_gemm_rm_bias(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int M, int N, int K, const ann_real *bias_host)
{
	return cublas_gemm_rm_epi(dC, dA, dB, M, N, K, bias_host,
	    CUBLASLT_EPILOGUE_BIAS);
}

static int cublas_gemm_rm_bias_relu(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int M, int N, int K, const ann_real *bias_host)
{
	return cublas_gemm_rm_epi(dC, dA, dB, M, N, K, bias_host,
	    CUBLASLT_EPILOGUE_RELU_BIAS);
}

/*
** H2: row-major C = A B^T via column-major C^T = B A^T. A is M x K, B is
** N x K (both row-major), C is M x N. CUBLAS_OP_T reads B transposed in place
** -- no host (or device) transpose copy.
*/
static int cublas_gemm_rm_bt_ex(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int M, int N, int K, int accumulate, void *out16);

static int cublas_gemm_rm_bt(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int M, int N, int K, int accumulate)
{
	return cublas_gemm_rm_bt_ex(dC, dA, dB, M, N, K, accumulate, NULL);
}

/* out16: write the product as BF16 there (Lt, no FP32 C) instead of dC. BF16 mode, beta = 0 only. */
static int cublas_gemm_rm_bt_ex(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int M, int N, int K, int accumulate, void *out16)
{
	const ann_real alpha = 1.0f;
	const ann_real beta = accumulate ? 1.0f : 0.0f;
	size_t nA;
	size_t nB;
	int rc;

	if (cublas_ensure() != 0)
		return -1;
	pthread_mutex_lock(&g_gemm_mu);
	if (g_compute_prec == GPU_PREC_FP32) {
		rc = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
		    &alpha, dB, K, dA, K, &beta, dC, N)
		    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
		pthread_mutex_unlock(&g_gemm_mu);
		return rc;
	}
	nA = (size_t)M * (size_t)K;
	nB = (size_t)N * (size_t)K;
#if CUDART_VERSION >= 12000
	/*
	** FP8 dgrad: E4M3 weight × E5M2 dY with OP_T on weight (per-tensor
	** scale is transpose-safe; sm_89/sm_120 rejects E5M2×E5M2). Delayed
	** scale on dY via its fp8_act_slot.
	*/
	if (g_fp8_dgrad && g_fp8_dgrad_fail < 8 && !accumulate
	    && (K & 15) == 0 && (N & 15) == 0 && (M & 15) == 0) {
		void *wt_fp8 = NULL;
		float *wt_scale = NULL;
		fp8_act_slot *as = fp8_act_scale(dA, 1);

		if (as && fp8_scales_ensure() == 0
		    && ensure_fp8_buf(&g_fp8_A, &g_fp8_A_cap, nA) == 0
		    && fp8_wt_get_ex(dB, N, K, 0, 0, &wt_fp8, &wt_scale) == 0
		    && fp8_quant_tensor(g_fp8_A, FP8_SCALE_CUR(as), dA, (int)nA, 1)
		    == 0
		    && cublas_lt_gemm_fp8_ex(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
		    &alpha, wt_fp8, K, wt_scale, CUDA_R_8F_E4M3, g_fp8_A, K,
		    FP8_SCALE_CUR(as), CUDA_R_8F_E5M2, &beta, dC, N) == 0) {
			fp8_act_scale_commit(FP8_SCALE_CUR(as), 1);
			g_fp8_dgrad_fail = 0;
			/* The BF16 twin the wgrad put is not consumed here: drop it, or a later wgrad could peek a stale one. */
			grad_twin_forget(dA);
			pthread_mutex_unlock(&g_gemm_mu);
			return 0;
		}
		g_fp8_dgrad_fail++;
	}
#endif
	{
		int use_bf16 = (g_compute_prec == GPU_PREC_BF16);
		void *hB_use;

		hB_use = half_wt_get(dB, nB, use_bf16);
		if (!hB_use) {
			if (ensure_half_buf(&g_half_B, &g_half_B_cap, nB * 2)
			    != 0) {
				pthread_mutex_unlock(&g_gemm_mu);
				return -1;
			}
			if (half_wt_convert(g_half_B, dB, (int)nB, use_bf16)
			    != 0) {
				pthread_mutex_unlock(&g_gemm_mu);
				return -1;
			}
			hB_use = g_half_B;
		}
		if (use_bf16) {
			const void *hA_use = NULL;
			int ldA = K;

			/* converted by the wgrad, or written by its producer */
			hA_use = grad_twin_take(dA, nA, K, &ldA);
			if (hA_use)
				g_grad_twin_hit++;
			else {
				g_grad_twin_miss++;
				/* Scratch sized on a miss only: the head's twin needs none. */
				if (ensure_half_buf(&g_half_A, &g_half_A_cap, nA * 2)
				    == 0
				    && half_wt_convert(g_half_A, dA, (int)nA, 1) == 0)
					hA_use = g_half_A;
			}
			if (!hA_use) {
				pthread_mutex_unlock(&g_gemm_mu);
				return -1;
			}
			if (out16 && !accumulate)
				rc = cublas_lt_gemm_half(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
				    &alpha, hB_use, K, hA_use, ldA, &beta, (float *)out16, N, 1,
				    CUBLASLT_EPILOGUE_DEFAULT, NULL, NULL, 1);
			else
				rc = cublas_gemm_ex_bf16(CUBLAS_OP_T, CUBLAS_OP_N, N,
				    M, K, &alpha, hB_use, K, hA_use, ldA, &beta, dC, N);
			if (rc == 0 && hA_use == g_grad_twin[GRAD_TWIN_INPLACE].half)
				g_ce_inplace_used |= 2;
			pthread_mutex_unlock(&g_gemm_mu);
			return rc;
		}
		pthread_mutex_unlock(&g_gemm_mu);
		return -1;
	}
}

/*
** H2: row-major C (+)= A^T B via column-major C^T = B^T A. A is T x M, B is
** T x N, C is M x N. accumulate != 0 keeps C (beta = 1: dW += A^T B).
*/
static int cublas_gemm_rm_at(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int T, int M, int N, int accumulate)
{
	const ann_real alpha = 1.0f;
	const ann_real beta = accumulate ? 1.0f : 0.0f;
	size_t nA;
	size_t nB;
	int rc;

	if (cublas_ensure() != 0)
		return -1;
	pthread_mutex_lock(&g_gemm_mu);
	if (g_compute_prec == GPU_PREC_FP32) {
		rc = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, N, M, T,
		    &alpha, dB, N, dA, M, &beta, dC, N)
		    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
		pthread_mutex_unlock(&g_gemm_mu);
		return rc;
	}
	nA = (size_t)T * (size_t)M;
	nB = (size_t)T * (size_t)N;
	if (g_compute_prec == GPU_PREC_BF16) {
		const void *hA_use = act_bf16_get(dA, (int)nA);
		const void *hB_use = act_bf16_get(dB, (int)nB);
		int ldB = N;

		/*
		** The scratch buffers are sized only when a conversion is
		** needed: the head's delta comes as a twin and would otherwise
		** grow g_half_B to 134 MB for nothing.
		*/
		if (!hA_use) {
			if (ensure_half_buf(&g_half_A, &g_half_A_cap, nA * 2) != 0
			    || half_wt_convert(g_half_A, dA, (int)nA, 1) != 0) {
				pthread_mutex_unlock(&g_gemm_mu);
				return -1;
			}
			hA_use = g_half_A;
		}
		if (!hB_use)	/* masked by the fused ReLU bwd, the head's CE kernel... */
			hB_use = grad_twin_peek(dB, nB, N, &ldB);
		if (!hB_use) {
			/* Convert G once, into the one-shot twin the dgrad reuses. */
			hB_use = grad_twin_put(dB, nB);
			if (!hB_use) {
				if (ensure_half_buf(&g_half_B, &g_half_B_cap,
				    nB * 2) != 0
				    || half_wt_convert(g_half_B, dB, (int)nB, 1)
				    != 0) {
					pthread_mutex_unlock(&g_gemm_mu);
					return -1;
				}
				hB_use = g_half_B;
			}
		}
		rc = cublas_gemm_ex_bf16(CUBLAS_OP_N, CUBLAS_OP_T, N, M, T,
		    &alpha, hB_use, ldB, hA_use, M, &beta, dC, N);
		if (rc == 0 && hB_use == g_grad_twin[GRAD_TWIN_INPLACE].half)
			g_ce_inplace_used |= 1;
		pthread_mutex_unlock(&g_gemm_mu);
		return rc;
	}
	pthread_mutex_unlock(&g_gemm_mu);
	return -1;
}
#endif

__global__ static void gemm_kernel(const ann_real *A, const ann_real *B, ann_real *C,
	int M, int N, int K)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	ann_real sum;

	if (row >= M || col >= N)
		return;
	sum = 0.0f;
	for (k = 0; k < K; k++)
		sum += A[row * K + k] * B[k * N + col];
	C[row * N + col] = sum;
}

static pthread_once_t g_st_once = PTHREAD_ONCE_INIT;

static void compute_stream_create(void)
{
	if (cudaStreamCreate(&g_st) != cudaSuccess)
		g_st = 0;	/* legacy stream: everything works but capture */
}

extern "C" int gpu_available(void)
{
	int n;

	if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0)
		return 0;
	pthread_once(&g_st_once, compute_stream_create);
	return 1;
}

extern "C" int gpu_mem_info(size_t *free_bytes, size_t *total_bytes)
{
	size_t f;
	size_t t;

	if (free_bytes)
		*free_bytes = 0;
	if (total_bytes)
		*total_bytes = 0;
	if (cudaMemGetInfo(&f, &t) != cudaSuccess)
		return -1;
	if (free_bytes)
		*free_bytes = f;
	if (total_bytes)
		*total_bytes = t;
	return 0;
}

extern "C" const char *gpu_backend_name(void)
{
#ifdef ANN_CUBLAS
	return "cuda+cublas";
#else
	return "cuda";
#endif
}

/*
** Graph mode: exact-size block recycling behind gpu_malloc / gpu_free.
** Every mat temporary of a microbatch is a gpu_malloc / gpu_free pair; a
** steady microbatch asks for the same sizes in the same order, so after one
** uncaptured microbatch a capture finds every block it needs in the pool
** (no cudaMalloc inside the capture) and gets the same addresses back.
**
** Blocks handed out or returned inside a capture are the graph's own
** temporaries (pb->graph). The next uncaptured microbatch may borrow them,
** but a replay while one is out would overwrite a live buffer: they are
** counted in g_pool_graph_out and gpu_graph_replayable waits for zero.
** Returning any other block bumps g_dev_free_gen exactly as cudaFree does:
** the graph may still read it through its owner.
*/
typedef struct {
	void	*p;
	size_t	n;
	int	free;
	int	graph;	/* a temporary of the current graph */
	int	out;	/* a graph temporary lent outside the capture */
} pool_blk;
typedef struct {
	size_t	n;
	int	*stk;	/* g_pool indices of free blocks, LIFO */
	int	cnt;
	int	cap;
} pool_bin;
#define POOL_HASH	(1 << 17)	/* g_pool index + 1 by address, 0 = empty */
/* gpu_malloc / gpu_free also run on OpenMP threads (per-head attention). */
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pool_blk *g_pool;
static int g_pool_n;
static int g_pool_cap;
static int *g_pool_hash;
static pool_bin *g_pool_bin;
static int g_pool_nbin;
static long g_pool_graph_out;

static unsigned pool_slot(const void *p)
{
	unsigned long long k = (unsigned long long)(size_t)p;

	k ^= k >> 29;
	k *= 0x9e3779b97f4a7c15ULL;
	return (unsigned)(k >> 40) & (POOL_HASH - 1);
}

static int pool_find(const void *p)
{
	unsigned h;
	int i;

	if (!g_pool_hash)
		return -1;
	for (h = pool_slot(p); (i = g_pool_hash[h]) != 0; h = (h + 1) & (POOL_HASH - 1))
		if (g_pool[i - 1].p == p)
			return i - 1;
	return -1;
}

static pool_bin *pool_bin_for(size_t n, int create)
{
	int i;

	for (i = 0; i < g_pool_nbin; i++)
		if (g_pool_bin[i].n == n)
			return &g_pool_bin[i];
	if (!create)
		return NULL;
	{
		pool_bin *nb = (pool_bin *)realloc(g_pool_bin,
		    (size_t)(g_pool_nbin + 1) * sizeof(pool_bin));

		if (!nb)
			return NULL;
		g_pool_bin = nb;
	}
	memset(&g_pool_bin[g_pool_nbin], 0, sizeof(pool_bin));
	g_pool_bin[g_pool_nbin].n = n;
	return &g_pool_bin[g_pool_nbin++];
}

static int pool_push(int i)
{
	pool_bin *b = pool_bin_for(g_pool[i].n, 1);

	if (!b)
		return -1;
	if (b->cnt == b->cap) {
		int nc = b->cap ? b->cap * 2 : 16;
		int *ns = (int *)realloc(b->stk, (size_t)nc * sizeof(int));

		if (!ns)
			return -1;
		b->stk = ns;
		b->cap = nc;
	}
	b->stk[b->cnt++] = i;
	g_pool[i].free = 1;
	return 0;
}

/* Track a fresh cudaMalloc block; -1 when the tables cannot grow. */
static int pool_add(void *p, size_t n)
{
	unsigned h;

	if (!g_pool_hash) {
		g_pool_hash = (int *)calloc(POOL_HASH, sizeof(int));
		if (!g_pool_hash)
			return -1;
	}
	/* Half full at most: the table is never rehashed. */
	if (g_pool_n >= POOL_HASH / 2)
		return -1;
	if (g_pool_n == g_pool_cap) {
		int nc = g_pool_cap ? g_pool_cap * 2 : 1024;
		pool_blk *np = (pool_blk *)realloc(g_pool,
		    (size_t)nc * sizeof(pool_blk));

		if (!np)
			return -1;
		g_pool = np;
		g_pool_cap = nc;
	}
	g_pool[g_pool_n].p = p;
	g_pool[g_pool_n].n = n;
	g_pool[g_pool_n].free = 0;
	g_pool[g_pool_n].graph = 0;
	g_pool[g_pool_n].out = 0;
	for (h = pool_slot(p); g_pool_hash[h] != 0; h = (h + 1) & (POOL_HASH - 1))
		;
	g_pool_hash[h] = ++g_pool_n;
	return 0;
}

/*
** Give every free block back to the driver and forget all of them (blocks
** still in use become ordinary allocations again). A graph may hold any of
** these addresses: g_dev_free_gen moves on.
*/
static void pool_release_all(void)
{
	int i;

	for (i = 0; i < g_pool_n; i++)
		if (g_pool[i].free)
			(void)cudaFree(g_pool[i].p);
	for (i = 0; i < g_pool_nbin; i++)
		free(g_pool_bin[i].stk);
	free(g_pool_bin);
	g_pool_bin = NULL;
	g_pool_nbin = 0;
	free(g_pool);
	g_pool = NULL;
	g_pool_n = 0;
	g_pool_cap = 0;
	free(g_pool_hash);
	g_pool_hash = NULL;
	g_pool_graph_out = 0;
	g_dev_free_gen++;
}

/*
** At gpu_device_sync (the optimizer step): give back the idle blocks no
** graph uses (an evaluation's leftovers, or every temporary before the
** first capture), so the pool never holds more than a microbatch needs,
** and compact the tables. A graph cannot hold a block that is free and not
** its own: freeing it from its owner already moved g_dev_free_gen on.
*/
static void pool_trim_idle(void)
{
	int *remap;
	int i;
	int n;
	unsigned h;

	n = 0;
	for (i = 0; i < g_pool_n; i++)
		n += !g_pool[i].p || (g_pool[i].free && !g_pool[i].graph);
	if (n == 0)
		return;
	remap = (int *)malloc((size_t)(g_pool_n ? g_pool_n : 1) * sizeof(int));
	if (!remap)
		return;
	n = 0;
	for (i = 0; i < g_pool_n; i++) {
		if (!g_pool[i].p || (g_pool[i].free && !g_pool[i].graph)) {
			/* (cudaFree): the real call, no new generation. */
			if (g_pool[i].p)
				(void)(cudaFree)(g_pool[i].p);
			remap[i] = -1;
			continue;
		}
		remap[i] = n;
		g_pool[n++] = g_pool[i];
	}
	g_pool_n = n;
	/* The graph's blocks keep their LIFO order in every bin. */
	for (i = 0; i < g_pool_nbin; i++) {
		pool_bin *b = &g_pool_bin[i];
		int j;
		int k = 0;

		for (j = 0; j < b->cnt; j++)
			if (remap[b->stk[j]] >= 0)
				b->stk[k++] = remap[b->stk[j]];
		b->cnt = k;
	}
	free(remap);
	memset(g_pool_hash, 0, POOL_HASH * sizeof(int));
	for (i = 0; i < g_pool_n; i++) {
		for (h = pool_slot(g_pool[i].p); g_pool_hash[h] != 0;
		    h = (h + 1) & (POOL_HASH - 1))
			;
		g_pool_hash[h] = i + 1;
	}
}

extern "C" void *gpu_malloc(size_t nbytes)
{
	void *p;

	if (nbytes == 0 || !gpu_available())
		return NULL;
	if (g_graph_mode) {
		pool_bin *b;

		pthread_mutex_lock(&g_pool_mu);
		b = pool_bin_for(nbytes, 0);
		if (b && b->cnt > 0) {
			int i = b->stk[--b->cnt];

			g_pool[i].free = 0;
			if (capturing_now())
				g_pool[i].graph = 1;
			else if (g_pool[i].graph) {
				g_pool[i].out = 1;
				g_pool_graph_out++;
			}
			p = g_pool[i].p;
			pthread_mutex_unlock(&g_pool_mu);
			return p;
		}
		if (capturing_now()) {
			capture_refuse("gpu_malloc of a size the pool lacks", __LINE__);
			pthread_mutex_unlock(&g_pool_mu);
			return NULL;
		}
		if (cudaMalloc(&p, nbytes) != cudaSuccess) {
			/* Out of memory: the pool's idle blocks go first. */
			(void)cudaGetLastError();
			pool_release_all();
			if (cudaMalloc(&p, nbytes) != cudaSuccess) {
				pthread_mutex_unlock(&g_pool_mu);
				return NULL;
			}
		}
		if (pool_add(p, nbytes) != 0) {
			static int warned;

			if (!warned++)
				fprintf(stderr, "[graph] block pool full: plain allocations from now on\n");
		}
		pthread_mutex_unlock(&g_pool_mu);
		return p;
	}
	if (cudaMalloc(&p, nbytes) != cudaSuccess)
		return NULL;
	return p;
}

/*
** Twin caches are keyed on the FP32 device address. cudaMalloc reuses
** addresses, so a twin of a freed buffer must not survive the free: a new
** buffer at the same address (same size) would be served the old data.
*/
static void twins_forget_dev(const void *p)
{
	int i;

	for (i = 0; i < HALF_WT_SLOTS; i++) {
		if (g_half_wt[i].fp32 == (const ann_real *)p) {
			if (g_half_wt[i].half)
				cudaFree(g_half_wt[i].half);
			g_half_wt[i].half = NULL;
			g_half_wt[i].fp32 = NULL;
			g_half_wt[i].n = 0;
			g_half_wt[i].valid = 0;
		}
	}
	for (i = 0; i < ACT_BF16_SLOTS; i++) {
		if (g_act_bf16[i].fp32 == (const ann_real *)p) {
			g_act_bf16[i].fp32 = NULL;
			g_act_bf16[i].live = 0;
		}
	}
	grad_twin_forget(p);
	qkv_bf16_forget(p);
#if CUDART_VERSION >= 12000 && defined(ANN_CUBLAS)
	fp8_wt_invalidate((const ann_real *)p);
	fp8_act_forget(p);
#endif
}

/* Frees refused inside a capture: done right after it ends. */
#define DEFER_FREE_MAX	64
static void *g_defer_free[DEFER_FREE_MAX];
static int g_defer_free_n;

extern "C" void gpu_free(void *p)
{
	int i;

	if (!p)
		return;
	twins_forget_dev(p);
	if (g_graph_mode) {
		pthread_mutex_lock(&g_pool_mu);
		if ((i = pool_find(p)) >= 0 && !g_pool[i].free) {
			if (g_pool[i].out) {
				g_pool[i].out = 0;
				g_pool_graph_out--;
			}
			if (capturing_now())
				g_pool[i].graph = 1;
			else if (!g_pool[i].graph)
				g_dev_free_gen++;
			if (pool_push(i) == 0) {
				pthread_mutex_unlock(&g_pool_mu);
				return;
			}
			/* No room to keep it: an ordinary free, the entry is dead. */
			g_pool[i].p = NULL;
			g_pool[i].n = 0;
		}
		pthread_mutex_unlock(&g_pool_mu);
	}
	if (capturing_now()) {
		capture_refuse("gpu_free of an unpooled buffer", __LINE__);
		if (g_defer_free_n < DEFER_FREE_MAX) {
			g_defer_free[g_defer_free_n++] = p;
			return;
		}
	}
	cudaFree(p);
}

extern "C" int gpu_memcpy_h2d(void *dst, const void *src, size_t nbytes)
{
	if (!dst || !src || nbytes == 0)
		return -1;
	/* Async on default stream; same-stream kernels/copies order themselves. */
	return cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, g_st)
	    == cudaSuccess ? 0 : -1;
}

extern "C" int gpu_memcpy_d2h(void *dst, const void *src, size_t nbytes)
{
	if (!dst || !src || nbytes == 0)
		return -1;
	/* Host callers read dst immediately; keep D2H synchronous. */
	return cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost) == cudaSuccess
	    ? 0 : -1;
}

#define HOST_PIN_SLOTS 256
static void *g_host_pin[HOST_PIN_SLOTS];
static int g_host_pin_n;

extern "C" int gpu_host_pin(void *p, size_t nbytes)
{
	int i;

	if (!p || nbytes == 0)
		return -1;
	for (i = 0; i < g_host_pin_n; i++)
		if (g_host_pin[i] == p)
			return 0;
	if (g_host_pin_n >= HOST_PIN_SLOTS)
		return -1;
	if (cudaHostRegister(p, nbytes, cudaHostRegisterDefault) != cudaSuccess)
		return -1;
	g_host_pin[g_host_pin_n++] = p;
	return 0;
}

extern "C" int gpu_memcpy_d2h_async(void *dst, const void *src, size_t nbytes)
{
	if (!dst || !src || nbytes == 0)
		return -1;
	return cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, g_st)
	    == cudaSuccess ? 0 : -1;
}

extern "C" int gpu_stream_sync(void)
{
	return cudaStreamSynchronize(g_st) == cudaSuccess ? 0 : -1;
}

extern "C" int gpu_memcpy_d2d(void *dst, const void *src, size_t nbytes)
{
	if (!dst || !src || nbytes == 0)
		return -1;
	return cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, g_st)
	    == cudaSuccess ? 0 : -1;
}

extern "C" int gpu_memset(void *dst, int value, size_t nbytes)
{
	if (!dst || nbytes == 0)
		return -1;
	return cudaMemsetAsync(dst, value, nbytes, g_st) == cudaSuccess ? 0 : -1;
}

static size_t mat_bytes(const mat *m)
{
	return (size_t)m->rows * (size_t)m->cols * sizeof(ann_real);
}

/*
** Device pointer for reading m. Pinned mats reuse m->d_data (never cudaMalloc);
** if the host is authoritative (on_device == 0) the twin is refreshed by H2D
** first. Unpinned mats get a temporary buffer (*owned = 1).
*/
static int ensure_dev(const mat *m, ann_real **out, int *owned)
{
	*owned = 0;
	*out = NULL;
	if (m->d_data) {
		if (m->on_device != MAT_DEV_DEVICE
		    && m->on_device != MAT_DEV_SYNCED
		    && m->on_device != MAT_DEV_ALIASED) {
			if (!m->data
			    || gpu_memcpy_h2d(m->d_data, m->data, mat_bytes(m)) != 0)
				return -1;
			half_wt_invalidate(m->d_data);
		}
		*out = m->d_data;
		return 0;
	}
	if (!m->data)
		return -1;
	*out = (ann_real *)gpu_malloc(mat_bytes(m));
	if (!*out)
		return -1;
	if (gpu_memcpy_h2d(*out, m->data, mat_bytes(m)) != 0) {
		gpu_free(*out);
		*out = NULL;
		return -1;
	}
	*owned = 1;
	return 0;
}

/*
** ensure_dev for a K or V cache mat that may be lazy (ANN_VRAM_LAZY): the
** fused FA2 QKV GEMM leaves K and V only inside the layer's BF16 qkv slot, so
** the FP32 mats own no storage at all (no device buffer, no host copy) and
** *out stays NULL. A caller that really needs the FP32 values checks *out.
*/
static int vram_lazy_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_VRAM_LAZY");

		on = !(e && e[0] == '0');
	}
	return on;
}

static int ensure_dev_kv(const mat *m, ann_real **out, int *owned)
{
	if (!m->d_data && !m->data) {
		*owned = 0;
		*out = NULL;
		return 0;
	}
	return ensure_dev(m, out, owned);
}

/*
** Output buffer for a pure-write result. Always attach a persistent twin so
** BATCH/act-checkpoint can drop and re-alloc without falling back to a host
** mirror farm (owned temps used to force D2H into m->data).
*/
static int ensure_dev_out(mat *m, ann_real **out, int *owned)
{
	*owned = 0;
	*out = NULL;
	if (mat_device_alloc(m) != 0)
		return -1;
	*out = m->d_data;
	return *out ? 0 : -1;
}

/*
** After a launch: always surface launch errors; only block on the device in
** eager mode. BATCH defers to gpu_device_sync() once per microbatch.
*/
static int stream_is_capturing(void)
{
	return capturing_now();
}

static int launch_done(void)
{
	if (cudaGetLastError() != cudaSuccess)
		return -1;
	if (g_sync_mode == GPU_SYNC_BATCH || stream_is_capturing())
		return 0;
	return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}

/* Deferred bias col-sums: accumulate on device, flush to host at sync. */
#define COLSUM_ACC_SLOTS	96
typedef struct {
	ann_real	*host;
	ann_real	*dev;
	int		cols;
	int		dirty;
} colsum_acc;
static colsum_acc g_colsum_acc[COLSUM_ACC_SLOTS];

static void colsum_flush_all(void)
{
	int i;
	int j;
	ann_real *tmp;
	size_t nb;

	for (i = 0; i < COLSUM_ACC_SLOTS; i++) {
		colsum_acc *a = &g_colsum_acc[i];

		if (!a->dirty || !a->host || !a->dev || a->cols <= 0)
			continue;
		nb = (size_t)a->cols * sizeof(ann_real);
		tmp = (ann_real *)malloc(nb);
		if (!tmp)
			continue;
		if (cudaMemcpy(tmp, a->dev, nb, cudaMemcpyDeviceToHost)
		    == cudaSuccess) {
			for (j = 0; j < a->cols; j++)
				a->host[j] += tmp[j];
		}
		free(tmp);
		(void)cudaMemsetAsync(a->dev, 0, nb, g_st);
		a->dirty = 0;
	}
}

static colsum_acc *colsum_acc_get(ann_real *db_host, int cols)
{
	int i;
	int free_slot = -1;
	size_t nb;

	for (i = 0; i < COLSUM_ACC_SLOTS; i++) {
		if (g_colsum_acc[i].host == db_host
		    && g_colsum_acc[i].cols == cols)
			return &g_colsum_acc[i];
		if (free_slot < 0 && !g_colsum_acc[i].host)
			free_slot = i;
	}
	if (free_slot < 0)
		return NULL;
	nb = (size_t)cols * sizeof(ann_real);
	if (cudaMalloc((void **)&g_colsum_acc[free_slot].dev, nb)
	    != cudaSuccess) {
		g_colsum_acc[free_slot].dev = NULL;
		return NULL;
	}
	(void)cudaMemsetAsync(g_colsum_acc[free_slot].dev, 0, nb, g_st);
	g_colsum_acc[free_slot].host = db_host;
	g_colsum_acc[free_slot].cols = cols;
	g_colsum_acc[free_slot].dirty = 0;
	return &g_colsum_acc[free_slot];
}

/* Debug: name of the pending (and cleared) CUDA error, "" when none. */
extern "C" const char *gpu_debug_last_error(void)
{
	cudaError_t e = cudaGetLastError();

	return e == cudaSuccess ? "" : cudaGetErrorString(e);
}

/*
** ANN_OP_PROF=1: per-operation GPU time from cudaEvent pairs recorded on
** the default stream around every gpu_* entry (nsys records no CUDA
** trace on this WSL). Pairs are resolved at gpu_device_sync and the
** table is printed by gpu_op_prof_dump (the optimizer step).
*/
enum {
	OP_GEMM_FWD, OP_GEMM_DGRAD, OP_GEMM_WGRAD, OP_QKV, OP_ATTN_FWD,
	OP_ATTN_BWD, OP_LN_FWD, OP_LN_BWD, OP_SOFTMAX, OP_CE, OP_COLSUM,
	OP_ELEMWISE, OP_RESID, OP_EMBED, OP_POSENC, OP_PACK, OP_ADAM,
	OP_CONVERT, OP_ATT_PACK, OP_ATT_GEMM, OP_ATT_SMX, OP_N
};
static const char *g_op_name[OP_N] = {
	"gemm_fwd", "gemm_dgrad", "gemm_wgrad", "qkv_fused", "attn_fwd",
	"attn_bwd", "ln_fwd", "ln_bwd", "softmax", "ce", "colsum", "elemwise",
	"resid", "embed", "posenc", "pack", "adam", "convert", "att_pack",
	"att_gemm", "att_smx"
};
#define OP_EV_SLOTS	16384
typedef struct {
	int		op;
	cudaEvent_t	e0;
	cudaEvent_t	e1;
} op_ev;
static op_ev g_op_ev[OP_EV_SLOTS];
static int g_op_ev_n;
static int g_op_ev_made;
static double g_op_ms[OP_N];
static long g_op_calls[OP_N];
static int g_op_depth;
/* OpenMP threads call gpu_* concurrently (per-head attention): one slot table. */
static pthread_mutex_t g_op_mu = PTHREAD_MUTEX_INITIALIZER;

static int op_prof_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_OP_PROF");

		on = (e && e[0] == '1') ? 1 : 0;
	}
	return on;
}

static int op_begin(int op)
{
	op_ev *e;
	int i;

	if (graph_debug_on() && g_st) {
		static int said;
		cudaStreamCaptureStatus cs;

		/* Name the first op that starts after the capture broke. */
		if (cudaStreamIsCapturing(g_st, &cs) == cudaSuccess) {
			if (cs == cudaStreamCaptureStatusNone)
				said = 0;
			else if (cs == cudaStreamCaptureStatusInvalidated && !said++)
				fprintf(stderr, "[graph] capture already invalid when %s starts\n",
				    g_op_name[op]);
		}
	}
	if (!op_prof_on() || stream_is_capturing())
		return -1;
	pthread_mutex_lock(&g_op_mu);
	if (g_op_depth > 0 || g_op_ev_n >= OP_EV_SLOTS) {
		pthread_mutex_unlock(&g_op_mu);
		return -1;
	}
	e = &g_op_ev[g_op_ev_n];
	if (g_op_ev_n >= g_op_ev_made) {
		if (cudaEventCreate(&e->e0) != cudaSuccess
		    || cudaEventCreate(&e->e1) != cudaSuccess) {
			(void)cudaGetLastError();
			pthread_mutex_unlock(&g_op_mu);
			return -1;
		}
		g_op_ev_made++;
	}
	e->op = op;
	if (cudaEventRecord(e->e0, g_st) != cudaSuccess) {
		static int warned;

		if (!warned++)
			fprintf(stderr, "[opprof] event record failed: %s\n",
			    cudaGetErrorString(cudaGetLastError()));
		(void)cudaGetLastError();	/* never poison launch checks */
		pthread_mutex_unlock(&g_op_mu);
		return -1;
	}
	g_op_depth++;
	i = g_op_ev_n++;
	pthread_mutex_unlock(&g_op_mu);
	return i;
}

/* Operand conversions run inside a GEMM scope: record them without the depth guard. */
static int op_begin_nested(int op)
{
	op_ev *e;
	int i;

	if (!op_prof_on() || stream_is_capturing())
		return -1;
	pthread_mutex_lock(&g_op_mu);
	if (g_op_ev_n >= OP_EV_SLOTS) {
		pthread_mutex_unlock(&g_op_mu);
		return -1;
	}
	e = &g_op_ev[g_op_ev_n];
	if (g_op_ev_n >= g_op_ev_made) {
		if (cudaEventCreate(&e->e0) != cudaSuccess
		    || cudaEventCreate(&e->e1) != cudaSuccess) {
			(void)cudaGetLastError();
			pthread_mutex_unlock(&g_op_mu);
			return -1;
		}
		g_op_ev_made++;
	}
	e->op = op;
	if (cudaEventRecord(e->e0, g_st) != cudaSuccess) {
		(void)cudaGetLastError();
		pthread_mutex_unlock(&g_op_mu);
		return -1;
	}
	g_op_depth++;	/* op_end decrements */
	i = g_op_ev_n++;
	pthread_mutex_unlock(&g_op_mu);
	return i;
}

static void op_end(int i)
{
	if (i < 0)
		return;
	pthread_mutex_lock(&g_op_mu);
	if (cudaEventRecord(g_op_ev[i].e1, g_st) != cudaSuccess)
		(void)cudaGetLastError();
	g_op_depth--;
	pthread_mutex_unlock(&g_op_mu);
}

struct OpScope {
	int	i;
	OpScope(int op) { i = op_begin(op); }
	~OpScope() { op_end(i); }
};

/* After a device sync: fold the recorded pairs into the per-op totals. */
static void op_prof_resolve(void)
{
	int i;
	float ms;

	pthread_mutex_lock(&g_op_mu);
	for (i = 0; i < g_op_ev_n; i++) {
		cudaError_t err = cudaEventElapsedTime(&ms, g_op_ev[i].e0,
		    g_op_ev[i].e1);

		if (err == cudaSuccess) {
			g_op_ms[g_op_ev[i].op] += ms;
			g_op_calls[g_op_ev[i].op]++;
		} else {
			static int warned;

			if (!warned++)
				fprintf(stderr, "[opprof] elapsed failed on %s: %s\n",
				    g_op_name[g_op_ev[i].op], cudaGetErrorString(err));
			(void)cudaGetLastError();
		}
	}
	g_op_ev_n = 0;
	pthread_mutex_unlock(&g_op_mu);
}

extern "C" void gpu_op_prof_dump(void)
{
	double tot = 0.0;
	long mb;
	int i;

	if (!op_prof_on())
		return;
	if (cudaDeviceSynchronize() == cudaSuccess)
		op_prof_resolve();
	for (i = 0; i < OP_N; i++)
		tot += g_op_ms[i];
	mb = g_op_calls[OP_CE] > 0 ? g_op_calls[OP_CE] : 1;
	/* convert is also inside the gemm_* rows: the total excludes it. */
	tot -= g_op_ms[OP_CONVERT];
	fprintf(stderr, "[opprof] %ld microbatches, %.1f ms/microbatch of GPU op time (convert counted inside gemm rows); grad twin hits %ld misses %ld\n",
	    mb, tot / (double)mb, g_grad_twin_hit, g_grad_twin_miss);
	g_grad_twin_hit = g_grad_twin_miss = 0;
	for (i = 0; i < OP_N; i++) {
		if (g_op_calls[i] == 0)
			continue;
		fprintf(stderr, "[opprof]   %-11s %6.1f ms/mb %5.1f%% %6ld calls/mb\n",
		    g_op_name[i], g_op_ms[i] / (double)mb,
		    tot > 0.0 ? 100.0 * g_op_ms[i] / tot : 0.0,
		    g_op_calls[i] / mb);
		g_op_ms[i] = 0.0;
		g_op_calls[i] = 0;
	}
}

extern "C" void gpu_device_sync(void)
{
	cudaError_t err;

	if (!gpu_available() || stream_is_capturing())
		return;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		fprintf(stderr, "gpu_device_sync: %s\n", cudaGetErrorString(err));
	else {
		colsum_flush_all();
		if (op_prof_on())
			op_prof_resolve();
		if (g_graph_mode) {
			pthread_mutex_lock(&g_pool_mu);
			pool_trim_idle();
			pthread_mutex_unlock(&g_pool_mu);
		}
	}
}

/*
** Finish an output mat: BATCH + pinned leaves the result on device
** (on_device = 1, no PCIe D2H); otherwise D2H into host and mark host-auth.
*/
static int finish_out(mat *m, ann_real *dptr, int owned)
{
	/*
	** A dW aliased into the Adam flat stays MAT_DEV_ALIASED: turning it
	** into a plain DEVICE mat made mat_free cudaFree a slice of the flat
	** (sticky "invalid argument" that fails the next launch check).
	*/
	if (g_sync_mode == GPU_SYNC_BATCH && m->d_data && !owned) {
		if (m->on_device != MAT_DEV_ALIASED)
			m->on_device = MAT_DEV_DEVICE;
		return 0;
	}
	/*
	** Eager: mirror the result to the host and leave the host
	** authoritative (host code may edit it in place without marking). A
	** device-only mat (no host pages) stays device-authoritative: marking
	** it HOST handed every consumer a NULL host pointer and the CPU
	** fallbacks read zeros, which made eager-mode forwards (resume start
	** NLL, lm info, generation) evaluate a loaded model near random.
	*/
	if (m->data && gpu_memcpy_d2h(m->data, dptr, mat_bytes(m)) != 0)
		return -1;
	if (m->d_data && !owned && m->on_device != MAT_DEV_ALIASED)
		m->on_device = m->data ? MAT_DEV_HOST : MAT_DEV_DEVICE;
	return 0;
}

extern "C" int gpu_mat_mul(mat *c, const mat *a, const mat *b)
{
	OpScope _op(OP_GEMM_FWD);
	ann_real *dA;
	ann_real *dB;
	ann_real *dC;
	int ownA;
	int ownB;
	int ownC;
	dim3 block;
	dim3 grid;
	int rc;

	if (!c || !a || !b)
		return -1;
	if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols)
		return -1;
	if (!gpu_available())
		return -1;
	if (ensure_dev(a, &dA, &ownA) != 0)
		return -1;
	if (ensure_dev(b, &dB, &ownB) != 0) {
		if (ownA)
			gpu_free(dA);
		return -1;
	}
	if (ensure_dev_out(c, &dC, &ownC) != 0) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		return -1;
	}
#ifdef ANN_CUBLAS
	if (cublas_gemm_rm(dC, dA, dB, a->rows, b->cols, a->cols) != 0) {
#endif
		block = dim3(16, 16);
		grid = dim3((c->cols + 15) / 16, (c->rows + 15) / 16);
		gemm_kernel<<<grid, block, 0, g_st>>>(dA, dB, dC, a->rows, b->cols,
		    a->cols);
#ifdef ANN_CUBLAS
	}
#endif
	rc = -1;
	if (launch_done() == 0 && finish_out(c, dC, ownC) == 0)
		rc = 0;
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownC)
		gpu_free(dC);
	return rc;
}

__global__ static void gemm_bt_kernel(const ann_real *A, const ann_real *B, ann_real *C,
	int M, int N, int K)
{
	/* C[m,n] = sum_k A[m,k] * B[n,k] */
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	ann_real sum;

	if (row >= M || col >= N)
		return;
	sum = 0.0f;
	for (k = 0; k < K; k++)
		sum += A[row * K + k] * B[col * K + k];
	C[row * N + col] = sum;
}

__global__ static void outer_add_kernel(ann_real *DW, const ann_real *A, const ann_real *B,
	int T, int M, int N)
{
	/* DW[m,n] += sum_t A[t,m] * B[t,n] */
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int t;
	ann_real sum;

	if (m >= M || n >= N)
		return;
	sum = 0.0f;
	for (t = 0; t < T; t++)
		sum += A[t * M + m] * B[t * N + n];
	DW[m * N + n] += sum;
}

extern "C" int gpu_mat_mul_bt(mat *c, const mat *a, const mat *b)
{
	OpScope _op(OP_GEMM_DGRAD);
	ann_real *dA;
	ann_real *dB;
	ann_real *dC;
	int ownA;
	int ownB;
	int ownC;
	dim3 block;
	dim3 grid;
	int rc;

	if (!c || !a || !b || !gpu_available())
		return -1;
	if (a->cols != b->cols || c->rows != a->rows || c->cols != b->rows)
		return -1;
	if (ensure_dev(a, &dA, &ownA) != 0)
		return -1;
	if (ensure_dev(b, &dB, &ownB) != 0) {
		if (ownA)
			gpu_free(dA);
		return -1;
	}
	if (ensure_dev_out(c, &dC, &ownC) != 0) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		return -1;
	}
#ifdef ANN_CUBLAS
	if (cublas_gemm_rm_bt(dC, dA, dB, a->rows, b->rows, a->cols, 0) != 0) {
#endif
		block = dim3(16, 16);
		grid = dim3((c->cols + 15) / 16, (c->rows + 15) / 16);
		gemm_bt_kernel<<<grid, block, 0, g_st>>>(dA, dB, dC, a->rows, b->rows,
		    a->cols);
#ifdef ANN_CUBLAS
	}
#endif
	rc = -1;
	if (launch_done() == 0 && finish_out(c, dC, ownC) == 0)
		rc = 0;
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownC)
		gpu_free(dC);
	return rc;
}

__global__ static void gemm_at_kernel(const ann_real *A, const ann_real *B, ann_real *C,
	int T, int M, int N)
{
	/* C[m,n] = sum_t A[t,m] * B[t,n] */
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int t;
	ann_real sum;

	if (m >= M || n >= N)
		return;
	sum = 0.0f;
	for (t = 0; t < T; t++)
		sum += A[t * M + m] * B[t * N + n];
	C[m * N + n] = sum;
}

/* c = a^T b ; a (T x m), b (T x n) -> c (m x n). cuBLAS OP_T when built in. */
/*
** c += a b^T (BF16 / FP32 cuBLAS with beta = 1): the dx accumulation of
** the attention backward without a scratch product and an axpy pass. c is
** uploaded when host-authoritative. -1 = no cuBLAS or GEMM failure.
*/
extern "C" int gpu_mat_mul_bt_acc(mat *c, const mat *a, const mat *b)
{
#ifdef ANN_CUBLAS
	OpScope _op(OP_GEMM_DGRAD);
	ann_real *dA;
	ann_real *dB;
	ann_real *dC;
	int ownA;
	int ownB;
	int ownC;
	int rc;

	if (!c || !a || !b || !gpu_available())
		return -1;
	if (a->cols != b->cols || c->rows != a->rows || c->cols != b->rows)
		return -1;
	if (ensure_dev(a, &dA, &ownA) != 0)
		return -1;
	if (ensure_dev(b, &dB, &ownB) != 0) {
		if (ownA)
			gpu_free(dA);
		return -1;
	}
	if (ensure_dev(c, &dC, &ownC) != 0) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		return -1;
	}
	rc = -1;
	if (cublas_gemm_rm_bt(dC, dA, dB, a->rows, b->rows, a->cols, 1) == 0
	    && launch_done() == 0 && finish_out(c, dC, ownC) == 0)
		rc = 0;
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownC)
		gpu_free(dC);
	return rc;
#else
	(void)c; (void)a; (void)b;
	return -1;
#endif
}

/*
** c = a b^T written only as the BF16 gradient twin of c (training, BF16
** mode): the attention backward reads the twin strided, so the FP32 c is
** never written (it is marked device-resident but holds nothing). -1 =
** caller uses gpu_mat_mul_bt / mat_mul_wt.
*/
extern "C" int gpu_mat_mul_bt_twin(mat *c, const mat *a, const mat *b)
{
#ifdef ANN_CUBLAS
	OpScope _op(OP_GEMM_DGRAD);
	ann_real *dA;
	ann_real *dB;
	ann_real *dC;
	int ownA;
	int ownB;
	int ownC;
	void *tw;
	int rc = -1;

	if (!c || !a || !b || !gpu_available() || g_compute_prec != GPU_PREC_BF16
	    || g_sync_mode == GPU_SYNC_EAGER || !twin_emit_on())
		return -1;
#if CUDART_VERSION >= 12000
	if (g_fp8_dgrad)	/* the FP8 dgrad writes FP32 and would leave the twin unwritten */
		return -1;
#endif
	if (a->cols != b->cols || c->rows != a->rows || c->cols != b->rows)
		return -1;
	if (ensure_dev(a, &dA, &ownA) != 0)
		return -1;
	if (ensure_dev(b, &dB, &ownB) != 0) {
		if (ownA)
			gpu_free(dA);
		return -1;
	}
	if (ensure_dev_out(c, &dC, &ownC) != 0 || ownC) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		return -1;
	}
	pthread_mutex_lock(&g_gemm_mu);
	tw = grad_twin_reserve(dC, (size_t)c->rows * (size_t)c->cols);
	pthread_mutex_unlock(&g_gemm_mu);
	if (tw && cublas_gemm_rm_bt_ex(dC, dA, dB, a->rows, b->rows, a->cols, 0, tw) == 0
	    && launch_done() == 0 && finish_out(c, dC, 0) == 0)
		rc = 0;
	else if (tw) {
		grad_twin_forget(dC);
		(void)cudaGetLastError();
	}
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	return rc;
#else
	(void)c; (void)a; (void)b;
	return -1;
#endif
}

extern "C" int gpu_mat_mul_at(mat *c, const mat *a, const mat *b)
{
	OpScope _op(OP_GEMM_WGRAD);
	ann_real *dA;
	ann_real *dB;
	ann_real *dC;
	int ownA;
	int ownB;
	int ownC;
	dim3 block;
	dim3 grid;
	int rc;

	if (!c || !a || !b || !gpu_available())
		return -1;
	if (a->rows != b->rows || c->rows != a->cols || c->cols != b->cols)
		return -1;
	if (ensure_dev(a, &dA, &ownA) != 0)
		return -1;
	if (ensure_dev(b, &dB, &ownB) != 0) {
		if (ownA)
			gpu_free(dA);
		return -1;
	}
	if (ensure_dev_out(c, &dC, &ownC) != 0) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		return -1;
	}
#ifdef ANN_CUBLAS
	if (cublas_gemm_rm_at(dC, dA, dB, a->rows, a->cols, b->cols, 0) != 0) {
#endif
		block = dim3(16, 16);
		grid = dim3((c->cols + 15) / 16, (c->rows + 15) / 16);
		gemm_at_kernel<<<grid, block, 0, g_st>>>(dA, dB, dC, a->rows, a->cols,
		    b->cols);
#ifdef ANN_CUBLAS
	}
#endif
	rc = -1;
	if (launch_done() == 0 && finish_out(c, dC, ownC) == 0)
		rc = 0;
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownC)
		gpu_free(dC);
	return rc;
}

/*
** dw += a^T b. dw is read-modify-write: a device-authoritative twin is used
** as is; a host-authoritative one is uploaded first (host zero_grad etc.).
*/
extern "C" int gpu_mat_outer_add(mat *dw, const mat *a, const mat *b)
{
	OpScope _op(OP_GEMM_WGRAD);
	ann_real *dW;
	ann_real *dA;
	ann_real *dB;
	int ownW;
	int ownA;
	int ownB;
	dim3 block;
	dim3 grid;
	int rc;

	if (!dw || !a || !b || !gpu_available())
		return -1;
	if (a->rows != b->rows || dw->rows != a->cols || dw->cols != b->cols)
		return -1;
	if (ensure_dev(a, &dA, &ownA) != 0)
		return -1;
	if (ensure_dev(b, &dB, &ownB) != 0) {
		if (ownA)
			gpu_free(dA);
		return -1;
	}
	if (ensure_dev(dw, &dW, &ownW) != 0) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		return -1;
	}
#ifdef ANN_CUBLAS
	if (cublas_gemm_rm_at(dW, dA, dB, a->rows, a->cols, b->cols, 1) != 0) {
#endif
		block = dim3(16, 16);
		grid = dim3((dw->cols + 15) / 16, (dw->rows + 15) / 16);
		outer_add_kernel<<<grid, block, 0, g_st>>>(dW, dA, dB, a->rows, a->cols,
		    b->cols);
#ifdef ANN_CUBLAS
	}
#endif
	rc = -1;
	if (launch_done() == 0 && finish_out(dw, dW, ownW) == 0)
		rc = 0;
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownW)
		gpu_free(dW);
	return rc;
}

/*
** ---- Track K: elementwise kernels + persistent scratch ------------------
**
** Host pointers are staged through per-slot persistent device buffers (grown
** on demand, never freed per call). Device pointers are used in place. One
** mutex serialises the staged path so OpenMP callers can share the slots.
*/
#define EW_SLOTS	8
#define EW_BLOCK	256

static pthread_mutex_t g_ew_mu = PTHREAD_MUTEX_INITIALIZER;

static void *g_ew_buf[EW_SLOTS];
static size_t g_ew_cap[EW_SLOTS];

typedef struct {
	ann_real	*dev;
	void		*host;
	size_t		bytes;
	int		down;
} ew_ref;

static int ew_is_dev(const void *p)
{
	cudaPointerAttributes a;

	/*
	** Legal during a stream capture (probed on CUDA 13.3 in every capture
	** mode), so a host pointer is never taken for a device one there.
	*/
	if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
		(void)cudaGetLastError();
		return 0;
	}
	return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
}

/* Small host vectors (bias/LN) reused across launches until g_wt_epoch bumps. */
/* 128: 12 layers x (LN1, LN2 gamma/beta + FFN b1/b2) + head is about 75. */
#define BIAS_CACHE_SLOTS	128
typedef struct {
	const void	*host;
	void		*dev;
	size_t		bytes;
	unsigned	epoch;
	void		*dev16;		/* BF16 copy of dev (lt_bias_cached) */
	size_t		bytes16;
	const void	*src16;		/* dev it was converted from */
	unsigned	epoch16;
} bias_cache_ent;
static bias_cache_ent g_bias_cache[BIAS_CACHE_SLOTS];

static void *bias_cache_get(const void *host, size_t bytes)
{
	int i;
	int free_slot = -1;
	bias_cache_ent *e;

	if (!host || bytes == 0 || bytes > 64 * 1024)
		return NULL;
	for (i = 0; i < BIAS_CACHE_SLOTS; i++) {
		e = &g_bias_cache[i];
		if (e->host == host) {
			if (e->bytes < bytes || !e->dev) {
				if (e->dev)
					cudaFree(e->dev);
				e->dev = NULL;
				e->bytes = 0;
				if (cudaMalloc(&e->dev, bytes) != cudaSuccess) {
					e->dev = NULL;
					e->host = NULL;
					return NULL;
				}
				e->bytes = bytes;
				e->epoch = g_wt_epoch - 1;	/* force upload */
			}
			if (e->epoch != g_wt_epoch) {
				if (cudaMemcpyAsync(e->dev, host, bytes,
				    cudaMemcpyHostToDevice, g_st) != cudaSuccess)
					return NULL;
				e->epoch = g_wt_epoch;
			} else if (twin_check_on()) {
				unsigned char *tmp = (unsigned char *)malloc(bytes);

				if (tmp && cudaMemcpy(tmp, e->dev, bytes,
				    cudaMemcpyDeviceToHost) == cudaSuccess
				    && memcmp(tmp, host, bytes) != 0)
					fprintf(stderr, "[twin] STALE param host=%p bytes=%zu slot=%d\n",
					    host, bytes, i);
				free(tmp);
			}
			return e->dev;
		}
		if (free_slot < 0 && !e->host)
			free_slot = i;
	}
	if (free_slot < 0)
		return NULL;
	e = &g_bias_cache[free_slot];
	e->host = host;
	e->bytes = bytes;
	e->dev = NULL;
	if (cudaMalloc(&e->dev, bytes) != cudaSuccess) {
		e->host = NULL;
		e->dev = NULL;
		return NULL;
	}
	if (cudaMemcpyAsync(e->dev, host, bytes, cudaMemcpyHostToDevice, g_st)
	    != cudaSuccess) {
		cudaFree(e->dev);
		e->host = NULL;
		e->dev = NULL;
		return NULL;
	}
	e->epoch = g_wt_epoch;
	return e->dev;
}

/*
** B1: device copy of a cuBLASLt epilogue bias (FFN b1 / b2) served from the
** parameter cache above instead of a blocking pageable cudaMemcpy per GEMM
** (24 syncs per microbatch), plus its BF16 copy for the BF16-output W1
** epilogue, converted once per weight epoch with the same f32_to_bf16 RN
** kernel (12 conversions per microbatch saved). Same values, so the GEMMs
** are bit-identical. The host bias only changes in an Adam scatter or a
** load, both of which bump g_wt_epoch (as for the LN and head vectors
** already served from this cache). ANN_LT_BIAS_CACHE=0 restores the
** per-GEMM upload. NULL = use the upload path.
*/
static int lt_bias_cache_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_LT_BIAS_CACHE");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

static const void *lt_bias_cached(const ann_real *host, int N, int bf16)
{
	bias_cache_ent *e = NULL;
	void *dev;
	size_t nb16;
	int i;

	if (!host || N <= 0 || !lt_bias_cache_on())
		return NULL;
	pthread_mutex_lock(&g_ew_mu);
	dev = bias_cache_get(host, (size_t)N * sizeof(ann_real));
	if (!dev || !bf16) {
		pthread_mutex_unlock(&g_ew_mu);
		return dev;
	}
	for (i = 0; i < BIAS_CACHE_SLOTS; i++) {
		if (g_bias_cache[i].host == host && g_bias_cache[i].dev == dev) {
			e = &g_bias_cache[i];
			break;
		}
	}
	nb16 = (size_t)N * 2;
	if (!e) {
		pthread_mutex_unlock(&g_ew_mu);
		return NULL;
	}
	if (!e->dev16 || e->bytes16 < nb16) {
		if (e->dev16)
			cudaFree(e->dev16);
		e->dev16 = NULL;
		e->bytes16 = 0;
		e->src16 = NULL;
		if (cudaMalloc(&e->dev16, nb16) != cudaSuccess) {
			e->dev16 = NULL;
			pthread_mutex_unlock(&g_ew_mu);
			return NULL;
		}
		e->bytes16 = nb16;
	}
	if (e->src16 != dev || e->epoch16 != g_wt_epoch) {
		if (half_wt_convert(e->dev16, (const ann_real *)dev, N, 1) != 0) {
			e->src16 = NULL;
			pthread_mutex_unlock(&g_ew_mu);
			return NULL;
		}
		e->src16 = dev;
		e->epoch16 = g_wt_epoch;
	}
	pthread_mutex_unlock(&g_ew_mu);
	return e->dev16;
}

extern "C" void gpu_host_forget(const void *host)
{
	int i;

	if (!host || !gpu_available())
		return;
	/*
	** A pinned (cudaHostRegister) vector must be unregistered before the
	** heap hands its pages to another allocation: a later vector at an
	** overlapping address would look "already pinned" or fail to register.
	*/
	for (i = 0; i < g_host_pin_n; i++) {
		if (g_host_pin[i] == host) {
			(void)cudaHostUnregister((void *)host);
			g_host_pin[i] = g_host_pin[--g_host_pin_n];
			break;
		}
	}
	pthread_mutex_lock(&g_ew_mu);
	devvec_forget(host);
	for (i = 0; i < BIAS_CACHE_SLOTS; i++) {
		if (g_bias_cache[i].host == host) {
			if (g_bias_cache[i].dev)
				cudaFree(g_bias_cache[i].dev);
			if (g_bias_cache[i].dev16)
				cudaFree(g_bias_cache[i].dev16);
			g_bias_cache[i].dev = NULL;
			g_bias_cache[i].host = NULL;
			g_bias_cache[i].bytes = 0;
			g_bias_cache[i].dev16 = NULL;
			g_bias_cache[i].bytes16 = 0;
			g_bias_cache[i].src16 = NULL;
		}
	}
	for (i = 0; i < COLSUM_ACC_SLOTS; i++) {
		if (g_colsum_acc[i].host == (ann_real *)host) {
			if (g_colsum_acc[i].dev)
				cudaFree(g_colsum_acc[i].dev);
			g_colsum_acc[i].dev = NULL;
			g_colsum_acc[i].host = NULL;
			g_colsum_acc[i].cols = 0;
			g_colsum_acc[i].dirty = 0;
		}
	}
	pthread_mutex_unlock(&g_ew_mu);
}

/*
** Device-resident twins of small host vectors that only kernels read and
** write (LayerNorm per-row mean / inv_std). Keyed on the host address;
** the host copy is refreshed only in eager mode (CPU fallbacks, tests).
** Staging them per call cost a synchronous D2H in every LN forward.
*/
#define DEVVEC_SLOTS	512
typedef struct {
	const void	*host;
	void		*dev;
	size_t		bytes;
} devvec_ent;
static devvec_ent g_devvec[DEVVEC_SLOTS];

static void *devvec_get(const void *host, size_t bytes)
{
	int i;
	int free_slot = -1;
	devvec_ent *e;

	if (!host || bytes == 0)
		return NULL;
	for (i = 0; i < DEVVEC_SLOTS; i++) {
		e = &g_devvec[i];
		if (e->host == host) {
			if (e->bytes < bytes) {
				cudaFree(e->dev);
				e->dev = NULL;
				e->bytes = 0;
				if (cudaMalloc(&e->dev, bytes) != cudaSuccess) {
					e->host = NULL;
					return NULL;
				}
				e->bytes = bytes;
			}
			return e->dev;
		}
		if (free_slot < 0 && !e->host)
			free_slot = i;
	}
	if (free_slot < 0)
		return NULL;
	e = &g_devvec[free_slot];
	if (cudaMalloc(&e->dev, bytes) != cudaSuccess) {
		e->dev = NULL;
		return NULL;
	}
	(void)cudaMemsetAsync(e->dev, 0, bytes, g_st);
	e->host = host;
	e->bytes = bytes;
	return e->dev;
}

static void devvec_forget(const void *host)
{
	int i;

	for (i = 0; i < DEVVEC_SLOTS; i++) {
		if (g_devvec[i].host == host) {
			if (g_devvec[i].dev)
				cudaFree(g_devvec[i].dev);
			g_devvec[i].dev = NULL;
			g_devvec[i].host = NULL;
			g_devvec[i].bytes = 0;
		}
	}
}

/*
** Bind host (or device) memory to a device pointer; slot = scratch id.
** Host memory is staged through the slot's scratch on every call. The
** persistent bias cache is NOT consulted here: it is keyed on the host
** address and only refreshed by gpu_half_wt_invalidate, which is right for
** parameter vectors (ew_acquire_param) and wrong for anything else. CE
** targets (malloc'ed per call at a recycled address) and the LayerNorm
** inv_std of the backward used to hit it and were served stale.
*/
static int ew_acquire(ew_ref *r, int slot, const void *host, size_t bytes,
	int up, int down)
{
	r->dev = NULL;
	r->host = (void *)host;
	r->bytes = bytes;
	r->down = 0;
	if (!host || bytes == 0)
		return -1;
	if (ew_is_dev(host)) {
		r->dev = (ann_real *)host;
		return 0;
	}
	if (g_ew_cap[slot] < bytes) {
		if (g_ew_buf[slot])
			cudaFree(g_ew_buf[slot]);
		g_ew_buf[slot] = NULL;
		g_ew_cap[slot] = 0;
		if (cudaMalloc(&g_ew_buf[slot], bytes) != cudaSuccess) {
			g_ew_buf[slot] = NULL;
			return -1;
		}
		g_ew_cap[slot] = bytes;
	}
	r->dev = (ann_real *)g_ew_buf[slot];
	if (up && cudaMemcpy(r->dev, host, bytes, cudaMemcpyHostToDevice)
	    != cudaSuccess)
		return -1;
	r->down = down;
	return 0;
}

/*
** Read-only parameter vector (bias, LN gamma/beta): served from the
** persistent device cache, re-uploaded after gpu_half_wt_invalidate(NULL)
** (every optimizer step). Falls back to the staged path.
*/
static int ew_acquire_param(ew_ref *r, int slot, const void *host,
	size_t bytes)
{
	r->dev = NULL;
	r->host = (void *)host;
	r->bytes = bytes;
	r->down = 0;
	if (!host || bytes == 0)
		return -1;
	if (bytes <= 64 * 1024) {
		void *cached = bias_cache_get(host, bytes);

		if (cached) {
			r->dev = (ann_real *)cached;
			return 0;
		}
	}
	return ew_acquire(r, slot, host, bytes, 1, 0);
}

/* Mark an aliased ref as needing D2H (only meaningful when staged). */
static void ew_mark_down(ew_ref *r)
{
	if ((void *)r->dev != r->host)
		r->down = 1;
}

/* Copy back if staged; returns 1 if a (synchronising) D2H happened, -1 err. */
static int ew_release(ew_ref *r)
{
	if (!r->down)
		return 0;
	if (cudaMemcpy(r->host, r->dev, r->bytes, cudaMemcpyDeviceToHost)
	    != cudaSuccess)
		return -1;
	return 1;
}

/* Check the launch; EAGER mode with no D2H still waits for the device. */
static int ew_launch_ok(void)
{
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static int ew_finish_sync(int did_d2h)
{
	if (stream_is_capturing())
		return 0;
	if (!did_d2h && g_sync_mode == GPU_SYNC_EAGER
	    && cudaDeviceSynchronize() != cudaSuccess)
		return -1;
	return 0;
}

static int ew_block_for(int n)
{
	int bs;

	bs = 32;
	while (bs < n && bs < EW_BLOCK)
		bs <<= 1;
	return bs;
}

__device__ static ann_real ew_block_sum(ann_real v, ann_real *sh)
{
	int i;

	sh[threadIdx.x] = v;
	__syncthreads();
	for (i = blockDim.x >> 1; i > 0; i >>= 1) {
		if ((int)threadIdx.x < i)
			sh[threadIdx.x] += sh[threadIdx.x + i];
		__syncthreads();
	}
	v = sh[0];
	__syncthreads();
	return v;
}

__device__ static ann_real ew_block_max(ann_real v, ann_real *sh)
{
	int i;

	sh[threadIdx.x] = v;
	__syncthreads();
	for (i = blockDim.x >> 1; i > 0; i >>= 1) {
		if ((int)threadIdx.x < i && sh[threadIdx.x + i] > sh[threadIdx.x])
			sh[threadIdx.x] = sh[threadIdx.x + i];
		__syncthreads();
	}
	v = sh[0];
	__syncthreads();
	return v;
}

/* One block per row; each thread owns a fixed strided set of columns. */
__global__ static void softmax_rows_kernel(ann_real *x, int n_cols)
{
	__shared__ ann_real sh[EW_BLOCK];
	ann_real *row = x + (size_t)blockIdx.x * (size_t)n_cols;
	ann_real m = -INFINITY;
	ann_real s = 0.0f;
	ann_real inv;
	int j;

	for (j = threadIdx.x; j < n_cols; j += blockDim.x)
		if (row[j] > m)
			m = row[j];
	m = ew_block_max(m, sh);
	for (j = threadIdx.x; j < n_cols; j += blockDim.x) {
		row[j] = expf(row[j] - m);
		s += row[j];
	}
	s = ew_block_sum(s, sh);
	inv = 1.0f / s;
	for (j = threadIdx.x; j < n_cols; j += blockDim.x)
		row[j] *= inv;
}

/* dx may alias dy or y: dx[j] is written only by the thread that read it. */
__global__ static void softmax_bwd_kernel(ann_real *dx, const ann_real *y,
	const ann_real *dy, int n_cols)
{
	__shared__ ann_real sh[EW_BLOCK];
	size_t off = (size_t)blockIdx.x * (size_t)n_cols;
	ann_real s = 0.0f;
	int j;

	for (j = threadIdx.x; j < n_cols; j += blockDim.x)
		s += y[off + j] * dy[off + j];
	s = ew_block_sum(s, sh);
	for (j = threadIdx.x; j < n_cols; j += blockDim.x)
		dx[off + j] = y[off + j] * (dy[off + j] - s);
}

/*
** One global load of the row. Mean and variance run on that shared copy,
** then y / x_hat (and optional E4M3 y) are written once.
** Dynamic smem is d floats. d * sizeof(float) must fit the launch limit.
*/
__global__ static void ln_fwd_kernel(ann_real *y, const ann_real *x,
	const ann_real *gamma, const ann_real *beta, ann_real *x_hat,
	ann_real *mean_out, ann_real *inv_out, int d, ann_real eps,
	__nv_fp8_storage_t *y8, const float *scale8, float *amax8,
	__nv_bfloat16 *y16)
{
	extern __shared__ ann_real sx[];
	__shared__ ann_real sh[EW_BLOCK];
	size_t off = (size_t)blockIdx.x * (size_t)d;
	ann_real mean = 0.0f;
	ann_real var = 0.0f;
	ann_real inv;
	ann_real amax = 0.0f;
	ann_real inv_s = 1.0f;
	int j;

	for (j = threadIdx.x; j < d; j += blockDim.x)
		sx[j] = x[off + j];
	__syncthreads();
	for (j = threadIdx.x; j < d; j += blockDim.x)
		mean += sx[j];
	mean = ew_block_sum(mean, sh) / (ann_real)d;
	for (j = threadIdx.x; j < d; j += blockDim.x) {
		ann_real v = sx[j] - mean;

		var += v * v;
	}
	var = ew_block_sum(var, sh) / (ann_real)d;
	inv = 1.0f / sqrtf(var + eps);
	if (y8 && scale8) {
		ann_real s = scale8[0];

		inv_s = (s > 0.f) ? (1.f / s) : 0.f;
	}
	for (j = threadIdx.x; j < d; j += blockDim.x) {
		ann_real xh = (sx[j] - mean) * inv;
		ann_real yv = xh * gamma[j] + beta[j];

		if (x_hat)
			x_hat[off + j] = xh;
		if (y16)	/* BF16-only output: the twin the GEMMs read */
			y16[off + j] = __float2bfloat16(yv);
		else
			y[off + j] = yv;
		if (y8) {
			amax = fmaxf(amax, fabsf(yv));
			y8[off + j] = __nv_cvt_float_to_fp8(yv * inv_s, __NV_SATFINITE,
			    __NV_E4M3);
		}
	}
	if (y8 && amax8) {
		amax = ew_block_max(amax, sh);
		if (threadIdx.x == 0 && amax > 0.f)
			atomicMax((int *)amax8, __float_as_int(amax));
	}
	if (threadIdx.x == 0) {
		if (mean_out)
			mean_out[blockIdx.x] = mean;
		if (inv_out)
			inv_out[blockIdx.x] = inv;
	}
}

/* One block per row; dgamma/dbeta accumulate across rows with atomics. */
__global__ static void ln_bwd_kernel(ann_real *dx, const ann_real *dy,
	const ann_real *gamma, const ann_real *x_hat, const ann_real *inv_std,
	ann_real *dgamma, ann_real *dbeta, int d, int accumulate,
	__nv_bfloat16 *dx16)
{
	__shared__ ann_real sh[EW_BLOCK];
	size_t off = (size_t)blockIdx.x * (size_t)d;
	ann_real sum_dy = 0.0f;
	ann_real sum_dyx = 0.0f;
	ann_real inv = inv_std[blockIdx.x];
	ann_real g;
	ann_real xh;
	int j;

	/* dgamma / dbeta are column sums done by col_sum_tiled_kernel. */
	(void)dgamma;
	(void)dbeta;
	for (j = threadIdx.x; j < d; j += blockDim.x) {
		g = dy[off + j];
		xh = x_hat[off + j];
		g *= gamma[j];
		sum_dy += g;
		sum_dyx += g * xh;
	}
	sum_dy = ew_block_sum(sum_dy, sh);
	sum_dyx = ew_block_sum(sum_dyx, sh);
	for (j = threadIdx.x; j < d; j += blockDim.x) {
		ann_real v;

		g = dy[off + j] * gamma[j];
		v = (g - (sum_dy + x_hat[off + j] * sum_dyx) / (ann_real)d) * inv;
		if (accumulate)
			v += dx[off + j];
		dx[off + j] = v;
		if (dx16)	/* twin for the GEMMs that read this gradient next */
			dx16[off + j] = __float2bfloat16(v);
	}
}

/*
** db[c] += sum over rows of x[r,c] (* x2[r,c] when x2 is given). One thread
** per column and a 64-row slab per block: the slabs run in parallel and each
** column gets rows/64 atomics instead of one serial 4096-row walk per thread
** (col_sum_kernel) or 4096 atomics per column (the old LN backward).
*/
#define COLSUM_SLAB	64
__global__ static void col_sum_tiled_kernel(ann_real *db, const ann_real *x,
	const ann_real *x2, int rows, int cols, ann_real *part)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r0 = blockIdx.y * COLSUM_SLAB;
	int r1 = r0 + COLSUM_SLAB;
	int r;
	float s;

	if (c >= cols)
		return;
	if (r1 > rows)
		r1 = rows;
	s = 0.0f;
	if (x2) {
		for (r = r0; r < r1; r++)
			s += x[(size_t)r * (size_t)cols + (size_t)c]
			    * x2[(size_t)r * (size_t)cols + (size_t)c];
	} else {
		for (r = r0; r < r1; r++)
			s += x[(size_t)r * (size_t)cols + (size_t)c];
	}
	if (part)	/* deterministic: colsum_reduce_kernel adds the slabs in order */
		part[(size_t)blockIdx.y * (size_t)cols + (size_t)c] = s;
	else
		atomicAdd(&db[c], s);
}

/*
** ANN_DET_SUM (default on; 0 = float atomics): a column sum spread over
** several blocks (row slabs, warps, CE row strides) writes one partial row
** per block into g_cs_part, and colsum_reduce_kernel adds the partials to
** the accumulator in one fixed association (runs of consecutive partials,
** then the run sums in order). FP32 sums of the same terms as the atomics,
** whose order is arbitrary too, but always the same one: the atomics' order changed
** from run to run, so the last bit of a bias or LayerNorm gradient did,
** and through Adam and the ReLU masks that bit grew into a different
** trajectory (test_gpu check 7: a 2% gradient gap two steps later in about
** half the runs). A single block per column keeps its one atomic: one add
** onto the accumulator has only one order. g_cs_part is used under g_ew_mu.
*/
static int det_sum_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_DET_SUM");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

static void *g_cs_part;
static size_t g_cs_part_cap;

/* Partial-row scratch of n floats (grow-only; the first microbatch sizes it before any capture); NULL = use the atomics. */
static ann_real *cs_part(size_t n)
{
	if (!det_sum_on() || ensure_half_buf(&g_cs_part, &g_cs_part_cap,
	    n * sizeof(ann_real)) != 0)
		return NULL;
	return (ann_real *)g_cs_part;
}

/*
** acc[c] += the partials part[p * stride + c], p = 0 .. np - 1, in a fixed
** association: CSR_GROUPS threads per column each add a run of consecutive
** partials in order, then the group sums are added to acc in group order.
** Up to 512 partials (the LayerNorm backward's warps at 4096 rows) a single
** thread per column would wait on 512 dependent loads; this waits on
** np / 16 + 16.
*/
#define CSR_COLS	32
#define CSR_GROUPS	16
__global__ static void colsum_reduce_kernel(ann_real *acc,
	const ann_real *part, int np, int cols, int stride)
{
	__shared__ ann_real gs[CSR_GROUPS][CSR_COLS];
	const int cx = threadIdx.x;
	const int g = threadIdx.y;
	const int c = blockIdx.x * CSR_COLS + cx;
	const int per = (np + CSR_GROUPS - 1) / CSR_GROUPS;
	const int p1 = min((g + 1) * per, np);
	ann_real s = 0.0f;
	int p;

	if (c < cols)
		for (p = g * per; p < p1; p++)
			s += part[(size_t)p * (size_t)stride + (size_t)c];
	gs[g][cx] = s;
	__syncthreads();
	if (g == 0 && c < cols) {
		ann_real a = acc[c];
		int k;

		for (k = 0; k < CSR_GROUPS; k++)
			a += gs[k][cx];
		acc[c] = a;
	}
}

static void colsum_reduce_launch(ann_real *acc, const ann_real *part, int np,
	int cols, int stride)
{
	colsum_reduce_kernel<<<(cols + CSR_COLS - 1) / CSR_COLS,
	    dim3(CSR_COLS, CSR_GROUPS), 0, g_st>>>(acc, part, np, cols, stride);
}

/* db[c] += sum over rows of x[r,c] (* x2[r,c]): col_sum_tiled_kernel, slabs reduced in a fixed order. */
static void col_sum_tiled_run(ann_real *db, const ann_real *x,
	const ann_real *x2, int rows, int cols)
{
	dim3 grid((cols + EW_BLOCK - 1) / EW_BLOCK,
	    (rows + COLSUM_SLAB - 1) / COLSUM_SLAB);
	ann_real *part = grid.y > 1 ? cs_part((size_t)grid.y * (size_t)cols)
	    : NULL;

	col_sum_tiled_kernel<<<grid, EW_BLOCK, 0, g_st>>>(db, x, x2, rows, cols,
	    part);
	if (part)
		colsum_reduce_launch(db, part, (int)grid.y, cols, cols);
}

/*
** ANN_LN_FAST: one warp per LayerNorm row, float4 loads/stores, the row held
** in registers (NV float4 per lane, d % 4 == 0, d <= 128 * LNF_MAXNV). Same
** math as ln_fwd_kernel / ln_bwd_kernel; the row sums use shuffles, so the
** FP32 summation order differs (equally exact). Rows per block: LNF_WARPS.
*/
#define LNF_WARPS	4
#define LNF_MAXNV	16
#define LNF_BWD_MINB	4

static int ln_fast_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_LN_FAST");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

__device__ static inline float lnf_warp_sum(float v)
{
	v += __shfl_xor_sync(0xffffffffu, v, 16);
	v += __shfl_xor_sync(0xffffffffu, v, 8);
	v += __shfl_xor_sync(0xffffffffu, v, 4);
	v += __shfl_xor_sync(0xffffffffu, v, 2);
	v += __shfl_xor_sync(0xffffffffu, v, 1);
	return v;
}

template <int NV>
__global__ static void ln_fwd_warp_kernel(ann_real *y, const ann_real *x,
	const ann_real *gamma, const ann_real *beta, ann_real *x_hat,
	ann_real *mean_out, ann_real *inv_out, int d, ann_real eps,
	__nv_bfloat16 *y16, int rows)
{
	int lane = threadIdx.x & 31;
	int row = blockIdx.x * LNF_WARPS + (threadIdx.x >> 5);
	int d4 = d >> 2;
	size_t off;
	float4 v[NV];
	float s = 0.0f;
	float mean;
	float var = 0.0f;
	float inv;
	int i;

	if (row >= rows)
		return;
	off = (size_t)row * (size_t)d4;
	for (i = 0; i < NV; i++) {
		int j = lane + 32 * i;

		if (j < d4) {
			v[i] = ((const float4 *)x)[off + j];
			s += (v[i].x + v[i].y) + (v[i].z + v[i].w);
		}
	}
	mean = lnf_warp_sum(s) / (float)d;
	for (i = 0; i < NV; i++) {
		int j = lane + 32 * i;

		if (j < d4) {
			float a = v[i].x - mean, b = v[i].y - mean;
			float c = v[i].z - mean, e = v[i].w - mean;

			var += (a * a + b * b) + (c * c + e * e);
		}
	}
	var = lnf_warp_sum(var) / (float)d;
	inv = 1.0f / sqrtf(var + eps);
	for (i = 0; i < NV; i++) {
		int j = lane + 32 * i;

		if (j < d4) {
			float4 g = ((const float4 *)gamma)[j];
			float4 b = ((const float4 *)beta)[j];
			float4 h;
			float4 o;

			h.x = (v[i].x - mean) * inv;
			h.y = (v[i].y - mean) * inv;
			h.z = (v[i].z - mean) * inv;
			h.w = (v[i].w - mean) * inv;
			o.x = h.x * g.x + b.x;
			o.y = h.y * g.y + b.y;
			o.z = h.z * g.z + b.z;
			o.w = h.w * g.w + b.w;
			if (x_hat)
				((float4 *)x_hat)[off + j] = h;
			if (y16) {
				__nv_bfloat162 p0 = __floats2bfloat162_rn(o.x, o.y);
				__nv_bfloat162 p1 = __floats2bfloat162_rn(o.z, o.w);
				uint2 u;

				u.x = *(unsigned *)&p0;
				u.y = *(unsigned *)&p1;
				((uint2 *)y16)[off + j] = u;
			} else {
				((float4 *)y)[off + j] = o;
			}
		}
	}
	if (lane == 0) {
		if (mean_out)
			mean_out[row] = mean;
		if (inv_out)
			inv_out[row] = inv;
	}
}

/*
** x_hat for the backward: loaded, or (B12, x_in non-NULL) recomputed from
** the LayerNorm input and the stored row mean / inv_std as the forward
** kernels compute it, (x - mean) * inv with no FMA contraction, so the
** values are bit-identical and the 16 MB x_hat per LayerNorm is never stored.
*/
__device__ __forceinline__ static float4 lnf_xhat(const ann_real *x_hat,
	const ann_real *x_in, size_t i, float mu, float inv)
{
	float4 h;

	if (!x_in)
		return ((const float4 *)x_hat)[i];
	h = ((const float4 *)x_in)[i];
	h.x = __fmul_rn(__fsub_rn(h.x, mu), inv);
	h.y = __fmul_rn(__fsub_rn(h.y, mu), inv);
	h.z = __fmul_rn(__fsub_rn(h.z, mu), inv);
	h.w = __fmul_rn(__fsub_rn(h.w, mu), inv);
	return h;
}

/*
** Backward: each warp owns rpw consecutive rows, keeps its dgamma / dbeta
** partial sums in registers (per-lane columns) and adds them to the global
** accumulators once at the end: no separate column-sum launches. dx may be
** read (accumulate) and written by the same lane only. With part (ANN_DET_SUM)
** the partials go to part[warp][dgamma | dbeta] instead, reduced in a fixed order
** by colsum_reduce_kernel.
*/
template <int NV>
__global__ static void __launch_bounds__(LNF_WARPS * 32, LNF_BWD_MINB) ln_bwd_warp_kernel(
	ann_real *dx, const ann_real *dy, const ann_real *gamma,
	const ann_real *x_hat, const ann_real *inv_std, ann_real *dgamma,
	ann_real *dbeta, int d, int accumulate, __nv_bfloat16 *dx16, int rows,
	int rpw, const ann_real *x_in, const ann_real *mean_in, ann_real *part)
{
	int lane = threadIdx.x & 31;
	int warp = blockIdx.x * LNF_WARPS + (threadIdx.x >> 5);
	int d4 = d >> 2;
	int r0 = warp * rpw;
	int r1 = r0 + rpw;
	float4 ag[NV];
	float4 ab[NV];
	float invd = 1.0f / (float)d;
	int i;
	int r;

	if (r1 > rows)
		r1 = rows;
	if (r0 >= r1)
		return;
#pragma unroll
	for (i = 0; i < NV; i++) {
		ag[i] = make_float4(0.f, 0.f, 0.f, 0.f);
		ab[i] = ag[i];
	}
	for (r = r0; r < r1; r++) {
		size_t off = (size_t)r * (size_t)d4;
		float inv = inv_std[r];
		float mu = x_in ? mean_in[r] : 0.0f;
		float sd = 0.0f;
		float sdx = 0.0f;
		float a;
		float b;

		/*
		** Pass 1 sums, pass 2 reloads dy / x_hat / gamma: they are L1
		** hits and keeping the row in registers made the kernel spill.
		*/
#pragma unroll
		for (i = 0; i < NV; i++) {
			int j = lane + 32 * i;

			if (j < d4) {
				float4 q = ((const float4 *)dy)[off + j];
				float4 h = lnf_xhat(x_hat, x_in, off + j, mu, inv);
				float4 g = ((const float4 *)gamma)[j];
				float tx = q.x * g.x, ty = q.y * g.y;
				float tz = q.z * g.z, tw = q.w * g.w;

				ab[i].x += q.x; ab[i].y += q.y;
				ab[i].z += q.z; ab[i].w += q.w;
				ag[i].x += q.x * h.x; ag[i].y += q.y * h.y;
				ag[i].z += q.z * h.z; ag[i].w += q.w * h.w;
				sd += (tx + ty) + (tz + tw);
				sdx += (tx * h.x + ty * h.y) + (tz * h.z + tw * h.w);
			}
		}
		a = lnf_warp_sum(sd);
		b = lnf_warp_sum(sdx);
#pragma unroll
		for (i = 0; i < NV; i++) {
			int j = lane + 32 * i;

			if (j < d4) {
				float4 q = ((const float4 *)dy)[off + j];
				float4 h = lnf_xhat(x_hat, x_in, off + j, mu, inv);
				float4 g = ((const float4 *)gamma)[j];
				float4 o;

				o.x = (q.x * g.x - (a + h.x * b) * invd) * inv;
				o.y = (q.y * g.y - (a + h.y * b) * invd) * inv;
				o.z = (q.z * g.z - (a + h.z * b) * invd) * inv;
				o.w = (q.w * g.w - (a + h.w * b) * invd) * inv;
				if (accumulate) {
					float4 p = ((const float4 *)dx)[off + j];

					o.x += p.x; o.y += p.y; o.z += p.z; o.w += p.w;
				}
				((float4 *)dx)[off + j] = o;
				if (dx16) {
					__nv_bfloat162 p0 = __floats2bfloat162_rn(o.x, o.y);
					__nv_bfloat162 p1 = __floats2bfloat162_rn(o.z, o.w);
					uint2 u;

					u.x = *(unsigned *)&p0;
					u.y = *(unsigned *)&p1;
					((uint2 *)dx16)[off + j] = u;
				}
			}
		}
	}
#pragma unroll
	for (i = 0; i < NV; i++) {
		int j = lane + 32 * i;

		if (j < d4 && part) {
			float *pw = part + (size_t)warp * 2 * (size_t)d;

			((float4 *)pw)[j] = ag[i];
			((float4 *)(pw + d))[j] = ab[i];
		} else if (j < d4) {
			float *pg = dgamma + 4 * j;
			float *pb = dbeta + 4 * j;

			atomicAdd(pg, ag[i].x); atomicAdd(pg + 1, ag[i].y);
			atomicAdd(pg + 2, ag[i].z); atomicAdd(pg + 3, ag[i].w);
			atomicAdd(pb, ab[i].x); atomicAdd(pb + 1, ab[i].y);
			atomicAdd(pb + 2, ab[i].z); atomicAdd(pb + 3, ab[i].w);
		}
	}
}

/* Eligible shapes / pointers (16-byte alignment for the float4 accesses). */
static int ln_fast_ok(int d, const void *a, const void *b, const void *c,
	const void *e, const void *f)
{
	const void *p[5];
	int i;

	p[0] = a; p[1] = b; p[2] = c; p[3] = e; p[4] = f;
	if (!ln_fast_on() || d % 4 != 0 || d > 128 * LNF_MAXNV)
		return 0;
	for (i = 0; i < 5; i++)
		if (p[i] && ((size_t)p[i] & 15) != 0)
			return 0;
	return 1;
}

#define LNF_DISPATCH(NVV, KERNEL, GRID, ...) \
	switch (NVV) { \
	case 1: KERNEL<1><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 2: KERNEL<2><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 3: KERNEL<3><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 4: KERNEL<4><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 5: KERNEL<5><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 6: KERNEL<6><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 7: KERNEL<7><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 8: KERNEL<8><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 9: KERNEL<9><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 10: KERNEL<10><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 11: KERNEL<11><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 12: KERNEL<12><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 13: KERNEL<13><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 14: KERNEL<14><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	case 15: KERNEL<15><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	default: KERNEL<16><<<GRID, LNF_WARPS * 32, 0, g_st>>>(__VA_ARGS__); break; \
	}

static int ln_fwd_fast_launch(ann_real *y, const ann_real *x,
	const ann_real *gamma, const ann_real *beta, ann_real *x_hat,
	ann_real *mean, ann_real *inv, int T, int d, ann_real eps,
	__nv_bfloat16 *y16)
{
	int nv = (d / 4 + 31) / 32;
	int grid = (T + LNF_WARPS - 1) / LNF_WARPS;

	LNF_DISPATCH(nv, ln_fwd_warp_kernel, grid, y, x, gamma, beta, x_hat,
	    mean, inv, d, eps, y16, T);
	return 0;
}

/* Rows per warp of the backward: enough warps to fill the GPU, few atomics. */
static int ln_bwd_rpw(void)
{
	static int v = 0;

	if (!v) {
		const char *e = getenv("ANN_LN_RPW");

		v = e ? atoi(e) : 8;
		if (v < 1)
			v = 8;
	}
	return v;
}

static int ln_bwd_fast_launch(ann_real *dx, const ann_real *dy,
	const ann_real *gamma, const ann_real *x_hat, const ann_real *inv_std,
	ann_real *dgamma, ann_real *dbeta, int T, int d, int accumulate,
	__nv_bfloat16 *dx16, const ann_real *x_in, const ann_real *mean_in)
{
	int nv = (d / 4 + 31) / 32;
	int rpw = ln_bwd_rpw();
	int warps = (T + rpw - 1) / rpw;
	int grid = (warps + LNF_WARPS - 1) / LNF_WARPS;
	ann_real *part = warps > 1 ? cs_part((size_t)warps * 2 * (size_t)d)
	    : NULL;

	LNF_DISPATCH(nv, ln_bwd_warp_kernel, grid, dx, dy, gamma, x_hat,
	    inv_std, dgamma, dbeta, d, accumulate, dx16, T, rpw, x_in, mean_in, part);
	if (part) {
		colsum_reduce_launch(dgamma, part, warps, d, 2 * d);
		colsum_reduce_launch(dbeta, part + d, warps, d, 2 * d);
	}
	return 0;
}

__global__ static void bias_add_kernel(ann_real *x, const ann_real *bias,
	int n, int cols)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		x[i] += bias[i % cols];
}

/*
** FFN W1 finish: pre already has bias (from Lt epilogue) OR apply bias here
** when bias != NULL. Write ReLU(pre) into hidden.
*/
__global__ static void bias_relu_dst_kernel(ann_real *hidden, ann_real *pre,
	const ann_real *bias, int n, int cols)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float v;

	if (i >= n)
		return;
	v = pre[i];
	if (bias)
		v += bias[i % cols];
	pre[i] = v;
	hidden[i] = v > 0.0f ? v : 0.0f;
}

__global__ static void relu_bwd_kernel(ann_real *dx, const ann_real *ref, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n && ref[i] <= 0.0f)
		dx[i] = 0.0f;
}

static int gpu_softmax_rows_locked(ann_real *rows, int n_rows, int n_cols)
{
	ew_ref r;
	int rc;
	int dn;

	if (!rows || n_rows <= 0 || n_cols <= 0 || !gpu_available())
		return -1;
	if (ew_acquire(&r, 0, rows,
	    (size_t)n_rows * (size_t)n_cols * sizeof(ann_real), 1, 1) != 0)
		return -1;
	softmax_rows_kernel<<<n_rows, ew_block_for(n_cols), 0, g_st>>>(r.dev, n_cols);
	if (ew_launch_ok() != 0)
		return -1;
	rc = ew_release(&r);
	if (rc < 0)
		return -1;
	dn = rc;
	return ew_finish_sync(dn) == 0 ? 0 : -1;
}

extern "C" int gpu_softmax_rows(ann_real *rows, int n_rows, int n_cols)
{
	OpScope _op(OP_SOFTMAX);
	int rc;

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_softmax_rows_locked(rows, n_rows, n_cols);
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}

static int gpu_softmax_bwd_rows_locked(ann_real *dx, const ann_real *y,
	const ann_real *dy, int n_rows, int n_cols)
{
	ew_ref ry;
	ew_ref rdy;
	ew_ref rdx;
	size_t nb;
	int rc;

	if (!dx || !y || !dy || n_rows <= 0 || n_cols <= 0 || !gpu_available())
		return -1;
	nb = (size_t)n_rows * (size_t)n_cols * sizeof(ann_real);
	if (ew_acquire(&ry, 0, y, nb, 1, 0) != 0)
		return -1;
	if (dy == y)
		rdy = ry;
	else if (ew_acquire(&rdy, 1, dy, nb, 1, 0) != 0)
		return -1;
	if (dx == dy) {
		rdx = rdy;
		ew_mark_down(&rdx);
	} else if (dx == y) {
		rdx = ry;
		ew_mark_down(&rdx);
	} else if (ew_acquire(&rdx, 2, dx, nb, 0, 1) != 0) {
		return -1;
	}
	softmax_bwd_kernel<<<n_rows, ew_block_for(n_cols), 0, g_st>>>(rdx.dev, ry.dev,
	    rdy.dev, n_cols);
	if (ew_launch_ok() != 0)
		return -1;
	rc = ew_release(&rdx);
	if (rc < 0)
		return -1;
	return ew_finish_sync(rc) == 0 ? 0 : -1;
}

extern "C" int gpu_softmax_bwd_rows(ann_real *dx, const ann_real *y,
	const ann_real *dy, int n_rows, int n_cols)
{
	int rc;

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_softmax_bwd_rows_locked(dx, y, dy, n_rows, n_cols);
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}


/*
** M1: fused causal multi-head attention over B packed sequences (no per-head
** host loop). Layout: q/k/v/concat are (B*T) x (H*dh) row-major; sequence b
** owns rows [b*T, (b+1)*T). w is the softmax weights, row ((b*H+h)*T + i) of
** width T (rows are causal: zero for j > i). One launch per phase covers every
** (b, h, i) row, so the GPU sees B*H*T independent blocks.
*/
#define ATT_BLOCK	256	/* R11/R6: fused causal attn up to ctx 256 */

__device__ static ann_real att_sum(ann_real v, ann_real *sh)
{
	int i;

	sh[threadIdx.x] = v;
	__syncthreads();
	for (i = blockDim.x >> 1; i > 0; i >>= 1) {
		if ((int)threadIdx.x < i)
			sh[threadIdx.x] += sh[threadIdx.x + i];
		__syncthreads();
	}
	v = sh[0];
	__syncthreads();
	return v;
}

__device__ static ann_real att_max(ann_real v, ann_real *sh)
{
	int i;

	sh[threadIdx.x] = v;
	__syncthreads();
	for (i = blockDim.x >> 1; i > 0; i >>= 1) {
		if ((int)threadIdx.x < i && sh[threadIdx.x + i] > sh[threadIdx.x])
			sh[threadIdx.x] = sh[threadIdx.x + i];
		__syncthreads();
	}
	v = sh[0];
	__syncthreads();
	return v;
}

/* E8: device copy of host seqlens for the current fused launch (optional). */
static int *g_att_seqlens_dev;
static int g_att_seqlens_B;
static const int *g_att_seqlens_host;

extern "C" void gpu_attn_set_seqlens(const int *seqlens)
{
	g_att_seqlens_host = seqlens;
}

/*
** grid = B*H*T blocks (bh, i); dyn smem = (dh + T) floats.
** Optional L[block] = logsumexp for tiled bwd (no T×T HBM).
*/
__global__ static void attn_fwd_kernel(ann_real *concat, ann_real *w,
	const ann_real *q, const ann_real *k, const ann_real *v,
	int T, int H, int dh, ann_real scale, const int *seqlens,
	ann_real *Lstat)
{
	extern __shared__ ann_real sm[];
	__shared__ ann_real red[ATT_BLOCK];
	ann_real *qi = sm;		/* dh */
	ann_real *sc = sm + dh;		/* T */
	int i = blockIdx.x % T;
	int bh = blockIdx.x / T;
	int h = bh % H;
	int b = bh / H;
	int D = H * dh;
	size_t row0 = (size_t)b * T;
	int L = seqlens ? seqlens[b] : T;
	int j;
	int c;
	ann_real mx;
	ann_real sum;
	ann_real acc;
	ann_real *wr = w ? w + (size_t)blockIdx.x * T : NULL;

	if (L < 1)
		L = 1;
	if (L > T)
		L = T;
	for (c = threadIdx.x; c < dh; c += blockDim.x)
		qi[c] = q[(row0 + i) * D + h * dh + c];
	__syncthreads();
	mx = -1e30f;
	for (j = threadIdx.x; j < T; j += blockDim.x) {
		acc = 0.0f;
		if (i < L && j <= i && j < L) {
			const ann_real *kj = k + (row0 + j) * D + h * dh;
			for (c = 0; c < dh; c++)
				acc += qi[c] * kj[c];
			acc *= scale;
		} else {
			acc = -1e30f;
		}
		sc[j] = acc;
		if (acc > mx)
			mx = acc;
	}
	mx = att_max(mx, red);
	sum = 0.0f;
	for (j = threadIdx.x; j < T; j += blockDim.x) {
		acc = (i < L && j <= i && j < L) ? __expf(sc[j] - mx) : 0.0f;
		sc[j] = acc;
		sum += acc;
	}
	sum = att_sum(sum, red);
	if (sum < 1e-20f)
		sum = 1.0f;
	if (Lstat && threadIdx.x == 0)
		Lstat[blockIdx.x] = mx + __logf(sum);
	for (j = threadIdx.x; j < T; j += blockDim.x) {
		sc[j] /= sum;
		if (wr)
			wr[j] = sc[j];
	}
	__syncthreads();
	for (c = threadIdx.x; c < dh; c += blockDim.x) {
		acc = 0.0f;
		if (i < L) {
			for (j = 0; j <= i && j < L; j++)
				acc += sc[j] * v[(row0 + j) * D + h * dh + c];
		}
		concat[(row0 + i) * D + h * dh + c] = acc;
	}
}

/*
** Backward phase 1: per (bh, i): dA_j = do_i . v_j, dS_ij = A_ij (dA_j - sum)
** * scale (zero for j > i), written to ds; dQ_i = sum_j dS_ij k_j.
*/
__global__ static void attn_bwd_rows_kernel(ann_real *dq, ann_real *ds,
	const ann_real *dconcat, const ann_real *w, const ann_real *k,
	const ann_real *v, int T, int H, int dh, ann_real scale)
{
	extern __shared__ ann_real sm[];
	__shared__ ann_real red[ATT_BLOCK];
	ann_real *doi = sm;		/* dh */
	ann_real *dsr = sm + dh;	/* T */
	int i = blockIdx.x % T;
	int bh = blockIdx.x / T;
	int h = bh % H;
	int b = bh / H;
	int D = H * dh;
	size_t row0 = (size_t)b * T;
	const ann_real *wr = w + (size_t)blockIdx.x * T;
	ann_real *dsg = ds + (size_t)blockIdx.x * T;
	int j;
	int c;
	ann_real acc;
	ann_real dot;

	for (c = threadIdx.x; c < dh; c += blockDim.x)
		doi[c] = dconcat[(row0 + i) * D + h * dh + c];
	__syncthreads();
	dot = 0.0f;
	for (j = threadIdx.x; j < T; j += blockDim.x) {
		acc = 0.0f;
		if (j <= i) {
			const ann_real *vj = v + (row0 + j) * D + h * dh;
			for (c = 0; c < dh; c++)
				acc += doi[c] * vj[c];
		}
		dsr[j] = acc;			/* dA_j */
		dot += wr[j] * acc;
	}
	dot = att_sum(dot, red);
	for (j = threadIdx.x; j < T; j += blockDim.x) {
		acc = (j <= i) ? wr[j] * (dsr[j] - dot) * scale : 0.0f;
		dsr[j] = acc;
		dsg[j] = acc;
	}
	__syncthreads();
	for (c = threadIdx.x; c < dh; c += blockDim.x) {
		acc = 0.0f;
		for (j = 0; j <= i; j++)
			acc += dsr[j] * k[(row0 + j) * D + h * dh + c];
		dq[(row0 + i) * D + h * dh + c] = acc;
	}
}

/* Backward phase 2: per (bh, j): dK_j = sum_{i>=j} dS_ij q_i, dV_j = sum A_ij do_i. */
__global__ static void attn_bwd_cols_kernel(ann_real *dk, ann_real *dv,
	const ann_real *dconcat, const ann_real *w, const ann_real *ds,
	const ann_real *q, int T, int H, int dh)
{
	int j = blockIdx.x % T;
	int bh = blockIdx.x / T;
	int h = bh % H;
	int b = bh / H;
	int D = H * dh;
	size_t row0 = (size_t)b * T;
	const ann_real *wb = w + (size_t)bh * T * T;
	const ann_real *dsb = ds + (size_t)bh * T * T;
	int i;
	int c;
	ann_real ak;
	ann_real av;

	for (c = threadIdx.x; c < dh; c += blockDim.x) {
		ak = 0.0f;
		av = 0.0f;
		for (i = j; i < T; i++) {
			ak += dsb[(size_t)i * T + j]
			    * q[(row0 + i) * D + h * dh + c];
			av += wb[(size_t)i * T + j]
			    * dconcat[(row0 + i) * D + h * dh + c];
		}
		dk[(row0 + j) * D + h * dh + c] = ak;
		dv[(row0 + j) * D + h * dh + c] = av;
	}
}

static pthread_mutex_t g_att_mu = PTHREAD_MUTEX_INITIALIZER;
static ann_real *g_att_ds;
static size_t g_att_ds_cap;
#define FA_BC	32	/* KV tile width for tiled bwd */

static int att_args_ok(int B, int T, int H, int dh)
{
	return B > 0 && T > 0 && T <= ATT_BLOCK && H > 0 && dh > 0
	    && gpu_available();
}

extern "C" int gpu_attn_causal_fwd(mat *concat, mat *w, const mat *q,
	const mat *k, const mat *v, int B, int T, int H, int dh)
{
	ann_real *dq;
	ann_real *dk;
	ann_real *dv;
	ann_real *dc;
	ann_real *dw;
	int oq;
	int ok;
	int ov;
	int oc;
	int rc;
	size_t smem;

	if (!concat || !w || !q || !k || !v || !att_args_ok(B, T, H, dh))
		return -1;
	if (q->rows != B * T || q->cols != H * dh || k->rows != q->rows
	    || v->rows != q->rows || concat->rows != q->rows
	    || concat->cols != q->cols || w->rows != B * H * T || w->cols != T
	    || !w->d_data)
		return -1;
	pthread_mutex_lock(&g_att_mu);
	rc = -1;
	oq = ok = ov = oc = 0;
	dq = dk = dv = dc = NULL;
	if (ensure_dev(q, &dq, &oq) != 0)
		goto out;
	if (ensure_dev(k, &dk, &ok) != 0)
		goto out;
	if (ensure_dev(v, &dv, &ov) != 0)
		goto out;
	if (ensure_dev_out(concat, &dc, &oc) != 0)
		goto out;
	dw = w->d_data;
	smem = (size_t)(dh + T) * sizeof(ann_real);
	/* E8: upload seqlens when set for this pack. */
	if (g_att_seqlens_host) {
		if (!g_att_seqlens_dev || g_att_seqlens_B < B) {
			if (g_att_seqlens_dev)
				gpu_free(g_att_seqlens_dev);
			g_att_seqlens_dev = (int *)gpu_malloc((size_t)B
			    * sizeof(int));
			g_att_seqlens_B = g_att_seqlens_dev ? B : 0;
		}
		if (g_att_seqlens_dev
		    && gpu_memcpy_h2d(g_att_seqlens_dev, g_att_seqlens_host,
		    (size_t)B * sizeof(int)) != 0)
			goto out;
	}
	attn_fwd_kernel<<<B * H * T, ATT_BLOCK, smem, g_st>>>(dc, dw, dq, dk, dv,
	    T, H, dh, (ann_real)(1.0 / sqrt((double)dh)),
	    g_att_seqlens_host ? g_att_seqlens_dev : NULL, NULL);
	if (launch_done() == 0 && finish_out(concat, dc, oc) == 0) {
		w->on_device = MAT_DEV_DEVICE;	/* host copy never read */
		rc = 0;
	}
out:
	if (oq && dq)
		gpu_free(dq);
	if (ok && dk)
		gpu_free(dk);
	if (ov && dv)
		gpu_free(dv);
	if (oc && dc)
		gpu_free(dc);
	pthread_mutex_unlock(&g_att_mu);
	return rc;
}

extern "C" int gpu_attn_flash_fwd(mat *concat, const mat *q, const mat *k,
	const mat *v, int B, int T, int H, int dh, mat *lse)
{
	OpScope _op(OP_ATTN_FWD);
	ann_real *dq;
	ann_real *dk;
	ann_real *dv;
	ann_real *dc;
	ann_real *dL;
	int oq;
	int ok;
	int ov;
	int oc;
	int oL;
	int rc;
	size_t smem;

	if (!concat || !q || !k || !v || !lse || !att_args_ok(B, T, H, dh)
	    || !lse->d_data)
		return -1;
	if (q->rows != B * T || q->cols != H * dh || k->rows != q->rows
	    || v->rows != q->rows || concat->rows != q->rows
	    || concat->cols != q->cols || lse->rows != B * H * T
	    || lse->cols != 1)
		return -1;
	pthread_mutex_lock(&g_att_mu);
	rc = -1;
	oq = ok = ov = oc = oL = 0;
	dq = dk = dv = dc = dL = NULL;
	if (ensure_dev(q, &dq, &oq) != 0)
		goto flash_out;
	if (ensure_dev(k, &dk, &ok) != 0)
		goto flash_out;
	if (ensure_dev(v, &dv, &ov) != 0)
		goto flash_out;
	if (ensure_dev_out(concat, &dc, &oc) != 0)
		goto flash_out;
	if (ensure_dev_out(lse, &dL, &oL) != 0)
		goto flash_out;
	smem = (size_t)(dh + T) * sizeof(ann_real);
	if (g_att_seqlens_host) {
		if (!g_att_seqlens_dev || g_att_seqlens_B < B) {
			if (g_att_seqlens_dev)
				gpu_free(g_att_seqlens_dev);
			g_att_seqlens_dev = (int *)gpu_malloc((size_t)B
			    * sizeof(int));
			g_att_seqlens_B = g_att_seqlens_dev ? B : 0;
		}
		if (g_att_seqlens_dev
		    && gpu_memcpy_h2d(g_att_seqlens_dev, g_att_seqlens_host,
		    (size_t)B * sizeof(int)) != 0)
			goto flash_out;
	}
	attn_fwd_kernel<<<B * H * T, ATT_BLOCK, smem, g_st>>>(dc, NULL, dq, dk, dv,
	    T, H, dh, (ann_real)(1.0 / sqrt((double)dh)),
	    g_att_seqlens_host ? g_att_seqlens_dev : NULL, dL);
	if (launch_done() == 0 && finish_out(concat, dc, oc) == 0
	    && finish_out(lse, dL, oL) == 0)
		rc = 0;
flash_out:
	if (oq && dq)
		gpu_free(dq);
	if (ok && dk)
		gpu_free(dk);
	if (ov && dv)
		gpu_free(dv);
	if (oc && dc)
		gpu_free(dc);
	if (oL && dL)
		gpu_free(dL);
	pthread_mutex_unlock(&g_att_mu);
	return rc;
}

/*
** FlashAttention-2 backward (Dao, 2023, Alg. 2). Recompute
** P_ij = exp(S_ij - L_i) in KV tiles of width FA_BC. L_i is the forward
** row logsumexp of this same causal softmax (S_ij = -inf for j > i,
** and outside seqlens). No B·H·T·T scores and no dS tensor in HBM.
**
** A cuDNN SDPA forward is a different softmax unless it uses this mask,
** this scale, and this L. Pairing it with this backward would make
** dQ/dK/dV the gradient of another function than the O the forward
** wrote. Attention is about 1% of FLOPs at T=64, which only orders the
** work. T is not shortened and causal masking stays.
*/
__global__ static void attn_tiled_bwd_kernel(ann_real *dq, ann_real *dk,
	ann_real *dv, const ann_real *dO, const ann_real *O, const ann_real *q,
	const ann_real *k, const ann_real *v, const ann_real *Lstat, int T,
	int H, int dh, ann_real scale, const int *seqlens)
{
	extern __shared__ ann_real sm[];
	__shared__ ann_real red[ATT_BLOCK];
	ann_real *qi = sm;
	ann_real *doi = sm + dh;
	ann_real *oi = sm + 2 * dh;
	ann_real *dqi = sm + 3 * dh;
	ann_real *Kt = sm + 4 * dh;
	ann_real *Vt = Kt + FA_BC * dh;
	int i = blockIdx.x % T;
	int bh = blockIdx.x / T;
	int h = bh % H;
	int b = bh / H;
	int D = H * dh;
	size_t row0 = (size_t)b * T;
	int L = seqlens ? seqlens[b] : T;
	int j0;
	int j;
	int c;
	ann_real Di;
	ann_real Li;

	if (L < 1)
		L = 1;
	if (L > T)
		L = T;
	for (c = threadIdx.x; c < dh; c += blockDim.x) {
		qi[c] = q[(row0 + i) * D + h * dh + c];
		doi[c] = dO[(row0 + i) * D + h * dh + c];
		oi[c] = O[(row0 + i) * D + h * dh + c];
		dqi[c] = 0.0f;
	}
	__syncthreads();
	Di = 0.0f;
	for (c = threadIdx.x; c < dh; c += blockDim.x)
		Di += doi[c] * oi[c];
	Di = att_sum(Di, red);
	Li = Lstat[blockIdx.x];
	if (!(i < L)) {
		for (c = threadIdx.x; c < dh; c += blockDim.x)
			dq[(row0 + i) * D + h * dh + c] = 0.0f;
		return;
	}
	for (j0 = 0; j0 < T; j0 += FA_BC) {
		int j1 = j0 + FA_BC;

		/* Tile is entirely in the future of this causal softmax. */
		if (j0 > i)
			break;
		if (j1 > T)
			j1 = T;
		for (j = j0 + (int)threadIdx.x; j < j1; j += (int)blockDim.x) {
			const ann_real *kj = k + (row0 + j) * D + h * dh;
			const ann_real *vj = v + (row0 + j) * D + h * dh;
			int t;

			for (t = 0; t < dh; t++) {
				Kt[(j - j0) * dh + t] = kj[t];
				Vt[(j - j0) * dh + t] = vj[t];
			}
		}
		__syncthreads();
		for (j = j0; j < j1; j++) {
			ann_real s;
			ann_real p;
			ann_real dp;
			ann_real ds;
			int ok = (j <= i && j < L) ? 1 : 0;

			s = 0.0f;
			if (ok) {
				for (c = threadIdx.x; c < dh; c += blockDim.x)
					s += qi[c] * Kt[(j - j0) * dh + c];
			}
			s = att_sum(s, red) * scale;
			p = ok ? __expf(s - Li) : 0.0f;
			dp = 0.0f;
			if (ok) {
				for (c = threadIdx.x; c < dh; c += blockDim.x)
					dp += doi[c] * Vt[(j - j0) * dh + c];
			}
			dp = att_sum(dp, red);
			ds = ok ? p * (dp - Di) * scale : 0.0f;
			if (ok) {
				for (c = threadIdx.x; c < dh; c += blockDim.x) {
					dqi[c] += ds * Kt[(j - j0) * dh + c];
					atomicAdd(&dk[(row0 + j) * D + h * dh
					    + c], ds * qi[c]);
					atomicAdd(&dv[(row0 + j) * D + h * dh
					    + c], p * doi[c]);
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
	for (c = threadIdx.x; c < dh; c += blockDim.x)
		dq[(row0 + i) * D + h * dh + c] = dqi[c];
}

extern "C" int gpu_attn_tiled_bwd(mat *dq, mat *dk, mat *dv,
	const mat *dconcat, const mat *concat, const mat *q, const mat *k,
	const mat *v, const mat *lse, int B, int T, int H, int dh)
{
	OpScope _op(OP_ATTN_BWD);
	ann_real *a_dq;
	ann_real *a_dk;
	ann_real *a_dv;
	ann_real *a_dc;
	ann_real *a_o;
	ann_real *a_q;
	ann_real *a_k;
	ann_real *a_v;
	ann_real *a_L;
	size_t smem;
	size_t nbytes;
	int o[8];
	ann_real *ptr[8];
	int oL;
	int i;
	int rc;

	if (!dq || !dk || !dv || !dconcat || !concat || !q || !k || !v
	    || !lse || !att_args_ok(B, T, H, dh) || !lse->d_data)
		return -1;
	if (q->rows != B * T || q->cols != H * dh || dq->rows != q->rows
	    || dk->rows != q->rows || dv->rows != q->rows
	    || dconcat->rows != q->rows || concat->rows != q->rows
	    || lse->rows != B * H * T || lse->cols != 1)
		return -1;
	pthread_mutex_lock(&g_att_mu);
	rc = -1;
	oL = 0;
	a_L = NULL;
	for (i = 0; i < 8; i++) {
		o[i] = 0;
		ptr[i] = NULL;
	}
	if (ensure_dev(dconcat, &ptr[0], &o[0]) != 0
	    || ensure_dev(concat, &ptr[1], &o[1]) != 0
	    || ensure_dev(q, &ptr[2], &o[2]) != 0
	    || ensure_dev(k, &ptr[3], &o[3]) != 0
	    || ensure_dev(v, &ptr[4], &o[4]) != 0
	    || ensure_dev(lse, &a_L, &oL) != 0
	    || ensure_dev_out(dq, &ptr[5], &o[5]) != 0
	    || ensure_dev_out(dk, &ptr[6], &o[6]) != 0
	    || ensure_dev_out(dv, &ptr[7], &o[7]) != 0)
		goto tiled_bwd_out;
	a_dc = ptr[0];
	a_o = ptr[1];
	a_q = ptr[2];
	a_k = ptr[3];
	a_v = ptr[4];
	a_dq = ptr[5];
	a_dk = ptr[6];
	a_dv = ptr[7];
	nbytes = (size_t)B * (size_t)T * (size_t)(H * dh) * sizeof(ann_real);
	if (cudaMemsetAsync(a_dk, 0, nbytes, g_st) != cudaSuccess
	    || cudaMemsetAsync(a_dv, 0, nbytes, g_st) != cudaSuccess)
		goto tiled_bwd_out;
	smem = (size_t)(4 * dh + 2 * FA_BC * dh) * sizeof(ann_real);
	attn_tiled_bwd_kernel<<<B * H * T, ATT_BLOCK, smem, g_st>>>(a_dq, a_dk, a_dv,
	    a_dc, a_o, a_q, a_k, a_v, a_L, T, H, dh,
	    (ann_real)(1.0 / sqrt((double)dh)),
	    g_att_seqlens_host ? g_att_seqlens_dev : NULL);
	if (launch_done() == 0 && finish_out(dq, a_dq, o[5]) == 0
	    && finish_out(dk, a_dk, o[6]) == 0
	    && finish_out(dv, a_dv, o[7]) == 0)
		rc = 0;
tiled_bwd_out:
	for (i = 0; i < 8; i++)
		if (o[i] && ptr[i])
			gpu_free(ptr[i]);
	if (oL && a_L)
		gpu_free(a_L);
	pthread_mutex_unlock(&g_att_mu);
	return rc;
}

/*
** FFN W1 finish after GEMM (optionally already bias-fused into pre):
** pre += bias (if bias non-NULL), hidden = ReLU(pre), publish BF16(hidden).
*/
extern "C" int gpu_ffn_w1_finish(ann_real *hidden, ann_real *pre,
	const ann_real *bias, int rows, int cols)
{
	OpScope _op(OP_ELEMWISE);
	ew_ref rh;
	ew_ref rp;
	ew_ref rb;
	size_t nb;
	int n;
	int rc;
	int dn;
	int have_bias;

	if (!hidden || !pre || rows <= 0 || cols <= 0 || !gpu_available())
		return -1;
	n = rows * cols;
	nb = (size_t)n * sizeof(ann_real);
	have_bias = bias != NULL;
	pthread_mutex_lock(&g_ew_mu);
	if (ew_acquire(&rp, 0, pre, nb, 1, 1) != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	if (ew_acquire(&rh, 1, hidden, nb, 0, 1) != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	if (have_bias
	    && ew_acquire_param(&rb, 2, bias, (size_t)cols * sizeof(ann_real))
	    != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	bias_relu_dst_kernel<<<(n + EW_BLOCK - 1) / EW_BLOCK, EW_BLOCK, 0, g_st>>>(
	    rh.dev, rp.dev, have_bias ? rb.dev : NULL, n, cols);
	if (ew_launch_ok() != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	dn = 0;
	if ((rc = ew_release(&rh)) < 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	dn |= rc;
	if ((rc = ew_release(&rp)) < 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	dn |= rc;
	rc = ew_finish_sync(dn) == 0 ? 0 : -1;
	pthread_mutex_unlock(&g_ew_mu);
	if (rc == 0)
		act_bf16_publish(hidden, n);
	return rc;
}

/* ANN_TWIN_EMIT=0: kernels do not write gradient twins (dK/dV, LayerNorm backward). */
static int twin_emit_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_TWIN_EMIT");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

static int ln_bf16_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_LN_BF16");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

static int hid_bf16_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_HID_BF16");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

static int gpu_mat_mul_epi(mat *c, const mat *a, const mat *b,
	const ann_real *bias, int with_relu)
{
	OpScope _op(OP_GEMM_FWD);
	ann_real *dA;
	ann_real *dB;
	ann_real *dC;
	int ownA;
	int ownB;
	int ownC;
	int rc;
	int bf16_only_out = 0;

	if (!c || !a || !b || !bias)
		return -1;
	if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols)
		return -1;
	if (!gpu_available())
		return -1;
	if (ensure_dev(a, &dA, &ownA) != 0)
		return -1;
	if (ensure_dev(b, &dB, &ownB) != 0) {
		if (ownA)
			gpu_free(dA);
		return -1;
	}
	if (ensure_dev_out(c, &dC, &ownC) != 0) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		return -1;
	}
#ifdef ANN_CUBLAS
	rc = -1;
	/*
	** Training (batch mode), BF16: ReLU(x W1 + b1) written once as the
	** BF16 twin of c, which is all the W2 GEMM, its wgrad and the fused
	** ReLU backward read; the FP32 c is never written (ANN_HID_BF16=0
	** keeps the FP32 output plus a conversion).
	*/
	if (with_relu && !ownC && g_compute_prec == GPU_PREC_BF16
	    && g_sync_mode != GPU_SYNC_EAGER && hid_bf16_on()) {
		act_bf16_slot *as = act_bf16_reserve(dC, a->rows * b->cols);

		if (as) {
			rc = cublas_gemm_rm_epi_res((ann_real *)as->half, dA, dB, a->rows,
			    b->cols, a->cols, bias, CUBLASLT_EPILOGUE_RELU_BIAS, NULL, 1);
			if (rc == 0) {
				static int reported;

				as->live = 1;
				bf16_only_out = 1;
				if (!reported++)
					fprintf(stderr, "[ffn] W1: ReLU output written as its BF16 twin only\n");
			} else {
				as->live = 0;
				as->bf16_only = 0;
				(void)cudaGetLastError();
			}
		}
	}
	if (rc != 0)
		rc = with_relu
		    ? cublas_gemm_rm_bias_relu(dC, dA, dB, a->rows, b->cols, a->cols,
		    bias)
		    : cublas_gemm_rm_bias(dC, dA, dB, a->rows, b->cols, a->cols, bias);
#else
	(void)with_relu;
	rc = -1;
#endif
	if (rc != 0) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		if (ownC)
			gpu_free(dC);
		return -1;
	}
	rc = -1;
	if (launch_done() == 0 && finish_out(c, dC, ownC) == 0) {
		rc = 0;
		if (with_relu && c->d_data && !bf16_only_out)
			act_bf16_publish(c->d_data, c->rows * c->cols);
	}
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownC)
		gpu_free(dC);
	return rc;
}

extern "C" int gpu_mat_mul_bias(mat *c, const mat *a, const mat *b,
	const ann_real *bias)
{
	return gpu_mat_mul_epi(c, a, b, bias, 0);
}

/*
** d = a b (+ bias) + res in one cuBLASLt call (BF16 mode): the residual
** add of a projection fused into its GEMM. -1 = caller does it in steps.
*/
extern "C" int gpu_mat_mul_res(mat *d, const mat *a, const mat *b,
	const ann_real *bias, const mat *res)
{
	OpScope _op(OP_GEMM_FWD);
#ifdef ANN_CUBLAS
	ann_real *dA, *dB, *dD, *dR;
	int ownA, ownB, ownD, ownR, rc;

	if (!d || !a || !b || !res || d == res)
		return -1;
	if (a->cols != b->rows || d->rows != a->rows || d->cols != b->cols
	    || res->rows != d->rows || res->cols != d->cols)
		return -1;
	if (!gpu_available() || g_compute_prec != GPU_PREC_BF16)
		return -1;
	if (ensure_dev(a, &dA, &ownA) != 0)
		return -1;
	if (ensure_dev(b, &dB, &ownB) != 0) {
		if (ownA)
			gpu_free(dA);
		return -1;
	}
	if (ensure_dev(res, &dR, &ownR) != 0) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		return -1;
	}
	if (ensure_dev_out(d, &dD, &ownD) != 0) {
		if (ownA)
			gpu_free(dA);
		if (ownB)
			gpu_free(dB);
		if (ownR)
			gpu_free(dR);
		return -1;
	}
	rc = cublas_gemm_rm_epi_res(dD, dA, dB, a->rows, b->cols, a->cols, bias,
	    bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT, dR, 0);
	if (rc == 0)
		rc = (launch_done() == 0 && finish_out(d, dD, ownD) == 0) ? 0 : -1;
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownR)
		gpu_free(dR);
	if (ownD)
		gpu_free(dD);
	return rc;
#else
	(void)d; (void)a; (void)b; (void)bias; (void)res;
	return -1;
#endif
}

extern "C" int gpu_mat_mul_bias_relu(mat *c, const mat *a, const mat *b,
	const ann_real *bias)
{
	return gpu_mat_mul_epi(c, a, b, bias, 1);
}

extern "C" int gpu_attn_causal_bwd(mat *dq, mat *dk, mat *dv,
	const mat *dconcat, const mat *w, const mat *q, const mat *k,
	const mat *v, int B, int T, int H, int dh)
{
	ann_real *a_dq;
	ann_real *a_dk;
	ann_real *a_dv;
	ann_real *a_dc;
	ann_real *a_q;
	ann_real *a_k;
	ann_real *a_v;
	size_t need;
	size_t smem;
	int o[7];
	int rc;
	int i;
	ann_real *ptr[7];

	if (!dq || !dk || !dv || !dconcat || !w || !q || !k || !v
	    || !att_args_ok(B, T, H, dh) || !w->d_data)
		return -1;
	if (q->rows != B * T || q->cols != H * dh || dq->rows != q->rows
	    || dk->rows != q->rows || dv->rows != q->rows
	    || dconcat->rows != q->rows || w->rows != B * H * T)
		return -1;
	pthread_mutex_lock(&g_att_mu);
	rc = -1;
	for (i = 0; i < 7; i++) {
		o[i] = 0;
		ptr[i] = NULL;
	}
	need = (size_t)B * H * T * T * sizeof(ann_real);
	if (g_att_ds_cap < need) {
		if (g_att_ds)
			cudaFree(g_att_ds);
		g_att_ds = NULL;
		g_att_ds_cap = 0;
		if (cudaMalloc((void **)&g_att_ds, need) != cudaSuccess) {
			g_att_ds = NULL;
			goto out;
		}
		g_att_ds_cap = need;
	}
	if (ensure_dev(dconcat, &ptr[0], &o[0]) != 0
	    || ensure_dev(q, &ptr[1], &o[1]) != 0
	    || ensure_dev(k, &ptr[2], &o[2]) != 0
	    || ensure_dev(v, &ptr[3], &o[3]) != 0
	    || ensure_dev_out(dq, &ptr[4], &o[4]) != 0
	    || ensure_dev_out(dk, &ptr[5], &o[5]) != 0
	    || ensure_dev_out(dv, &ptr[6], &o[6]) != 0)
		goto out;
	a_dc = ptr[0];
	a_q = ptr[1];
	a_k = ptr[2];
	a_v = ptr[3];
	a_dq = ptr[4];
	a_dk = ptr[5];
	a_dv = ptr[6];
	smem = (size_t)(dh + T) * sizeof(ann_real);
	attn_bwd_rows_kernel<<<B * H * T, ATT_BLOCK, smem, g_st>>>(a_dq, g_att_ds,
	    a_dc, w->d_data, a_k, a_v, T, H, dh,
	    (ann_real)(1.0 / sqrt((double)dh)));
	attn_bwd_cols_kernel<<<B * H * T, ATT_BLOCK, 0, g_st>>>(a_dk, a_dv, a_dc,
	    w->d_data, g_att_ds, a_q, T, H, dh);
	if (launch_done() == 0 && finish_out(dq, a_dq, o[4]) == 0
	    && finish_out(dk, a_dk, o[5]) == 0
	    && finish_out(dv, a_dv, o[6]) == 0)
		rc = 0;
out:
	for (i = 0; i < 7; i++)
		if (o[i] && ptr[i])
			gpu_free(ptr[i]);
	pthread_mutex_unlock(&g_att_mu);
	return rc;
}

/* ---- TC batched attention (40TF wave): pack heads + strided GemmEx ---- */

static ann_real *g_tc_q;
static ann_real *g_tc_k;
static ann_real *g_tc_v;
static ann_real *g_tc_o;
static ann_real *g_tc_dp;
static size_t g_tc_pack_cap;
static size_t g_tc_score_cap;
static void *g_tc_half_q;
static void *g_tc_half_k;
static size_t g_tc_half_q_cap;
static size_t g_tc_half_k_cap;
/* BF16 attention products: V, dO (packed heads), P and dS (T x T rows). */
static void *g_tc_half_v;
static void *g_tc_half_do;
static void *g_tc_half_p;
static void *g_tc_half_ds;
static size_t g_tc_half_v_cap;
static size_t g_tc_half_do_cap;
static size_t g_tc_half_p_cap;
static size_t g_tc_half_ds_cap;

__global__ static void attn_pack_heads_kernel(ann_real *out, const ann_real *in,
	int B, int T, int H, int dh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n = B * T * H * dh;
	int c;
	int t;
	int h;
	int b;
	int D;

	if (idx >= n)
		return;
	D = H * dh;
	c = idx % dh;
	t = (idx / dh) % T;
	h = (idx / (dh * T)) % H;
	b = idx / (dh * T * H);
	out[((b * H + h) * T + t) * dh + c] =
	    in[(b * T + t) * D + h * dh + c];
}

/* Pack heads and cast to BF16 in one pass (TC QK skips a later convert). */
__global__ static void attn_pack_heads_bf16_kernel(__nv_bfloat16 *out,
	const ann_real *in, int B, int T, int H, int dh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n = B * T * H * dh;
	int c;
	int t;
	int h;
	int b;
	int D;

	if (idx >= n)
		return;
	D = H * dh;
	c = idx % dh;
	t = (idx / dh) % T;
	h = (idx / (dh * T)) % H;
	b = idx / (dh * T * H);
	out[((b * H + h) * T + t) * dh + c] =
	    __float2bfloat16(in[(b * T + t) * D + h * dh + c]);
}

__global__ static void attn_unpack_heads_kernel(ann_real *out,
	const ann_real *in, int B, int T, int H, int dh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n = B * T * H * dh;
	int c;
	int t;
	int h;
	int b;
	int D;

	if (idx >= n)
		return;
	D = H * dh;
	c = idx % dh;
	t = (idx / dh) % T;
	h = (idx / (dh * T)) % H;
	b = idx / (dh * T * H);
	out[(b * T + t) * D + h * dh + c] =
	    in[((b * H + h) * T + t) * dh + c];
}

/* One block per (b,h,i) row: causal scale+softmax (parallel reduce; T<=256). */
__global__ static void attn_tc_softmax_kernel(ann_real *w,
	__nv_bfloat16 *w_bf16,
	int T, int H, ann_real scale, const int *seqlens)
{
	__shared__ ann_real red[ATT_BLOCK];
	int bh = blockIdx.x;
	int b = bh / H;
	int i = blockIdx.y;
	int L = seqlens ? seqlens[b] : T;
	int j;
	ann_real mx;
	ann_real sum;
	size_t base = (size_t)bh * T * T + (size_t)i * T;
	ann_real *wr = w + base;
	__nv_bfloat16 *wbr = w_bf16 ? (w_bf16 + base) : NULL;

	if (i >= T)
		return;
	if (L < 1)
		L = 1;
	if (L > T)
		L = T;
	for (j = (int)threadIdx.x; j < T; j += (int)blockDim.x) {
		ann_real a = wr[j] * scale;

		if (!(i < L && j <= i && j < L))
			a = -1e30f;
		wr[j] = a;
	}
	__syncthreads();
	mx = -1e30f;
	for (j = (int)threadIdx.x; j < T; j += (int)blockDim.x)
		if (wr[j] > mx)
			mx = wr[j];
	mx = att_max(mx, red);
	sum = 0.0f;
	for (j = (int)threadIdx.x; j < T; j += (int)blockDim.x) {
		ann_real a = (i < L && j <= i && j < L) ? __expf(wr[j] - mx)
		    : 0.0f;

		wr[j] = a;
		sum += a;
	}
	sum = att_sum(sum, red);
	if (sum < 1e-20f)
		sum = 1.0f;
	for (j = (int)threadIdx.x; j < T; j += (int)blockDim.x) {
		ann_real p = wr[j] / sum;
		wr[j] = p;
		if (wbr)
			wbr[j] = __float2bfloat16(p);
	}
}

/* Softmax bwd: dP = P * (dP - sum_k P_k * dP_k) in place on dp; uses w. */
__global__ static void attn_tc_softmax_bwd_kernel(ann_real *dp,
	__nv_bfloat16 *dp_bf16,
	const ann_real *w, int T, int H, const int *seqlens)
{
	__shared__ ann_real red[ATT_BLOCK];
	int bh = blockIdx.x;
	int b = bh / H;
	int i = blockIdx.y;
	int L = seqlens ? seqlens[b] : T;
	int j;
	ann_real dot;
	size_t base = (size_t)bh * T * T + (size_t)i * T;
	ann_real *drow = dp + base;
	const ann_real *prow = w + base;
	__nv_bfloat16 *dbr = dp_bf16 ? (dp_bf16 + base) : NULL;

	if (L < 1)
		L = 1;
	if (L > T)
		L = T;
	if (i >= T)
		return;
	if (i >= L) {
		for (j = (int)threadIdx.x; j < T; j += (int)blockDim.x) {
			drow[j] = 0.0f;
			if (dbr)
				dbr[j] = __float2bfloat16(0.0f);
		}
		return;
	}
	dot = 0.0f;
	for (j = (int)threadIdx.x; j < T; j += (int)blockDim.x)
		if (j <= i && j < L)
			dot += prow[j] * drow[j];
	dot = att_sum(dot, red);
	for (j = (int)threadIdx.x; j < T; j += (int)blockDim.x) {
		ann_real g = (j <= i && j < L) ? prow[j] * (drow[j] - dot) : 0.0f;
		drow[j] = g;
		if (dbr)
			dbr[j] = __float2bfloat16(g);
	}
}

static int tc_ensure_pack(size_t n_elems)
{
	size_t bytes = n_elems * sizeof(ann_real);

	if (g_tc_pack_cap >= bytes)
		return 0;
	if (g_tc_q)
		cudaFree(g_tc_q);
	if (g_tc_k)
		cudaFree(g_tc_k);
	if (g_tc_v)
		cudaFree(g_tc_v);
	if (g_tc_o)
		cudaFree(g_tc_o);
	g_tc_q = g_tc_k = g_tc_v = g_tc_o = NULL;
	g_tc_pack_cap = 0;
	if (cudaMalloc((void **)&g_tc_q, bytes) != cudaSuccess
	    || cudaMalloc((void **)&g_tc_k, bytes) != cudaSuccess
	    || cudaMalloc((void **)&g_tc_v, bytes) != cudaSuccess
	    || cudaMalloc((void **)&g_tc_o, bytes) != cudaSuccess)
		return -1;
	g_tc_pack_cap = bytes;
	return 0;
}

static int tc_ensure_dp(size_t bytes)
{
	if (g_tc_score_cap >= bytes)
		return 0;
	if (g_tc_dp)
		cudaFree(g_tc_dp);
	g_tc_dp = NULL;
	g_tc_score_cap = 0;
	if (cudaMalloc((void **)&g_tc_dp, bytes) != cudaSuccess)
		return -1;
	g_tc_score_cap = bytes;
	return 0;
}

#ifdef ANN_CUBLAS
/* QK from already-packed BF16 twins in g_tc_half_{q,k}. */
static int tc_gemm_qk_half(ann_real *scores, int T, int dh, int batch)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	long long stride = (long long)T * dh;
	long long stride_s = (long long)T * T;

	if (cublas_ensure() != 0)
		return -1;
	return cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_T,
	    CUBLAS_OP_N, T, T, dh, &alpha, g_tc_half_k, CUDA_R_16BF,
	    dh, stride, g_tc_half_q, CUDA_R_16BF, dh, stride, &beta,
	    scores, CUDA_R_32F, T, stride_s, batch,
	    CUBLAS_COMPUTE_32F_FAST_16BF,
	    CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS
	    ? 0 : -1;
}

/* Prefer Tensor Core strided GemmEx; fall back to FP32 SgemmStridedBatched. */
static int tc_gemm_qk_batched(ann_real *scores, const ann_real *Q,
	const ann_real *K, int T, int dh, int batch)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	long long stride = (long long)T * dh;
	long long stride_s = (long long)T * T;
	size_t n;
	int threads = 256;

	if (cublas_ensure() != 0)
		return -1;
	if (g_compute_prec == GPU_PREC_FP32) {
		return cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T,
		    CUBLAS_OP_N, T, T, dh, &alpha, K, dh, stride, Q, dh, stride,
		    &beta, scores, T, stride_s, batch)
		    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
	}
	n = (size_t)batch * (size_t)T * (size_t)dh;
	if (ensure_half_buf(&g_tc_half_q, &g_tc_half_q_cap, n * 2) != 0
	    || ensure_half_buf(&g_tc_half_k, &g_tc_half_k_cap, n * 2) != 0)
		return -1;
	if (g_compute_prec == GPU_PREC_BF16) {
		int blocks = (int)((n + (size_t)threads * 4 - 1)
		    / ((size_t)threads * 4));

		if (blocks < 1)
			blocks = 1;
		f32_to_bf16_kernel<<<blocks, threads, 0, g_st>>>(
		    (__nv_bfloat16 *)g_tc_half_q, Q, (int)n);
		f32_to_bf16_kernel<<<blocks, threads, 0, g_st>>>(
		    (__nv_bfloat16 *)g_tc_half_k, K, (int)n);
		return tc_gemm_qk_half(scores, T, dh, batch);
	}
	return -1;
}

static int tc_gemm_bf16_strided(cublasOperation_t opa, cublasOperation_t opb,
	int m, int n, int k, float alpha_val, const void *A, int lda,
	long long strideA, const void *B, int ldb, long long strideB,
	ann_real *C, int ldc, long long strideC, int batch)
{
	const float alpha = alpha_val;
	const float beta = 0.0f;

	if (cublas_ensure() != 0)
		return -1;
	return cublasGemmStridedBatchedEx(g_cublas, opa, opb, m, n, k, &alpha,
	    A, CUDA_R_16BF, lda, strideA, B, CUDA_R_16BF, ldb, strideB, &beta,
	    C, CUDA_R_32F, ldc, strideC, batch, CUBLAS_COMPUTE_32F_FAST_16BF,
	    CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

static int tc_gemm_pv_batched(ann_real *O, const ann_real *W, const ann_real *V,
	int T, int dh, int batch)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	long long stride_v = (long long)T * dh;
	long long stride_w = (long long)T * T;

	if (cublas_ensure() != 0)
		return -1;
	/*
	** O = W * V. FP32 Sgemm for W (softmax probs): casting dense T×T W to
	** BF16 each step was bandwidth-heavier than the TC win at T=64.
	*/
	return cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
	    dh, T, T, &alpha, V, dh, stride_v, W, T, stride_w, &beta, O, dh,
	    stride_v, batch) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

static int tc_gemm_wt_do(ann_real *dV, const ann_real *W, const ann_real *dO,
	int T, int dh, int batch)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	long long stride_v = (long long)T * dh;
	long long stride_w = (long long)T * T;

	if (cublas_ensure() != 0)
		return -1;
	/* dV = W^T * dO */
	return cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
	    dh, T, T, &alpha, dO, dh, stride_v, W, T, stride_w, &beta, dV, dh,
	    stride_v, batch) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

static int tc_gemm_do_vt(ann_real *dP, const ann_real *dO, const ann_real *V,
	int T, int dh, int batch)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	long long stride_v = (long long)T * dh;
	long long stride_s = (long long)T * T;

	if (cublas_ensure() != 0)
		return -1;
	/* dP = dO * V^T  (T x T) */
	return cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
	    T, T, dh, &alpha, V, dh, stride_v, dO, dh, stride_v, &beta, dP, T,
	    stride_s, batch) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

static int tc_gemm_ds_k(ann_real *dQ, const ann_real *dS, const ann_real *K,
	int T, int dh, int batch, float scale)
{
	const float alpha = scale;
	const float beta = 0.0f;
	long long stride_v = (long long)T * dh;
	long long stride_s = (long long)T * T;

	if (cublas_ensure() != 0)
		return -1;
	/* dQ = dS * K */
	return cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
	    dh, T, T, &alpha, K, dh, stride_v, dS, T, stride_s, &beta, dQ, dh,
	    stride_v, batch) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

static int tc_gemm_dst_q(ann_real *dK, const ann_real *dS, const ann_real *Q,
	int T, int dh, int batch, float scale)
{
	const float alpha = scale;
	const float beta = 0.0f;
	long long stride_v = (long long)T * dh;
	long long stride_s = (long long)T * T;

	if (cublas_ensure() != 0)
		return -1;
	/* dK = dS^T * Q */
	return cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
	    dh, T, T, &alpha, Q, dh, stride_v, dS, T, stride_s, &beta, dK, dh,
	    stride_v, batch) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}
#endif /* ANN_CUBLAS */

static int tc_upload_seqlens(int B)
{
	if (!g_att_seqlens_host)
		return 0;
	if (!g_att_seqlens_dev || g_att_seqlens_B < B) {
		if (g_att_seqlens_dev)
			gpu_free(g_att_seqlens_dev);
		g_att_seqlens_dev = (int *)gpu_malloc((size_t)B * sizeof(int));
		g_att_seqlens_B = g_att_seqlens_dev ? B : 0;
	}
	if (!g_att_seqlens_dev)
		return -1;
	return gpu_memcpy_h2d(g_att_seqlens_dev, g_att_seqlens_host,
	    (size_t)B * sizeof(int));
}


/*
** ---- Fused causal attention on tensor cores (BF16 mma.sync, FP32 accumulate) --
**
** FlashAttention-2 structure, kernels in attn_fa2.cuh: one block per (b, h,
** 64-query tile), keys stream through shared memory in 64-wide tiles up to
** the causal diagonal; the score matrix is never written to HBM. Writes
** lse[b,h,i] = m_i + log l_i for the backward. Requirements: dh in
** {16,32,48,64}, T % 64 == 0, no seqlens. Q, K, V are read strided from
** the BF16 [B*T][3D] rows of the fused QKV GEMM; ANN_FA2=0 turns the whole
** set off and the TC strided GemmEx path runs instead.
*/
#include "attn_fa2.cuh"

static int fa2_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_FA2");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

/*
** Q, K, V of a layer as one BF16 row-major [B*T][3D] matrix written by the
** fused QKV GEMM (gpu_attn_qkv_fused, BF16 output) and read strided by
** the FA2 kernels: no FP32 q/k/v and no head packing. Keyed on the FP32 q
** mat's device buffer (stable per layer); live between a forward and its
** backward. 16 slots cover 12 layers; 24 MB per slot at B*T=4096, D=1024.
*/
#define QKV_BF16_SLOTS	16
typedef struct {
	const ann_real	*key;
	void		*buf;
	size_t		cap;
	int		N;
	int		D;
	int		live;
} qkv_bf16_slot;
static qkv_bf16_slot g_qkv_bf16[QKV_BF16_SLOTS];
static int g_qkv_bf16_rr;
/* One-shot hint from the attention layer: the next QKV projection feeds the FA2 kernels at (T, dh). */
static int g_qkv_hint_T;
static int g_qkv_hint_dh;

static qkv_bf16_slot *qkv_bf16_get(const ann_real *key, int N, int D, int create)
{
	int i;
	int free_slot = -1;
	qkv_bf16_slot *e;
	size_t need = (size_t)N * (size_t)D * 3 * 2;

	if (!key || N <= 0 || D <= 0)
		return NULL;
	for (i = 0; i < QKV_BF16_SLOTS; i++) {
		if (g_qkv_bf16[i].key == key && g_qkv_bf16[i].N == N
		    && g_qkv_bf16[i].D == D)
			return &g_qkv_bf16[i];
		if (free_slot < 0 && !g_qkv_bf16[i].key)
			free_slot = i;
	}
	if (!create)
		return NULL;
	if (free_slot < 0)
		free_slot = (g_qkv_bf16_rr++) % QKV_BF16_SLOTS;
	e = &g_qkv_bf16[free_slot];
	if (e->cap < need) {
		if (e->buf)
			cudaFree(e->buf);
		e->buf = NULL;
		e->cap = 0;
		if (cudaMalloc(&e->buf, need) != cudaSuccess) {
			e->buf = NULL;
			e->key = NULL;
			(void)cudaGetLastError();
			return NULL;
		}
		e->cap = need;
	}
	e->key = key;
	e->N = N;
	e->D = D;
	e->live = 0;
	return e;
}

static void qkv_bf16_forget(const void *p)
{
	int i;

	for (i = 0; i < QKV_BF16_SLOTS; i++)
		if (g_qkv_bf16[i].key == (const ann_real *)p) {
			g_qkv_bf16[i].key = NULL;
			g_qkv_bf16[i].live = 0;
		}
}

extern "C" void gpu_attn_qkv_bf16_hint(int T, int dh)
{
	g_qkv_hint_T = T;
	g_qkv_hint_dh = dh;
}

/* [Wq | Wk | Wv] as one BF16 [D][3D] matrix from the three weight twins; rebuilt after each optimizer step. */
__global__ static void qkv_wt_pack_kernel(__nv_bfloat16 *dst, const __nv_bfloat16 *wq,
	const __nv_bfloat16 *wk, const __nv_bfloat16 *wv, int D)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t n = (size_t)D * D * 3;
	int i, j;

	if (idx >= n)
		return;
	i = (int)(idx / (3 * D));
	j = (int)(idx % (3 * D));
	dst[idx] = j < D ? wq[(size_t)i * D + j]
	    : j < 2 * D ? wk[(size_t)i * D + j - D] : wv[(size_t)i * D + j - 2 * D];
}

/* One packed [Wq|Wk|Wv] per layer (keyed on its Wq twin), rebuilt when the weight epoch moves. */
#define QKV_WT_SLOTS	16
typedef struct {
	const void	*key;
	void		*buf;
	size_t		cap;
	long		epoch;
} qkv_wt_slot;
static qkv_wt_slot g_qkv_wt[QKV_WT_SLOTS];
static int g_qkv_wt_rr;

static void *qkv_wt_get(const void *hwq, const void *hwk, const void *hwv, int D)
{
	size_t n = (size_t)D * D * 3;
	qkv_wt_slot *e = NULL;
	int i;

	for (i = 0; i < QKV_WT_SLOTS; i++) {
		if (g_qkv_wt[i].key == hwq) {
			e = &g_qkv_wt[i];
			break;
		}
		if (!e && !g_qkv_wt[i].key)
			e = &g_qkv_wt[i];
	}
	if (!e)
		e = &g_qkv_wt[(g_qkv_wt_rr++) % QKV_WT_SLOTS];
	if (e->buf && e->key == hwq && e->epoch == (long)g_wt_epoch)
		return e->buf;
	if (ensure_half_buf(&e->buf, &e->cap, n * 2) != 0) {
		e->key = NULL;
		return NULL;
	}
	qkv_wt_pack_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_st>>>((__nv_bfloat16 *)e->buf,
	    (const __nv_bfloat16 *)hwq, (const __nv_bfloat16 *)hwk,
	    (const __nv_bfloat16 *)hwv, D);
	if (cudaGetLastError() != cudaSuccess) {
		e->key = NULL;
		return NULL;
	}
	e->key = hwq;
	e->epoch = (long)g_wt_epoch;
	return e->buf;
}

/*
** 1 when the FA2 attention will take this shape in BF16 mode. ANN_FA2=0
** answers 0, so the callers allocate the score buffer and run the TC path.
*/
extern "C" int gpu_attn_fa_usable(int T, int dh)
{
	return gpu_available() && g_compute_prec == GPU_PREC_BF16
	    && fa2_on() && fa2_ok(T, dh);
}

/*
** Fused tensor-core causal attention forward (BF16 mode, FA2 kernels). It
** reads the BF16 [B*T][3D] rows the fused QKV GEMM left in the layer's qkv
** slot (gpu_attn_qkv_fused). Writes concat and lse; -1 = no such slot (QKV
** GEMM failed, graph capture, ANN_FA2=0) and the caller uses another path.
*/
/* ANN_FA2_O16=0: concat's BF16 twin is converted after the FA2 forward instead of written by it. */
static int fa2_o16_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_FA2_O16");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

extern "C" int gpu_attn_fa_fwd(mat *concat, const mat *q, const mat *k,
	const mat *v, int B, int T, int H, int dh, mat *lse)
{
	OpScope _op(OP_ATTN_FWD);
	ann_real *dq, *dk, *dv, *dc;
	int oq, ok, ov, oc, rc = -1;
	act_bf16_slot *o16 = NULL;

	if (!concat || !q || !k || !v || !lse || !lse->d_data
	    || !att_args_ok(B, T, H, dh) || !fa2_ok(T, dh) || !fa2_on()
	    || g_att_seqlens_host || g_compute_prec != GPU_PREC_BF16)
		return -1;
	if (q->rows != B * T || q->cols != H * dh || lse->rows != B * H * T)
		return -1;
	pthread_mutex_lock(&g_att_mu);
	oq = ok = ov = oc = 0;
	if (ensure_dev(q, &dq, &oq) != 0 || ensure_dev_kv(k, &dk, &ok) != 0
	    || ensure_dev_kv(v, &dv, &ov) != 0
	    || ensure_dev_out(concat, &dc, &oc) != 0)
		goto fa_out;
	{
		qkv_bf16_slot *qs = oq ? NULL : qkv_bf16_get(dq, B * T, H * dh, 0);
		static int reported;
		const __nv_bfloat16 *bq;
		const int Dm = H * dh;

		if (!qs || !qs->live)
			goto fa_out;
		bq = (const __nv_bfloat16 *)qs->buf;
		/* B11: the kernel writes concat's BF16 twin itself (ANN_FA2_O16=0: converted after). */
		if (!oc && fa2_o16_on())
			o16 = act_bf16_slot_for(dc, B * T * H * dh);
		{ OpNest _n(OP_ATT_GEMM_ID); if (fa2_fwd_launch_lay(dc, lse->d_data, bq, bq + Dm, bq + 2 * Dm, B, T, H, dh,
		    (float)(1.0 / sqrt((double)dh)), fa2_lay_rows(T, 3 * Dm, dh),
		    o16 ? (__nv_bfloat16 *)o16->half : NULL) != 0)
			goto fa_out; }
		if (!reported++)
			fprintf(stderr, "[attn] FA2 mma.sync attention on the fused BF16 QKV rows\n");
	}
	if (launch_done() == 0 && finish_out(concat, dc, oc) == 0) {
		lse->on_device = MAT_DEV_DEVICE;
		rc = 0;
		/* BF16 twin of concat for the Wo GEMM now and its wgrad later. */
		if (o16)
			o16->live = 1;	/* written by the FA2 forward */
		else if (!oc)
			act_bf16_publish(dc, B * T * H * dh);
	}
fa_out:
	if (oq && dq)
		gpu_free(dq);
	if (ok && dk)
		gpu_free(dk);
	if (ov && dv)
		gpu_free(dv);
	if (oc && dc)
		gpu_free(dc);
	pthread_mutex_unlock(&g_att_mu);
	return rc;
}

/* ANN_FA2_TWIN_ONLY=0: the FA2 backward stores FP32 dK / dV next to their BF16 twins. */
static int fa2_twin_only_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_FA2_TWIN_ONLY");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

/*
** ANN_FA2_DVEC=0: compute dvec with attn_fa_dvec16_kernel before the FA2
** backward instead of inside its query-tile prologue (B5).
*/
static int fa2_dvec_fused_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_FA2_DVEC");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

/*
** ANN_QKV_BWD=0: the FA2 backward writes two dense dK / dV twins and the
** QKV projection backward runs three dgrad and three wgrad GEMMs, instead
** of the merged [dQ|dK|dV] rows and gpu_attn_qkv_bwd (B3).
*/
static int qkv_bwd_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_QKV_BWD");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

/*
** ANN_QKV_WBATCH=0: with the merged rows, the Wq / Wk / Wv wgrads run as
** three GEMMs on their column blocks instead of one strided-batched GEMM.
*/
static int qkv_wbatch_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_QKV_WBATCH");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

/*
** FP32 [rows][cols] to BF16 rows of stride ld (dQ into the merged rows);
** same RN rounding as f32_to_bf16_kernel. cols % 4 == 0, ld even.
*/
__global__ static void f32_to_bf16_ld_kernel(__nv_bfloat16 *dst, int ld,
	const float *src, int rows, int cols)
{
	size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
	size_t r;
	size_t c;
	float4 v;
	__nv_bfloat162 *d;

	if (i >= (size_t)rows * cols)
		return;
	r = i / cols;
	c = i % cols;
	v = *(const float4 *)(src + i);
	d = (__nv_bfloat162 *)(dst + r * ld + c);
	d[0] = __floats2bfloat162_rn(v.x, v.y);
	d[1] = __floats2bfloat162_rn(v.z, v.w);
}

/* D_i = sum_c dO[i,c] O[i,c] per (b,h,i), one thread per row. */
__global__ static void attn_fa_dvec_kernel(float *dvec, const ann_real *dO,
	const ann_real *O, int nrows, int T, int H, int dh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int bh = idx / T;
	int i = idx % T;
	int b = bh / H;
	int h = bh % H;
	int D = H * dh;
	size_t off;
	float acc = 0.0f;
	int c;

	if (idx >= nrows)
		return;
	off = ((size_t)b * T + i) * D + h * dh;
	for (c = 0; c < dh; c++)
		acc += dO[off + c] * O[off + c];
	dvec[idx] = acc;
}

/* Same from the BF16 twin of dO (model layout). */
__global__ static void attn_fa_dvec16_kernel(float *dvec, const __nv_bfloat16 *dO,
	const ann_real *O, int nrows, int T, int H, int dh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int bh = idx / T;
	int i = idx % T;
	int b = bh / H;
	int h = bh % H;
	int D = H * dh;
	size_t off;
	float acc = 0.0f;
	int c;

	if (idx >= nrows)
		return;
	off = ((size_t)b * T + i) * D + h * dh;
	for (c = 0; c < dh; c++)
		acc += __bfloat162float(dO[off + c]) * O[off + c];
	dvec[idx] = acc;
}

static float *g_fa_dvec;
static size_t g_fa_dvec_cap;

/*
** Fused tensor-core causal attention backward (BF16 mode, FA2 kernels);
** dq/dk/dv written. Reads the layer's BF16 QKV rows (consumed once); -1
** when they are gone, which the forward that chose this path rules out.
*/
extern "C" int gpu_attn_fa_bwd(mat *dq, mat *dk, mat *dv, const mat *dconcat,
	const mat *concat, const mat *q, const mat *k, const mat *v,
	const mat *lse, int B, int T, int H, int dh)
{
	OpScope _op(OP_ATTN_BWD);
	ann_real *ptr[7];
	int o[7];
	int i, rc = -1;
	size_t nrows;

	if (!dq || !dk || !dv || !dconcat || !concat || !q || !k || !v || !lse
	    || !lse->d_data || !att_args_ok(B, T, H, dh) || !fa2_ok(T, dh)
	    || !fa2_on() || g_att_seqlens_host || g_compute_prec != GPU_PREC_BF16)
		return -1;
	nrows = (size_t)B * H * T;
	if (g_fa_dvec_cap < nrows) {
		if (g_fa_dvec)
			cudaFree(g_fa_dvec);
		g_fa_dvec = NULL;
		g_fa_dvec_cap = 0;
		if (cudaMalloc((void **)&g_fa_dvec, nrows * sizeof(float))
		    != cudaSuccess) {
			g_fa_dvec = NULL;
			return -1;
		}
		g_fa_dvec_cap = nrows;
	}
	pthread_mutex_lock(&g_att_mu);
	for (i = 0; i < 7; i++) {
		o[i] = 0;
		ptr[i] = NULL;
	}
	if (ensure_dev(dconcat, &ptr[0], &o[0]) != 0
	    || ensure_dev(concat, &ptr[1], &o[1]) != 0
	    || ensure_dev(q, &ptr[2], &o[2]) != 0
	    || ensure_dev_kv(k, &ptr[3], &o[3]) != 0
	    || ensure_dev_kv(v, &ptr[4], &o[4]) != 0
	    || ensure_dev_out(dq, &ptr[5], &o[5]) != 0
	    || ensure_dev_out(dk, &ptr[6], &o[6]) != 0)
		goto fa_bwd_out;
	{
		ann_real *a_dv;
		int odv = 0;
		const __nv_bfloat16 *dO16 = NULL;
		size_t ndo = (size_t)B * T * H * dh;
		qkv_bf16_slot *qs = o[2] ? NULL : qkv_bf16_get(ptr[2], B * T, H * dh, 0);
		const __nv_bfloat16 *bq;
		const int Dm = H * dh;
		void *dk16 = NULL, *dv16 = NULL;
		const float *fused_of = NULL;
		int skip32;
		int merged = 0;

		if (!qs || !qs->live)
			goto fa_bwd_out;
		if (ensure_dev_out(dv, &a_dv, &odv) != 0)
			goto fa_bwd_out;
		if (cudaMemsetAsync(ptr[5], 0, ndo * sizeof(ann_real), g_st)
		    != cudaSuccess)
			goto fa_bwd_done;
		/* The Wo dgrad may have left dO only as a BF16 twin (gpu_mat_mul_bt_twin). */
		if (!o[0]) {
			pthread_mutex_lock(&g_gemm_mu);
			{
				int ld16 = H * dh;	/* dense twin: row stride D */

				dO16 = (const __nv_bfloat16 *)grad_twin_take(ptr[0], ndo, H * dh, &ld16);
				if (dO16 && ld16 != H * dh)
					dO16 = NULL;	/* a strided twin is not what the kernel expects */
			}
			pthread_mutex_unlock(&g_gemm_mu);
		}
		bq = (const __nv_bfloat16 *)qs->buf;
		qs->live = 0;
		/*
		** B3: dK / dV go straight into the merged [dQ|dK|dV] rows (dQ is
		** converted into its block after the kernel) for gpu_attn_qkv_bwd.
		** The FP8 paths quantise from the FP32 gradients and the twin check
		** compares twins with them: those keep the separate twins.
		*/
		g_qkv_d16.live = 0;
		g_qkv_d16.key = NULL;
		if (!o[5] && !o[6] && !odv && twin_emit_on()
		    && qkv_bwd_on() && !g_fp8_fwd && !g_fp8_dgrad && !twin_check_on()
		    && (Dm & 3) == 0) {
			pthread_mutex_lock(&g_gemm_mu);
			if (ensure_half_buf(&g_qkv_d16.buf, &g_qkv_d16.cap, 3 * ndo * 2) == 0) {
				dk16 = (void *)((__nv_bfloat16 *)g_qkv_d16.buf + Dm);
				dv16 = (void *)((__nv_bfloat16 *)g_qkv_d16.buf + 2 * Dm);
				merged = 1;
				/* nothing may serve an older twin of these buffers now */
				grad_twin_forget(ptr[5]);
				grad_twin_forget(ptr[6]);
				grad_twin_forget(a_dv);
			}
			pthread_mutex_unlock(&g_gemm_mu);
		}
		/* dK, dV also as BF16 twins: their wgrad and dgrad GEMMs then convert nothing. */
		if (!merged && !o[6] && !odv && twin_emit_on()) {
			pthread_mutex_lock(&g_gemm_mu);
			dk16 = grad_twin_reserve(ptr[6], ndo);
			dv16 = dk16 ? grad_twin_reserve(a_dv, ndo) : NULL;
			if (!dv16 && dk16)
				grad_twin_forget(ptr[6]);
			pthread_mutex_unlock(&g_gemm_mu);
			if (!dv16)
				dk16 = NULL;
		}
		if (dO16 && fa2_dvec_fused_on()) {
			/* dO's BF16 twin; the backward kernel computes dvec itself from it and O. */
			fused_of = ptr[1];
		} else if (dO16) {
			/* dO's BF16 twin (written by the Wo dgrad GEMM), read strided; dvec from it. */
			{ OpNest _n(OP_ATT_SMX_ID); attn_fa_dvec16_kernel<<<(unsigned)((nrows + 255) / 256), 256, 0, g_st>>>(
			    g_fa_dvec, dO16, ptr[1], (int)nrows, T, H, dh); }
		} else {
			/* FP32 dO only: BF16 rows in scratch (same rounding as the twin), dvec from the FP32. */
			int threads = 256;
			int n = (int)ndo;
			int blocks = (n + threads * 4 - 1) / (threads * 4);

			if (ensure_half_buf(&g_tc_half_do, &g_tc_half_do_cap, ndo * 2) != 0) {
				if (dk16) {
					grad_twin_forget(ptr[6]);
					grad_twin_forget(a_dv);
				}
				goto fa_bwd_done;
			}
			{ OpNest _n(OP_ATT_PACK_ID); f32_to_bf16_kernel<<<blocks, threads, 0, g_st>>>(
			    (__nv_bfloat16 *)g_tc_half_do, ptr[0], n); }
			{ OpNest _n(OP_ATT_SMX_ID); attn_fa_dvec_kernel<<<(unsigned)((nrows + 255) / 256), 256, 0, g_st>>>(
			    g_fa_dvec, ptr[0], ptr[1], (int)nrows, T, H, dh); }
		}
		/*
		** B9: with both twins written and no FP8 GEMM (which quantises from
		** FP32), the Wk / Wv wgrad (grad_twin_peek) and dgrad (grad_twin_take)
		** read only the twins: skip the 2 x 16 MB FP32 stores. ANN_TWIN_CHECK=1
		** compares twins with their FP32, so it keeps them.
		*/
		skip32 = dk16 && fa2_twin_only_on() && !g_fp8_fwd
		    && !g_fp8_dgrad && !twin_check_on();
		{ OpNest _n(OP_ATT_GEMM_ID); if (fa2_bwd_launch_lay16(ptr[5], skip32 ? NULL : ptr[6], skip32 ? NULL : a_dv, bq, bq + Dm, bq + 2 * Dm,
		    dO16 ? dO16 : (const __nv_bfloat16 *)g_tc_half_do, lse->d_data, g_fa_dvec,
		    B, T, H, dh, (float)(1.0 / sqrt((double)dh)),
		    fa2_lay_rows(T, 3 * Dm, dh), fa2_lay_rows(T, Dm, dh),
		    (__nv_bfloat16 *)dk16, (__nv_bfloat16 *)dv16, fused_of,
		    merged ? 3 * Dm : 0) != 0) {
			if (dk16) {
				grad_twin_forget(ptr[6]);
				grad_twin_forget(a_dv);
			}
			goto fa_bwd_done;
		} }
		if (merged) {
			/* dQ (atomics, FP32) into its block of the merged rows. */
			size_t n4 = ndo / 4;

			{ OpNest _n(OP_ATT_PACK_ID); f32_to_bf16_ld_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, g_st>>>(
			    (__nv_bfloat16 *)g_qkv_d16.buf, 3 * Dm, ptr[5], B * T, Dm); }
		}
		if (launch_done() == 0 && finish_out(dq, ptr[5], o[5]) == 0
		    && finish_out(dk, ptr[6], o[6]) == 0
		    && finish_out(dv, a_dv, odv) == 0) {
			rc = 0;
			if (merged) {
				g_qkv_d16.key = ptr[5];
				g_qkv_d16.n = ndo;
				g_qkv_d16.live = 1;
			}
		}
fa_bwd_done:
		if (odv && a_dv)
			gpu_free(a_dv);
	}
fa_bwd_out:
	for (i = 0; i < 7; i++)
		if (o[i] && ptr[i])
			gpu_free(ptr[i]);
	pthread_mutex_unlock(&g_att_mu);
	return rc;
}

/*
** B3: the Q/K/V projection backward from the merged BF16 [dQ|dK|dV] rows
** the FA2 backward left for dq:
**   dx = [dQ|dK|dV] [Wq|Wk|Wv]^T		one GEMM, K = 3 D (packed weights)
**   dWq, dWk, dWv += x_in^T [dQ|dK|dV]	one strided-batched GEMM
** instead of three dgrads (two of them accumulating into dx) and three
** wgrads. Same BF16 operands as the separate GEMMs; only the FP32
** summation order changes. Returns 1 when no merged rows are pending for
** dq (the caller runs the separate GEMMs), 0 done, -1 error: the merged
** rows were then the only copy of dK and dV.
*/
extern "C" int gpu_attn_qkv_bwd(mat *dx, mat *dwq, mat *dwk, mat *dwv,
	const mat *x_in, const mat *dq, const mat *wq, const mat *wk,
	const mat *wv)
{
#ifdef ANN_CUBLAS
	OpScope _op(OP_GEMM_DGRAD);
	const mat *w3[3];
	mat *dw3[3];
	ann_real *pw[3];
	ann_real *pdw[3];
	void *hw[3];
	int ow[3];
	int odw[3];
	ann_real *pdx = NULL;
	ann_real *px = NULL;
	int odx = 0;
	int ox = 0;
	const void *hx;
	const __nv_bfloat16 *g16;
	void *wqkv = NULL;
	const float one = 1.0f;
	const float zero = 0.0f;
	size_t nx;
	int N, D, i, rc = -1;

	if (!dq || !dq->d_data || !g_qkv_d16.live || g_qkv_d16.key != dq->d_data)
		return 1;
	g_qkv_d16.live = 0;	/* one shot, whatever happens below */
	g_qkv_d16.key = NULL;
	N = dq->rows;
	D = dq->cols;
	if (!dx || !dwq || !dwk || !dwv || !x_in || !wq || !wk || !wv
	    || g_qkv_d16.n != (size_t)N * (size_t)D || x_in->rows != N
	    || x_in->cols != D || dx->rows != N || dx->cols != D
	    || cublas_ensure() != 0)
		return -1;
	w3[0] = wq;
	w3[1] = wk;
	w3[2] = wv;
	dw3[0] = dwq;
	dw3[1] = dwk;
	dw3[2] = dwv;
	for (i = 0; i < 3; i++) {
		pw[i] = pdw[i] = NULL;
		ow[i] = odw[i] = 0;
		hw[i] = NULL;
		if (w3[i]->rows != D || w3[i]->cols != D || dw3[i]->rows != D
		    || dw3[i]->cols != D)
			return -1;
	}
	g16 = (const __nv_bfloat16 *)g_qkv_d16.buf;
	nx = (size_t)N * (size_t)D;
	if (ensure_dev_out(dx, &pdx, &odx) != 0 || ensure_dev(x_in, &px, &ox) != 0)
		goto qb_out;
	for (i = 0; i < 3; i++)
		if (ensure_dev(w3[i], &pw[i], &ow[i]) != 0
		    || ensure_dev(dw3[i], &pdw[i], &odw[i]) != 0)
			goto qb_out;
	pthread_mutex_lock(&g_gemm_mu);
	/* x_in's BF16 twin (the LayerNorm output may exist only as BF16). */
	hx = act_bf16_get(px, (int)nx);
	if (!hx && ensure_half_buf(&g_half_A, &g_half_A_cap, nx * 2) == 0
	    && half_wt_convert(g_half_A, px, (int)nx, 1) == 0)
		hx = g_half_A;
	for (i = 0; i < 3; i++)
		hw[i] = half_wt_get(pw[i], (size_t)D * (size_t)D, 1);
	if (hw[0] && hw[1] && hw[2])
		wqkv = qkv_wt_get(hw[0], hw[1], hw[2], D);
	rc = hx ? 0 : -1;
	/* dgrad: dx = G [Wq|Wk|Wv]^T, G and the packed weights both of row stride 3 D. */
	if (rc == 0 && wqkv)
		rc = cublas_gemm_ex_bf16(CUBLAS_OP_T, CUBLAS_OP_N, D, N, 3 * D,
		    &one, wqkv, 3 * D, g16, 3 * D, &zero, pdx, D);
	else if (rc == 0) {
		/* no packed weights (weights outside the Adam flat): per block, accumulating */
		for (i = 0; i < 3 && rc == 0; i++) {
			const void *hwi = hw[i];

			if (!hwi) {
				if (ensure_half_buf(&g_half_B, &g_half_B_cap,
				    (size_t)D * D * 2) != 0
				    || half_wt_convert(g_half_B, pw[i], D * D, 1) != 0) {
					rc = -1;
					break;
				}
				hwi = g_half_B;
			}
			rc = cublas_gemm_ex_bf16(CUBLAS_OP_T, CUBLAS_OP_N, D, N, D,
			    &one, hwi, D, g16 + (size_t)i * D, 3 * D,
			    i ? &one : &zero, pdx, D);
		}
	}
	/* wgrad: dW_i += x_in^T G_i; one batch when dWq, dWk, dWv are consecutive (Adam flat). */
	if (rc == 0) {
		OpNest _n(OP_GEMM_WGRAD);

		if (qkv_wbatch_on() && pdw[1] == pdw[0] + (size_t)D * D
		    && pdw[2] == pdw[1] + (size_t)D * D)
			rc = cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_N,
			    CUBLAS_OP_T, D, D, N, &one, g16, CUDA_R_16BF, 3 * D,
			    (long long)D, hx, CUDA_R_16BF, D, 0LL, &one, pdw[0],
			    CUDA_R_32F, D, (long long)D * D, 3,
			    CUBLAS_COMPUTE_32F_FAST_16BF,
			    CUBLAS_GEMM_DEFAULT_TENSOR_OP)
			    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
		else
			for (i = 0; i < 3 && rc == 0; i++)
				rc = cublas_gemm_ex_bf16(CUBLAS_OP_N, CUBLAS_OP_T,
				    D, D, N, &one, g16 + (size_t)i * D, 3 * D, hx, D,
				    &one, pdw[i], D);
	}
	pthread_mutex_unlock(&g_gemm_mu);
	if (rc == 0 && (launch_done() != 0 || finish_out(dx, pdx, odx) != 0))
		rc = -1;
	for (i = 0; i < 3 && rc == 0; i++)
		if (finish_out(dw3[i], pdw[i], odw[i]) != 0)
			rc = -1;
qb_out:
	if (ox && px)
		gpu_free(px);
	for (i = 0; i < 3; i++) {
		if (ow[i] && pw[i])
			gpu_free(pw[i]);
		if (odw[i] && pdw[i])
			gpu_free(pdw[i]);
	}
	if (rc != 0)
		fprintf(stderr, "[attn] merged QKV projection backward failed\n");
	return rc;
#else
	(void)dx; (void)dwq; (void)dwk; (void)dwv; (void)x_in; (void)dq;
	(void)wq; (void)wk; (void)wv;
	return 1;
#endif
}

extern "C" int gpu_attn_tc_fwd(mat *concat, mat *w, const mat *q, const mat *k,
	const mat *v, int B, int T, int H, int dh)
{
	OpScope _op(OP_ATTN_FWD);
#ifdef ANN_CUBLAS
	ann_real *dq;
	ann_real *dk;
	ann_real *dv;
	ann_real *dc;
	ann_real *dw;
	int oq;
	int ok;
	int ov;
	int oc;
	int rc;
	int batch;
	int npack;
	int threads = 256;
	dim3 sgrid;
	ann_real scale;

	if (!concat || !w || !q || !k || !v || !att_args_ok(B, T, H, dh))
		return -1;
	if (q->rows != B * T || q->cols != H * dh || w->rows != B * H * T
	    || w->cols != T)
		return -1;
	/* Act-checkpoint may have dropped d_data; repin score buffer. */
	if (!w->d_data) {
		mat_try_pin(w);
		if (!w->d_data)
			return -1;
	}
	batch = B * H;
	npack = batch * T * dh;
	pthread_mutex_lock(&g_att_mu);
	rc = -1;
	oq = ok = ov = oc = 0;
	if (tc_ensure_pack((size_t)npack) != 0)
		goto tc_fwd_out;
	if (ensure_dev(q, &dq, &oq) != 0 || ensure_dev(k, &dk, &ok) != 0
	    || ensure_dev(v, &dv, &ov) != 0
	    || ensure_dev_out(concat, &dc, &oc) != 0)
		goto tc_fwd_out;
	dw = w->d_data;
	if (tc_upload_seqlens(B) != 0)
		goto tc_fwd_out;
	/*
	** BF16: pack Q/K straight into half twins (skip FP32 pack + convert).
	** V stays FP32 for the PV Sgemm.
	*/
	if (g_compute_prec == GPU_PREC_BF16) {
		/*
		** All attention products on tensor cores: Q, K, V packed as
		** BF16, P written as BF16 by the softmax, O = P V in BF16 with
		** FP32 accumulation (the FP32 Sgemm products were 29% of the
		** microbatch for 1% of its FLOPs).
		*/
		if (ensure_half_buf(&g_tc_half_q, &g_tc_half_q_cap,
		    (size_t)npack * 2) != 0
		    || ensure_half_buf(&g_tc_half_k, &g_tc_half_k_cap,
		    (size_t)npack * 2) != 0
		    || ensure_half_buf(&g_tc_half_v, &g_tc_half_v_cap,
		    (size_t)npack * 2) != 0
		    || ensure_half_buf(&g_tc_half_p, &g_tc_half_p_cap,
		    (size_t)batch * T * T * 2) != 0)
			goto tc_fwd_out;
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_bf16_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>((__nv_bfloat16 *)g_tc_half_q, dq, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_bf16_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>((__nv_bfloat16 *)g_tc_half_k, dk, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_bf16_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>((__nv_bfloat16 *)g_tc_half_v, dv, B, T, H, dh); }
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_qk_half(dw, T, dh, batch) != 0)
			goto tc_fwd_out;
	} else
	{
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>(g_tc_q, dq, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>(g_tc_k, dk, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>(g_tc_v, dv, B, T, H, dh); }
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_qk_batched(dw, g_tc_q, g_tc_k, T, dh, batch) != 0)
			goto tc_fwd_out;
	}
	scale = (ann_real)(1.0 / sqrt((double)dh));
	sgrid = dim3(batch, T, 1);
	{
		int nth = 32;

		while (nth < T && nth < ATT_BLOCK)
			nth <<= 1;
		{ OpNest _n(OP_ATT_SMX_ID); attn_tc_softmax_kernel<<<sgrid, nth, 0, g_st>>>(dw,
		    g_compute_prec == GPU_PREC_BF16
		    ? (__nv_bfloat16 *)g_tc_half_p : NULL, T, H, scale,
		    g_att_seqlens_host ? g_att_seqlens_dev : NULL); }
	}
	if (g_compute_prec == GPU_PREC_BF16) {
		/* O = P V: col-major O^T[dh,T] = V^T[dh,T] P^T[T,T]. */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_bf16_strided(CUBLAS_OP_N,
		    CUBLAS_OP_N, dh, T, T, 1.0f, g_tc_half_v, dh,
		    (long long)T * dh, g_tc_half_p, T, (long long)T * T,
		    g_tc_o, dh, (long long)T * dh, batch) != 0)
			goto tc_fwd_out;
	} else
	if (OpNest(OP_ATT_GEMM_ID), tc_gemm_pv_batched(g_tc_o, dw, g_tc_v, T, dh, batch) != 0)
		goto tc_fwd_out;
	{ OpNest _n(OP_ATT_PACK_ID); attn_unpack_heads_kernel<<<(npack + threads - 1) / threads, threads, 0, g_st>>>(
	    dc, g_tc_o, B, T, H, dh); }
	if (launch_done() == 0 && finish_out(concat, dc, oc) == 0) {
		w->on_device = MAT_DEV_DEVICE;
		rc = 0;
	}
tc_fwd_out:
	if (oq && dq)
		gpu_free(dq);
	if (ok && dk)
		gpu_free(dk);
	if (ov && dv)
		gpu_free(dv);
	if (oc && dc)
		gpu_free(dc);
	pthread_mutex_unlock(&g_att_mu);
	return rc;
#else
	(void)concat;
	(void)w;
	(void)q;
	(void)k;
	(void)v;
	(void)B;
	(void)T;
	(void)H;
	(void)dh;
	return -1;
#endif
}

extern "C" int gpu_attn_tc_bwd(mat *dq, mat *dk, mat *dv, const mat *dconcat,
	mat *w, const mat *q, const mat *k, const mat *v, int B, int T, int H,
	int dh)
{
	OpScope _op(OP_ATTN_BWD);
#ifdef ANN_CUBLAS
	ann_real *a_q;
	ann_real *a_k;
	ann_real *a_v;
	ann_real *a_dc;
	ann_real *a_dq;
	ann_real *a_dk;
	ann_real *a_dv;
	ann_real *dw;
	int o[7];
	ann_real *ptr[7];
	int i;
	int rc;
	int batch;
	int npack;
	int threads = 256;
	dim3 sgrid;
	float scale;
	size_t score_bytes;

	if (!dq || !dk || !dv || !dconcat || !w || !q || !k || !v
	    || !att_args_ok(B, T, H, dh))
		return -1;
	if (!w->d_data) {
		mat_try_pin(w);
		if (!w->d_data)
			return -1;
	}
	batch = B * H;
	npack = batch * T * dh;
	score_bytes = (size_t)batch * (size_t)T * (size_t)T * sizeof(ann_real);
	pthread_mutex_lock(&g_att_mu);
	rc = -1;
	for (i = 0; i < 7; i++) {
		o[i] = 0;
		ptr[i] = NULL;
	}
	if (tc_ensure_pack((size_t)npack) != 0 || tc_ensure_dp(score_bytes) != 0)
		goto tc_bwd_out;
	if (ensure_dev(dconcat, &ptr[0], &o[0]) != 0
	    || ensure_dev(q, &ptr[1], &o[1]) != 0
	    || ensure_dev(k, &ptr[2], &o[2]) != 0
	    || ensure_dev(v, &ptr[3], &o[3]) != 0
	    || ensure_dev_out(dq, &ptr[4], &o[4]) != 0
	    || ensure_dev_out(dk, &ptr[5], &o[5]) != 0
	    || ensure_dev_out(dv, &ptr[6], &o[6]) != 0)
		goto tc_bwd_out;
	a_dc = ptr[0];
	a_q = ptr[1];
	a_k = ptr[2];
	a_v = ptr[3];
	a_dq = ptr[4];
	a_dk = ptr[5];
	a_dv = ptr[6];
	dw = w->d_data;
	if (tc_upload_seqlens(B) != 0)
		goto tc_bwd_out;
	scale = (float)(1.0 / sqrt((double)dh));
	sgrid = dim3(batch, T, 1);
	if (g_compute_prec == GPU_PREC_BF16) {
		int nth = 32;
		long long sv = (long long)T * dh;
		long long ss = (long long)T * T;

		while (nth < T && nth < ATT_BLOCK)
			nth <<= 1;
		if (ensure_half_buf(&g_tc_half_q, &g_tc_half_q_cap,
		    (size_t)npack * 2) != 0
		    || ensure_half_buf(&g_tc_half_k, &g_tc_half_k_cap,
		    (size_t)npack * 2) != 0
		    || ensure_half_buf(&g_tc_half_v, &g_tc_half_v_cap,
		    (size_t)npack * 2) != 0
		    || ensure_half_buf(&g_tc_half_do, &g_tc_half_do_cap,
		    (size_t)npack * 2) != 0
		    || ensure_half_buf(&g_tc_half_p, &g_tc_half_p_cap,
		    (size_t)batch * ss * 2) != 0
		    || ensure_half_buf(&g_tc_half_ds, &g_tc_half_ds_cap,
		    (size_t)batch * ss * 2) != 0)
			goto tc_bwd_out;
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_bf16_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>((__nv_bfloat16 *)g_tc_half_q, a_q, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_bf16_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>((__nv_bfloat16 *)g_tc_half_k, a_k, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_bf16_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>((__nv_bfloat16 *)g_tc_half_v, a_v, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_bf16_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>((__nv_bfloat16 *)g_tc_half_do, a_dc, B, T, H, dh); }
		/* S = Q K^T (recomputed), P = softmax(S) in FP32 + BF16. */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_qk_half(dw, T, dh, batch) != 0)
			goto tc_bwd_out;
		{ OpNest _n(OP_ATT_SMX_ID); attn_tc_softmax_kernel<<<sgrid, nth, 0, g_st>>>(dw,
		    (__nv_bfloat16 *)g_tc_half_p, T, H, (ann_real)scale,
		    g_att_seqlens_host ? g_att_seqlens_dev : NULL); }
		/* dP = dO V^T: col-major dP^T[T,T] = V[dh,T]^T dO^T[dh,T]. */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_bf16_strided(CUBLAS_OP_T,
		    CUBLAS_OP_N, T, T, dh, 1.0f, g_tc_half_v, dh, sv,
		    g_tc_half_do, dh, sv, g_tc_dp, T, ss, batch) != 0)
			goto tc_bwd_out;
		/* dV = P^T dO: col-major dV^T[dh,T] = dO^T[dh,T] P[T,T]. */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_bf16_strided(CUBLAS_OP_N,
		    CUBLAS_OP_T, dh, T, T, 1.0f, g_tc_half_do, dh, sv,
		    g_tc_half_p, T, ss, g_tc_v, dh, sv, batch) != 0)
			goto tc_bwd_out;
		/* dS = P (dP - rowsum(P dP)) in FP32 + BF16. */
		{ OpNest _n(OP_ATT_SMX_ID); attn_tc_softmax_bwd_kernel<<<sgrid, nth, 0, g_st>>>(g_tc_dp,
		    (__nv_bfloat16 *)g_tc_half_ds, dw, T, H,
		    g_att_seqlens_host ? g_att_seqlens_dev : NULL); }
		/* dQ = scale dS K: col-major dQ^T[dh,T] = K^T[dh,T] dS^T[T,T]. */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_bf16_strided(CUBLAS_OP_N,
		    CUBLAS_OP_N, dh, T, T, scale, g_tc_half_k, dh, sv,
		    g_tc_half_ds, T, ss, g_tc_o, dh, sv, batch) != 0)
			goto tc_bwd_out;
		/* dK = scale dS^T Q: col-major dK^T[dh,T] = Q^T[dh,T] dS[T,T]. */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_bf16_strided(CUBLAS_OP_N,
		    CUBLAS_OP_T, dh, T, T, scale, g_tc_half_q, dh, sv,
		    g_tc_half_ds, T, ss, g_tc_k, dh, sv, batch) != 0)
			goto tc_bwd_out;
	} else
	{
		int nth = 32;

		/* Pack Q/K/V and dO. */
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>(g_tc_q, a_q, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>(g_tc_k, a_k, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>(g_tc_v, a_v, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads, 0, g_st>>>(g_tc_o, a_dc, B, T, H, dh); }
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_qk_batched(dw, g_tc_q, g_tc_k, T, dh, batch) != 0)
			goto tc_bwd_out;
		while (nth < T && nth < ATT_BLOCK)
			nth <<= 1;
		{ OpNest _n(OP_ATT_SMX_ID); attn_tc_softmax_kernel<<<sgrid, nth, 0, g_st>>>(dw, NULL, T, H,
		    (ann_real)scale,
		    g_att_seqlens_host ? g_att_seqlens_dev : NULL); }
		/* dP = dO V^T (needs V); then dV = W^T dO (overwrites V). */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_do_vt(g_tc_dp, g_tc_o, g_tc_v, T, dh, batch) != 0)
			goto tc_bwd_out;
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_wt_do(g_tc_v, dw, g_tc_o, T, dh, batch) != 0)
			goto tc_bwd_out;
		{ OpNest _n(OP_ATT_SMX_ID); attn_tc_softmax_bwd_kernel<<<sgrid, nth, 0, g_st>>>(g_tc_dp, NULL, dw,
		    T, H, g_att_seqlens_host ? g_att_seqlens_dev : NULL); }
		/* dQ = dS K -> g_tc_o; dK = dS^T Q -> g_tc_k (Q still in g_tc_q). */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_ds_k(g_tc_o, g_tc_dp, g_tc_k, T, dh, batch, scale)
		    != 0)
			goto tc_bwd_out;
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_dst_q(g_tc_k, g_tc_dp, g_tc_q, T, dh, batch, scale)
		    != 0)
			goto tc_bwd_out;
	}
	{ OpNest _n(OP_ATT_PACK_ID); attn_unpack_heads_kernel<<<(npack + threads - 1) / threads, threads, 0, g_st>>>(
	    a_dq, g_tc_o, B, T, H, dh); }
	{ OpNest _n(OP_ATT_PACK_ID); attn_unpack_heads_kernel<<<(npack + threads - 1) / threads, threads, 0, g_st>>>(
	    a_dk, g_tc_k, B, T, H, dh); }
	{ OpNest _n(OP_ATT_PACK_ID); attn_unpack_heads_kernel<<<(npack + threads - 1) / threads, threads, 0, g_st>>>(
	    a_dv, g_tc_v, B, T, H, dh); }
	if (launch_done() == 0 && finish_out(dq, a_dq, o[4]) == 0
	    && finish_out(dk, a_dk, o[5]) == 0
	    && finish_out(dv, a_dv, o[6]) == 0)
		rc = 0;
tc_bwd_out:
	for (i = 0; i < 7; i++)
		if (o[i] && ptr[i])
			gpu_free(ptr[i]);
	pthread_mutex_unlock(&g_att_mu);
	return rc;
#else
	(void)dq;
	(void)dk;
	(void)dv;
	(void)dconcat;
	(void)w;
	(void)q;
	(void)k;
	(void)v;
	(void)B;
	(void)T;
	(void)H;
	(void)dh;
	return -1;
#endif
}

__global__ static void pack_seq_rows_kernel(ann_real *dst, const ann_real *src,
	int B, int T, int D)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n = B * (T - 1) * D;
	int b;
	int t;
	int c;

	if (idx >= n || T < 2)
		return;
	c = idx % D;
	t = (idx / D) % (T - 1);
	b = idx / (D * (T - 1));
	dst[idx] = src[(b * T + t) * D + c];
}

__global__ static void unpack_seq_rows_kernel(ann_real *dst, const ann_real *src,
	int B, int T, int D)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n = B * (T - 1) * D;
	int b;
	int t;
	int c;

	if (idx >= n || T < 2)
		return;
	c = idx % D;
	t = (idx / D) % (T - 1);
	b = idx / (D * (T - 1));
	dst[(b * T + t) * D + c] = src[idx];
}

extern "C" int gpu_attn_qkv_fused(mat *q, mat *k, mat *v, const mat *x,
	const mat *wq, const mat *wk, const mat *wv)
{
	OpScope _op(OP_QKV);
	/*
	** With the FA2 hint (BF16, shape ok): one fat X@[Wq|Wk|Wv] GEMM into
	** the layer's BF16 qkv rows, read strided by the attention kernels, so
	** nothing is packed or split. Otherwise three GemmEx sharing one
	** BF16/FP16 convert of X (same for Wq/Wk/Wv): the fat GEMM followed by
	** a pack/split pass canceled its win at T=64 / D=1024 on this card.
	*/
	static void *s_x_half;
	static size_t s_x_half_cap;
	ann_real *dx;
	ann_real *dwq;
	ann_real *dwk;
	ann_real *dwv;
	ann_real *dq;
	ann_real *dk;
	ann_real *dv;
	int ox;
	int owq;
	int owk;
	int owv;
	int oq;
	int ok;
	int ov;
	int N;
	int D;
	int use_bf16;
	size_t nA;
	const int hint_T = g_qkv_hint_T;	/* one-shot: taken whatever path runs */
	const int hint_dh = g_qkv_hint_dh;

	g_qkv_hint_T = 0;
	if (!q || !k || !v || !x || !wq || !wk || !wv || !gpu_available())
		return -1;
	N = x->rows;
	D = x->cols;
	if (D <= 0 || N <= 0 || wq->rows != D || wq->cols != D
	    || wk->rows != D || wk->cols != D || wv->rows != D || wv->cols != D
	    || q->rows != N || q->cols != D || k->rows != N || k->cols != D
	    || v->rows != N || v->cols != D)
		return -1;
	ox = owq = owk = owv = oq = ok = ov = 0;
	dx = dwq = dwk = dwv = dq = dk = dv = NULL;
	if (g_compute_prec == GPU_PREC_FP32)
		goto qkv_triple;
	if (ensure_dev(x, &dx, &ox) != 0)
		goto qkv_triple;
	if (ensure_dev(wq, &dwq, &owq) != 0)
		goto qkv_triple;
	if (ensure_dev(wk, &dwk, &owk) != 0)
		goto qkv_triple;
	if (ensure_dev(wv, &dwv, &owv) != 0)
		goto qkv_triple;
	if (ensure_dev_out(q, &dq, &oq) != 0)
		goto qkv_triple;
	/*
	** Lazy K/V (ANN_VRAM_LAZY): when the FA2 GEMM below will leave K and V
	** only in the BF16 qkv slot, their FP32 mats need no device buffer
	** (2 x 16 MB per layer). Any other path allocates them on demand.
	*/
	if (!(vram_lazy_on() && !k->d_data && !v->d_data && !k->data && !v->data
	    && g_compute_prec == GPU_PREC_BF16 && !g_fp8_fwd && hint_T > 0
	    && fa2_on() && fa2_ok(hint_T, hint_dh)
	    && (D & 7) == 0)) {
		if (ensure_dev_out(k, &dk, &ok) != 0)
			goto qkv_triple;
		if (ensure_dev_out(v, &dv, &ov) != 0)
			goto qkv_triple;
	}
	nA = (size_t)N * (size_t)D;
#if defined(ANN_CUBLAS) && CUDART_VERSION >= 12000
	if (g_fp8_fwd && (D & 15) == 0 && (N & 15) == 0) {
		void *wq8 = NULL, *wk8 = NULL, *wv8 = NULL;
		float *sq = NULL, *sk = NULL, *sv = NULL;
		const float alpha = 1.0f;
		const float beta = 0.0f;

		void *act8 = NULL;
		float *act_scale = NULL;
		int hot = 0;
		fp8_act_slot *as;

		pthread_mutex_lock(&g_gemm_mu);
		as = fp8_act_scale(dx, 0);
		hot = (fp8_hot_take(dx, (int)nA, &act8, &act_scale) == 0);
		if (!hot && as)
			act_scale = FP8_SCALE_CUR(as);
		if (act_scale && fp8_scales_ensure() == 0
		    && fp8_wt_get_ex(dwq, D, D, 1, 0, &wq8, &sq) == 0
		    && fp8_wt_get_ex(dwk, D, D, 1, 0, &wk8, &sk) == 0
		    && fp8_wt_get_ex(dwv, D, D, 1, 0, &wv8, &sv) == 0
		    && (hot || (ensure_fp8_buf(&g_fp8_A, &g_fp8_A_cap, nA) == 0
		    && fp8_quant_tensor(g_fp8_A, act_scale, dx, (int)nA, 0)
		    == 0))
		    && cublas_lt_gemm_fp8_ex(CUBLAS_OP_T, CUBLAS_OP_N, D, N, D,
		    &alpha, wq8, D, sq, CUDA_R_8F_E4M3,
		    hot ? act8 : g_fp8_A, D,
		    act_scale, CUDA_R_8F_E4M3, &beta,
		    dq, D) == 0
		    && cublas_lt_gemm_fp8_ex(CUBLAS_OP_T, CUBLAS_OP_N, D, N, D,
		    &alpha, wk8, D, sk, CUDA_R_8F_E4M3,
		    hot ? act8 : g_fp8_A, D,
		    act_scale, CUDA_R_8F_E4M3, &beta,
		    dk, D) == 0
		    && cublas_lt_gemm_fp8_ex(CUBLAS_OP_T, CUBLAS_OP_N, D, N, D,
		    &alpha, wv8, D, sv, CUDA_R_8F_E4M3,
		    hot ? act8 : g_fp8_A, D,
		    act_scale, CUDA_R_8F_E4M3, &beta,
		    dv, D) == 0) {
			if (hot)
				fp8_hot_finish();
			else
				fp8_act_scale_commit(act_scale, 0);
			pthread_mutex_unlock(&g_gemm_mu);
			if (launch_done() != 0)
				goto qkv_triple;
			q->on_device = MAT_DEV_DEVICE;
			k->on_device = MAT_DEV_DEVICE;
			v->on_device = MAT_DEV_DEVICE;
			if (ox && dx)
				gpu_free(dx);
			if (owq && dwq)
				gpu_free(dwq);
			if (owk && dwk)
				gpu_free(dwk);
			if (owv && dwv)
				gpu_free(dwv);
			return 0;
		}
		if (hot) {
			g_fp8_hot_src = NULL;
			g_fp8_hot_n = 0;
		}
		pthread_mutex_unlock(&g_gemm_mu);
	}
#endif
	use_bf16 = (g_compute_prec == GPU_PREC_BF16);
	{
		const void *cached = use_bf16 ? act_bf16_get(dx, (int)nA)
		    : NULL;
		void *x_half;

		if (cached)
			x_half = (void *)cached;
		else {
			if (ensure_half_buf(&s_x_half, &s_x_half_cap, nA * 2)
			    != 0)
				goto qkv_triple;
			if (half_wt_convert(s_x_half, dx, (int)nA, use_bf16)
			    != 0)
				goto qkv_triple;
			x_half = s_x_half;
		}
#ifdef ANN_CUBLAS
		/*
		** FA2 path: one GEMM X @ [Wq|Wk|Wv] straight to BF16 in the
		** layer's qkv slot; the attention kernels read it strided, so
		** the FP32 q/k/v are never written (their mats are marked
		** device-resident but hold nothing) and nothing is packed.
		*/
		if (use_bf16 && hint_T > 0 && fa2_on()
		    && fa2_ok(hint_T, hint_dh) && !oq && !ok && !ov
		    && (D & 7) == 0) {
			qkv_bf16_slot *qs = qkv_bf16_get(dq, N, D, 1);
			void *hwq = half_wt_get(dwq, (size_t)D * (size_t)D, 1);
			void *hwk = half_wt_get(dwk, (size_t)D * (size_t)D, 1);
			void *hwv = half_wt_get(dwv, (size_t)D * (size_t)D, 1);
			const float alpha = 1.0f;
			const float beta = 0.0f;
			void *wqkv;
			int rc = -1;

			if (qs && hwq && hwk && hwv) {
				pthread_mutex_lock(&g_gemm_mu);
				wqkv = qkv_wt_get(hwq, hwk, hwv, D);
				if (wqkv)
					rc = cublas_lt_gemm_half(CUBLAS_OP_N, CUBLAS_OP_N,
					    3 * D, N, D, &alpha, wqkv, 3 * D, x_half, D,
					    &beta, (float *)qs->buf, 3 * D, 1,
					    CUBLASLT_EPILOGUE_DEFAULT, NULL, NULL, 1);
				pthread_mutex_unlock(&g_gemm_mu);
			}
			if (rc == 0) {
				qs->live = 1;
				goto qkv_done;
			}
			(void)cudaGetLastError();
		}
		/*
		** Overlap the three independent projections on dedicated
		** streams. Falls back to serial rm_ha if setup fails.
		*/
		if ((!dk || !dv) && (ensure_dev_out(k, &dk, &ok) != 0
		    || ensure_dev_out(v, &dv, &ov) != 0))
			goto qkv_triple;	/* lazy K/V, FA2 GEMM did not run */
		{
			void *hwq;
			void *hwk;
			void *hwv;
			const float alpha = 1.0f;
			const float beta = 0.0f;
			int i;
			int par = 0;

			/*
			** Three streams and a host wait per call: not
			** capturable. A capture that would need them is
			** thrown away rather than run on another GEMM
			** path (other algorithms, other rounding).
			*/
			if (use_bf16 && stream_is_capturing())
				capture_refuse("parallel QKV streams", __LINE__);
			else if (use_bf16 && qkv_par_ensure() == 0) {
				hwq = half_wt_get(dwq, (size_t)D * (size_t)D,
				    1);
				hwk = half_wt_get(dwk, (size_t)D * (size_t)D,
				    1);
				hwv = half_wt_get(dwv, (size_t)D * (size_t)D,
				    1);
				if (hwq && hwk && hwv
				    && cudaEventRecord(g_qkv_ready, g_st)
				    == cudaSuccess) {
					void *hW[3];
					ann_real *dC[3];

					hW[0] = hwq;
					hW[1] = hwk;
					hW[2] = hwv;
					dC[0] = dq;
					dC[1] = dk;
					dC[2] = dv;
					par = 1;
					for (i = 0; i < 3; i++) {
						if (cudaStreamWaitEvent(
						    g_qkv_stream[i],
						    g_qkv_ready, 0)
						    != cudaSuccess
						    || cublas_gemm_ex_bf16_h(
						    g_qkv_blas[i], D, N, D,
						    &alpha, hW[i], D, x_half,
						    D, &beta, dC[i], D) != 0) {
							par = 0;
							break;
						}
					}
					if (par) {
						for (i = 0; i < 3; i++)
							if (cudaStreamSynchronize(
							    g_qkv_stream[i])
							    != cudaSuccess)
								par = 0;
					}
				}
			}
			if (!par
			    && (cublas_gemm_rm_ha(dq, dx, dwq, N, D, D,
			    x_half) != 0
			    || cublas_gemm_rm_ha(dk, dx, dwk, N, D, D,
			    x_half) != 0
			    || cublas_gemm_rm_ha(dv, dx, dwv, N, D, D,
			    x_half) != 0))
				goto qkv_triple;
		}
#else
		(void)x_half;
		goto qkv_triple;
#endif
	}
qkv_done:
	if (launch_done() != 0)
		goto qkv_triple;
	q->on_device = MAT_DEV_DEVICE;
	if (k->d_data)
		k->on_device = MAT_DEV_DEVICE;
	if (v->d_data)
		v->on_device = MAT_DEV_DEVICE;
	if (ox && dx)
		gpu_free(dx);
	if (owq && dwq)
		gpu_free(dwq);
	if (owk && dwk)
		gpu_free(dwk);
	if (owv && dwv)
		gpu_free(dwv);
	return 0;
qkv_triple:
	if (ox && dx)
		gpu_free(dx);
	if (owq && dwq)
		gpu_free(dwq);
	if (owk && dwk)
		gpu_free(dwk);
	if (owv && dwv)
		gpu_free(dwv);
	if (gpu_mat_mul(q, x, wq) != 0)
		return -1;
	if (gpu_mat_mul(k, x, wk) != 0)
		return -1;
	if (gpu_mat_mul(v, x, wv) != 0)
		return -1;
	return 0;
}

extern "C" int gpu_pack_seq_rows(ann_real *dst, const ann_real *src, int B,
	int T, int D)
{
	OpScope _op(OP_PACK);
	int n;
	int threads = 256;

	if (!dst || !src || B <= 0 || T < 2 || D <= 0 || !gpu_available())
		return -1;
	n = B * (T - 1) * D;
	pack_seq_rows_kernel<<<(n + threads - 1) / threads, threads, 0, g_st>>>(dst, src,
	    B, T, D);
	return launch_done();
}

extern "C" int gpu_unpack_seq_rows(ann_real *dst, const ann_real *src, int B,
	int T, int D)
{
	OpScope _op(OP_PACK);
	int n;
	int threads = 256;

	if (!dst || !src || B <= 0 || T < 2 || D <= 0 || !gpu_available())
		return -1;
	n = B * (T - 1) * D;
	unpack_seq_rows_kernel<<<(n + threads - 1) / threads, threads, 0, g_st>>>(dst,
	    src, B, T, D);
	return launch_done();
}

static int gpu_layernorm_forward_ex_locked(ann_real *y, const ann_real *x,
	const ann_real *gamma, const ann_real *beta, ann_real *x_hat,
	ann_real *mean, ann_real *inv_std, int T, int d, double eps,
	__nv_bfloat16 *y16)
{
	ew_ref rx;
	ew_ref ry;
	ew_ref rg;
	ew_ref rb;
	ew_ref rh;
	ew_ref rm;
	ew_ref ri;
	size_t mb;
	size_t vb;
	size_t gb;
	int did;
	int rc;

	if (!y || !x || !gamma || !beta || T <= 0 || d <= 0 || !gpu_available())
		return -1;
	mb = (size_t)T * (size_t)d * sizeof(ann_real);
	vb = (size_t)T * sizeof(ann_real);
	gb = (size_t)d * sizeof(ann_real);
	if (ew_acquire(&rx, 0, x, mb, 1, 0) != 0)
		return -1;
	if (y == x) {
		ry = rx;
		ew_mark_down(&ry);
	} else if (ew_acquire(&ry, 1, y, mb, 0, 1) != 0) {
		return -1;
	}
	if (ew_acquire_param(&rg, 2, gamma, gb) != 0
	    || ew_acquire_param(&rb, 3, beta, gb) != 0)
		return -1;
	memset(&rh, 0, sizeof(rh));
	memset(&rm, 0, sizeof(rm));
	memset(&ri, 0, sizeof(ri));
	if (x_hat && ew_acquire(&rh, 4, x_hat, mb, 0, 1) != 0)
		return -1;
	/* mean / inv_std: device resident (only kernels use them). */
	if (mean) {
		rm.dev = (ann_real *)(ew_is_dev(mean) ? (void *)mean
		    : devvec_get(mean, vb));
		if (!rm.dev)
			return -1;
	}
	if (inv_std) {
		ri.dev = (ann_real *)(ew_is_dev(inv_std) ? (void *)inv_std
		    : devvec_get(inv_std, vb));
		if (!ri.dev)
			return -1;
	}
	{
		__nv_fp8_storage_t *y8 = NULL;
		float *sc8 = NULL;
		float *am8 = NULL;
		int hot = 0;
		int bs = ew_block_for(d);

#if defined(ANN_CUBLAS) && CUDART_VERSION >= 12000
		if (d <= 4096
		    && fp8_hot_prepare(T * d, ry.dev, &y8, &sc8, &am8) == 0)
			hot = 1;
#endif
		if (!y8 && ln_fast_ok(d, ry.dev, rx.dev, rg.dev, rb.dev,
		    x_hat ? (void *)rh.dev : (void *)y16)
		    && (!x_hat || !y16 || ((size_t)y16 & 15) == 0))
			ln_fwd_fast_launch(ry.dev, rx.dev, rg.dev, rb.dev,
			    x_hat ? rh.dev : NULL, mean ? rm.dev : NULL,
			    inv_std ? ri.dev : NULL, T, d, (ann_real)eps, y16);
		else
			ln_fwd_kernel<<<T, bs, (size_t)d * sizeof(ann_real), g_st>>>(ry.dev,
			    rx.dev, rg.dev, rb.dev, x_hat ? rh.dev : NULL,
			    mean ? rm.dev : NULL, inv_std ? ri.dev : NULL, d,
			    (ann_real)eps, y8, sc8, am8, y16);
		if (ew_launch_ok() != 0)
			return -1;
#if defined(ANN_CUBLAS) && CUDART_VERSION >= 12000
		if (hot)
			fp8_hot_commit(ry.dev, T * d);
#endif
		(void)hot;
	}
	/* Download y last: with y == x a late failure must not lose the input. */
	did = 0;
	if (x_hat) {
		if ((rc = ew_release(&rh)) < 0)
			return -1;
		did |= rc;
	}
	/* Eager mode keeps the host stats valid for CPU fallbacks / tests. */
	if (g_sync_mode == GPU_SYNC_EAGER && !stream_is_capturing()) {
		if (mean && (void *)rm.dev != (void *)mean
		    && cudaMemcpy(mean, rm.dev, vb, cudaMemcpyDeviceToHost)
		    != cudaSuccess)
			return -1;
		if (inv_std && (void *)ri.dev != (void *)inv_std
		    && cudaMemcpy(inv_std, ri.dev, vb, cudaMemcpyDeviceToHost)
		    != cudaSuccess)
			return -1;
		did = 1;
	}
	if ((rc = ew_release(&ry)) < 0)
		return -1;
	did |= rc;
	return ew_finish_sync(did) == 0 ? 0 : -1;
}

extern "C" int gpu_layernorm_forward_ex(ann_real *y, const ann_real *x,
	const ann_real *gamma, const ann_real *beta, ann_real *x_hat,
	ann_real *mean, ann_real *inv_std, int T, int d, double eps)
{
	OpScope _op(OP_LN_FWD);
	int rc;
	act_bf16_slot *as = NULL;

	/*
	** BF16 training: the LayerNorm output goes straight into its BF16
	** twin (all its readers are GEMMs); the FP32 y is not written.
	** ANN_LN_BF16=0 keeps the FP32 output plus a conversion.
	*/
	if (y && T > 0 && d > 0 && g_compute_prec == GPU_PREC_BF16
	    && g_sync_mode != GPU_SYNC_EAGER
	    && ln_bf16_on() && ew_is_dev(y))
		as = act_bf16_reserve(y, T * d);
	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_layernorm_forward_ex_locked(y, x, gamma, beta, x_hat, mean, inv_std, T, d, eps,
	    as ? (__nv_bfloat16 *)as->half : NULL);
	pthread_mutex_unlock(&g_ew_mu);
	if (as) {
		if (rc == 0) {
			static int reported;

			as->live = 1;
			if (!reported++)
				fprintf(stderr, "[ln] forward output written as its BF16 twin only\n");
			return 0;
		}
		as->live = 0;
		as->bf16_only = 0;
		(void)cudaGetLastError();
		pthread_mutex_lock(&g_ew_mu);
		rc = gpu_layernorm_forward_ex_locked(y, x, gamma, beta, x_hat, mean, inv_std, T, d, eps, NULL);
		pthread_mutex_unlock(&g_ew_mu);
	}
	if (rc == 0 && y && T > 0 && d > 0
	    && g_compute_prec == GPU_PREC_BF16)
		act_bf16_publish(y, T * d);
	return rc;
}

/*
** x_in / mean non-NULL (B12): x_hat is NULL and recomputed in the kernel
** from the LayerNorm input and the row means; only the warp-per-row kernel
** does that, so any other shape returns -1 before launching anything.
*/
static int gpu_layernorm_backward_locked(mat *dx, const mat *dy, const ann_real *gamma,
	const ann_real *x_hat, const ann_real *inv_std, int T, int d,
	ann_real *dgamma, ann_real *dbeta, int accumulate,
	const ann_real *x_in, const ann_real *mean)
{
	ew_ref rdx;
	ew_ref rdy;
	ew_ref rg;
	ew_ref rh;
	ew_ref ri;
	ann_real *rmean = NULL;
	colsum_acc *acc_g;
	colsum_acc *acc_b;
	const ann_real *dyp;
	size_t mb;
	size_t vb;
	size_t gb;
	int fast_bwd = 0;
	int did;
	int rc;

	if (!dx || !dy || !gamma || !inv_std || !dgamma || !dbeta)
		return -1;
	if (x_in ? (!mean || x_hat) : !x_hat)
		return -1;
	if (!gpu_available() || T <= 0 || d <= 0)
		return -1;
	mb = (size_t)T * (size_t)d * sizeof(ann_real);
	vb = (size_t)T * sizeof(ann_real);
	gb = (size_t)d * sizeof(ann_real);
	/* Prefer device twins when authoritative (R1 residency). */
	dyp = (dy->d_data && dy->on_device != MAT_DEV_HOST) ? dy->d_data : dy->data;
	/* Accumulating needs the current dx on the device. */
	if (accumulate && (!dx->d_data || dx->on_device == MAT_DEV_HOST))
		return -1;
	if (dx->d_data) {
		if (ew_acquire(&rdx, 0, dx->d_data, mb, 0, 0) != 0)
			return -1;
	} else if (ew_acquire(&rdx, 0, dx->data, mb, 0, 1) != 0)
		return -1;
	if (ew_acquire(&rdy, 1, dyp, mb, 1, 0) != 0
	    || ew_acquire_param(&rg, 2, gamma, gb) != 0
	    || ew_acquire(&rh, 3, x_in ? x_in : x_hat, mb, 1, 0) != 0)
		return -1;
	ri.dev = (ann_real *)(ew_is_dev(inv_std) ? (void *)inv_std
	    : devvec_get(inv_std, vb));
	if (!ri.dev)
		return -1;
	if (x_in) {
		rmean = (ann_real *)(ew_is_dev(mean) ? (void *)mean
		    : devvec_get(mean, vb));
		if (!rmean || !ln_fast_ok(d, rdx.dev, rdy.dev, rg.dev, rh.dev,
		    NULL) || ((size_t)ri.dev & 3) != 0)
			return -1;
	}
	/*
	** dgamma / dbeta accumulate on the device (same accumulators as the
	** FFN bias grads) and reach the host at gpu_device_sync: staging them
	** through host copies cost two full-pipeline syncs per layer.
	*/
	acc_g = colsum_acc_get(dgamma, d);
	acc_b = colsum_acc_get(dbeta, d);
	if (!acc_g || !acc_b)
		return -1;
	{
		void *dx16 = NULL;

		/* Accumulate mode (training): the result feeds wgrad/dgrad GEMMs; hand them its BF16 twin. */
		if (accumulate && dx->d_data && g_compute_prec == GPU_PREC_BF16
		    && twin_emit_on()) {
			pthread_mutex_lock(&g_gemm_mu);
			dx16 = grad_twin_reserve(dx->d_data, (size_t)T * (size_t)d);
			pthread_mutex_unlock(&g_gemm_mu);
		}
		if (ln_fast_ok(d, rdx.dev, rdy.dev, rg.dev, rh.dev, dx16)
		    && ((size_t)ri.dev & 3) == 0 && ((size_t)acc_g->dev & 3) == 0
		    && ((size_t)acc_b->dev & 3) == 0) {
			fast_bwd = 1;
			ln_bwd_fast_launch(rdx.dev, rdy.dev, rg.dev,
			    x_in ? NULL : rh.dev, ri.dev, acc_g->dev, acc_b->dev,
			    T, d, accumulate, (__nv_bfloat16 *)dx16,
			    x_in ? rh.dev : NULL, rmean);
		} else if (x_in) {
			if (dx16) {
				pthread_mutex_lock(&g_gemm_mu);
				grad_twin_forget(dx->d_data);
				pthread_mutex_unlock(&g_gemm_mu);
			}
			return -1;
		} else {
			ln_bwd_kernel<<<T, ew_block_for(d), 0, g_st>>>(rdx.dev, rdy.dev, rg.dev, rh.dev,
			    ri.dev, NULL, NULL, d, accumulate, (__nv_bfloat16 *)dx16);
		}
		if (dx16 && ew_launch_ok() != 0) {
			grad_twin_forget(dx->d_data);
			return -1;
		}
	}
	if (ew_launch_ok() != 0)
		return -1;
	if (!fast_bwd) {
		col_sum_tiled_run(acc_b->dev, rdy.dev, NULL, T, d);
		col_sum_tiled_run(acc_g->dev, rdy.dev, rh.dev, T, d);
	}
	if (ew_launch_ok() != 0)
		return -1;
	acc_g->dirty = 1;
	acc_b->dirty = 1;
	did = 0;
	if ((rc = ew_release(&rdx)) < 0)
		return -1;
	did |= rc;
	if (dx->d_data)
		dx->on_device = MAT_DEV_DEVICE;
	if (ew_finish_sync(did) != 0)
		return -1;
	/* Eager mode: callers read host dgamma right away. */
	if (g_sync_mode == GPU_SYNC_EAGER && !stream_is_capturing())
		colsum_flush_all();
	return 0;
}

extern "C" int gpu_layernorm_backward(mat *dx, const mat *dy, const ann_real *gamma,
	const ann_real *x_hat, const ann_real *inv_std, int T, int d,
	ann_real *dgamma, ann_real *dbeta)
{
	OpScope _op(OP_LN_BWD);
	int rc;

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_layernorm_backward_locked(dx, dy, gamma, x_hat, inv_std, T, d, dgamma, dbeta, 0,
	    NULL, NULL);
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}

/* Same, dx += LayerNorm backward (dx device-resident): the residual-path
** gradient add of a block fused into the LN backward kernel. */
extern "C" int gpu_layernorm_backward_acc(mat *dx, const mat *dy, const ann_real *gamma,
	const ann_real *x_hat, const ann_real *inv_std, int T, int d,
	ann_real *dgamma, ann_real *dbeta)
{
	OpScope _op(OP_LN_BWD);
	int rc;

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_layernorm_backward_locked(dx, dy, gamma, x_hat, inv_std, T, d, dgamma, dbeta, 1,
	    NULL, NULL);
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}

/*
** B12: the same backward with x_hat recomputed from the LayerNorm input x
** and the row means stored by the forward (x_hat was not written).
** accumulate: dx += (dx device-resident). -1 = recompute the x_hat
** (gpu_layernorm_xhat) and take the paths above.
*/
extern "C" int gpu_layernorm_backward_x(mat *dx, const mat *dy, const ann_real *gamma,
	const ann_real *x, const ann_real *mean, const ann_real *inv_std,
	int T, int d, ann_real *dgamma, ann_real *dbeta, int accumulate)
{
	OpScope _op(OP_LN_BWD);
	int rc;

	if (!x || !mean)
		return -1;
	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_layernorm_backward_locked(dx, dy, gamma, NULL, inv_std, T, d, dgamma, dbeta,
	    accumulate ? 1 : 0, x, mean);
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}

__global__ static void ln_xhat_kernel(ann_real *x_hat, const ann_real *x,
	const ann_real *mean, const ann_real *inv_std, int T, int d)
{
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	int r;

	if (i >= (size_t)T * (size_t)d)
		return;
	r = (int)(i / (size_t)d);
	x_hat[i] = __fmul_rn(__fsub_rn(x[i], mean[r]), inv_std[r]);
}

/*
** Fallback of B12: write x_hat = (x - mean) * inv_std (as the forward does)
** into x_hat (device) when a backward needs it materialised.
*/
extern "C" int gpu_layernorm_xhat(ann_real *x_hat, const ann_real *x,
	const ann_real *mean, const ann_real *inv_std, int T, int d)
{
	ew_ref rx;
	ew_ref rh;
	ann_real *rm;
	ann_real *ri;
	size_t mb;
	size_t vb;
	size_t n;
	int rc;

	if (!x_hat || !x || !mean || !inv_std || T <= 0 || d <= 0
	    || !gpu_available())
		return -1;
	mb = (size_t)T * (size_t)d * sizeof(ann_real);
	vb = (size_t)T * sizeof(ann_real);
	n = (size_t)T * (size_t)d;
	pthread_mutex_lock(&g_ew_mu);
	rc = -1;
	if (ew_acquire(&rx, 0, x, mb, 1, 0) == 0
	    && ew_acquire(&rh, 1, x_hat, mb, 0, 1) == 0) {
		rm = (ann_real *)(ew_is_dev(mean) ? (void *)mean
		    : devvec_get(mean, vb));
		ri = (ann_real *)(ew_is_dev(inv_std) ? (void *)inv_std
		    : devvec_get(inv_std, vb));
		if (rm && ri) {
			ln_xhat_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_st>>>(
			    rh.dev, rx.dev, rm, ri, T, d);
			if (ew_launch_ok() == 0) {
				int did = ew_release(&rh);

				if (did >= 0 && ew_finish_sync(did) == 0)
					rc = 0;
			}
		}
	}
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}

static int gpu_bias_row_add_locked(ann_real *x, const ann_real *bias, int rows,
	int cols)
{
	ew_ref rx;
	ew_ref rb;
	long n;
	int rc;

	if (!x || !bias || rows <= 0 || cols <= 0 || !gpu_available())
		return -1;
	n = (long)rows * (long)cols;
	if (n > 0x7fffffffL)
		return -1;
	if (ew_acquire(&rx, 0, x, (size_t)n * sizeof(ann_real), 1, 1) != 0
	    || ew_acquire_param(&rb, 1, bias, (size_t)cols * sizeof(ann_real)) != 0)
		return -1;
	bias_add_kernel<<<(int)((n + 255) / 256), 256, 0, g_st>>>(rx.dev, rb.dev, (int)n,
	    cols);
	if (ew_launch_ok() != 0)
		return -1;
	rc = ew_release(&rx);
	if (rc < 0)
		return -1;
	return ew_finish_sync(rc) == 0 ? 0 : -1;
}

extern "C" int gpu_bias_row_add(ann_real *x, const ann_real *bias, int rows,
	int cols)
{
	OpScope _op(OP_ELEMWISE);
	int rc;

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_bias_row_add_locked(x, bias, rows, cols);
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}

static int gpu_relu_backward_locked(ann_real *dx, const ann_real *x_pre_relu_or_mask,
	int n)
{
	ew_ref rdx;
	ew_ref rr;
	int rc;

	if (!dx || !x_pre_relu_or_mask || n <= 0 || !gpu_available())
		return -1;
	if (ew_acquire(&rdx, 0, dx, (size_t)n * sizeof(ann_real), 1, 1) != 0
	    || ew_acquire(&rr, 1, x_pre_relu_or_mask,
	    (size_t)n * sizeof(ann_real), 1, 0) != 0)
		return -1;
	relu_bwd_kernel<<<(n + 255) / 256, 256, 0, g_st>>>(rdx.dev, rr.dev, n);
	if (ew_launch_ok() != 0)
		return -1;
	rc = ew_release(&rdx);
	if (rc < 0)
		return -1;
	return ew_finish_sync(rc) == 0 ? 0 : -1;
}

extern "C" int gpu_relu_backward(ann_real *dx, const ann_real *x_pre_relu_or_mask,
	int n)
{
	OpScope _op(OP_ELEMWISE);
	int rc;

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_relu_backward_locked(dx, x_pre_relu_or_mask, n);
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}

/*
** FFN ReLU backward in one pass: dst = BF16(dh masked by hid > 0) and
** acc[c] += column sums of the masked dh. One thread per 4 columns and
** COLSUM_SLAB rows (float4 loads, bfloat162 stores), FP32 sums, one
** atomic per column and slab (as col_sum_tiled_kernel). cols % 4 == 0.
*/
__global__ static void relu_bwd_pack_colsum_kernel(__nv_bfloat16 *dst,
	ann_real *acc, const ann_real *dh, const ann_real *hid,
	const __nv_bfloat16 *hid16, int rows, int cols, ann_real *part)
{
	const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	const int r0 = blockIdx.y * COLSUM_SLAB;
	const int r1 = min(r0 + COLSUM_SLAB, rows);
	float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
	int r;

	if (c0 >= cols)
		return;
	for (r = r0; r < r1; r++) {
		const size_t off = (size_t)r * cols + c0;
		float4 g = *(const float4 *)(dh + off);
		float4 h;

		if (hid16) {
			/* the twin the W2 GEMM used: half the bytes, same sign */
			const uint2 u = *(const uint2 *)(hid16 + off);
			const __nv_bfloat162 lo = *(const __nv_bfloat162 *)&u.x;
			const __nv_bfloat162 hi = *(const __nv_bfloat162 *)&u.y;

			h.x = __low2float(lo);
			h.y = __high2float(lo);
			h.z = __low2float(hi);
			h.w = __high2float(hi);
		} else
			h = *(const float4 *)(hid + off);
		g.x = h.x > 0.0f ? g.x : 0.0f;
		g.y = h.y > 0.0f ? g.y : 0.0f;
		g.z = h.z > 0.0f ? g.z : 0.0f;
		g.w = h.w > 0.0f ? g.w : 0.0f;
		s0 += g.x;
		s1 += g.y;
		s2 += g.z;
		s3 += g.w;
		*(__nv_bfloat162 *)(dst + off) = __floats2bfloat162_rn(g.x, g.y);
		*(__nv_bfloat162 *)(dst + off + 2) = __floats2bfloat162_rn(g.z, g.w);
	}
	if (part) {	/* ANN_DET_SUM: slab partials, reduced in a fixed order */
		*(float4 *)(part + (size_t)blockIdx.y * (size_t)cols + c0)
		    = make_float4(s0, s1, s2, s3);
		return;
	}
	atomicAdd(acc + c0, s0);
	atomicAdd(acc + c0 + 1, s1);
	atomicAdd(acc + c0 + 2, s2);
	atomicAdd(acc + c0 + 3, s3);
}

/*
** FFN backward, BF16 mode: the ReLU backward of dh (mask hidden > 0), its
** BF16 twin for the W1 wgrad and dgrad GEMMs, and the b1 gradient column
** sums in one kernel. The twin goes into a grad-twin slot keyed on dh's
** device buffer (the wgrad peeks it, the dgrad takes it); db1 accumulates
** like gpu_mat_col_sum_add. The FP32 dh is left unmasked: in this mode no
** GEMM reads it. -1 = run the three steps (gpu_relu_backward, the
** conversions inside the GEMMs, gpu_mat_col_sum_add).
*/
extern "C" int gpu_ffn_relu_bwd_fused(mat *dh, const mat *hidden,
	ann_real *db1_host)
{
	OpScope _op(OP_ELEMWISE);
	ann_real *ddh;
	const ann_real *dhid;
	colsum_acc *acc;
	void *twin;
	const void *hid16;
	int rows, cols;
	size_t n;

	if (!dh || !hidden || !db1_host || !gpu_available()
	    || g_compute_prec != GPU_PREC_BF16)
		return -1;
	rows = dh->rows;
	cols = dh->cols;
	if (hidden->rows != rows || hidden->cols != cols || (cols % 4) != 0
	    || rows <= 0)
		return -1;
	/* Both must be device-resident in place: the twin is keyed on dh. */
	if (!dh->d_data || dh->on_device != MAT_DEV_DEVICE || !hidden->d_data
	    || hidden->on_device != MAT_DEV_DEVICE)
		return -1;
	ddh = dh->d_data;
	dhid = hidden->d_data;
	n = (size_t)rows * (size_t)cols;
	pthread_mutex_lock(&g_gemm_mu);
	hid16 = act_bf16_get(dhid, (int)n);	/* NULL: mask from the FP32 hidden */
	twin = grad_twin_reserve(ddh, n);
	pthread_mutex_unlock(&g_gemm_mu);
	if (!twin)
		return -1;
	pthread_mutex_lock(&g_ew_mu);
	acc = colsum_acc_get(db1_host, cols);
	if (!acc) {
		pthread_mutex_unlock(&g_ew_mu);
		grad_twin_forget(ddh);
		return -1;
	}
	{
		dim3 grid((cols / 4 + EW_BLOCK - 1) / EW_BLOCK,
		    (rows + COLSUM_SLAB - 1) / COLSUM_SLAB);
		ann_real *part = grid.y > 1
		    ? cs_part((size_t)grid.y * (size_t)cols) : NULL;

		relu_bwd_pack_colsum_kernel<<<grid, EW_BLOCK, 0, g_st>>>(
		    (__nv_bfloat16 *)twin, acc->dev, ddh, dhid,
		    (const __nv_bfloat16 *)hid16, rows, cols, part);
		if (part)
			colsum_reduce_launch(acc->dev, part, (int)grid.y, cols,
			    cols);
	}
	if (ew_launch_ok() != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		grad_twin_forget(ddh);
		return -1;
	}
	acc->dirty = 1;
	if (g_sync_mode == GPU_SYNC_EAGER) {
		ann_real *db_tmp = (ann_real *)malloc((size_t)cols * sizeof(ann_real));
		int j;

		if (!db_tmp || cudaDeviceSynchronize() != cudaSuccess
		    || cudaMemcpy(db_tmp, acc->dev, (size_t)cols * sizeof(ann_real),
		    cudaMemcpyDeviceToHost) != cudaSuccess) {
			free(db_tmp);
			pthread_mutex_unlock(&g_ew_mu);
			grad_twin_forget(ddh);
			return -1;
		}
		for (j = 0; j < cols; j++)
			db1_host[j] += db_tmp[j];
		free(db_tmp);
		(void)cudaMemsetAsync(acc->dev, 0, (size_t)cols * sizeof(ann_real), g_st);
		acc->dirty = 0;
	}
	pthread_mutex_unlock(&g_ew_mu);
	{
		static int reported;

		if (!reported++)
			fprintf(stderr, "[ffn] ReLU backward, BF16 twin of dh and db1 in one kernel\n");
	}
	return 0;
}

/* Decoupled weight decay for the device Adam kernels (host path: lm_adam.c). */
static float g_adam_wd;

extern "C" void gpu_adam_set_weight_decay(double wd)
{
	g_adam_wd = wd > 0.0 ? (float)wd : 0.0f;
}

/* Sum of squares of x[0..n) into *acc (double atomics, sm_60+). */
__global__ static void sumsq_kernel(const ann_real *x, int n, double *acc)
{
	__shared__ double s[256];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int m;
	double v;

	v = i < n ? (double)x[i] * (double)x[i] : 0.0;
	s[threadIdx.x] = v;
	__syncthreads();
	for (m = blockDim.x >> 1; m > 0; m >>= 1) {
		if ((int)threadIdx.x < m)
			s[threadIdx.x] += s[threadIdx.x + m];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		atomicAdd(acc, s[0]);
}

/*
** Grid-stride variant (A1): a few blocks per SM each sum a strided share of x in
** double, reduce with warp shuffles, then one double atomicAdd per block. The
** one-block-per-256-values version above issued ~720k atomics on a single
** address per step (5.3 ms); the result only feeds the clip norm, whose
** summation order was already nondeterministic.
*/
__global__ static void sumsq_gs_kernel(const ann_real *x, size_t n, double *acc,
	double *part)
{
	__shared__ double s[32];
	size_t stride = (size_t)gridDim.x * blockDim.x;
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	size_t n4 = n >> 2;
	const float4 *x4 = (const float4 *)x;
	double v = 0.0;
	int lane = threadIdx.x & 31;
	int w = threadIdx.x >> 5;
	int k;

	for (; i < n4; i += stride) {
		float4 q = x4[i];

		v += (double)q.x * (double)q.x;
		v += (double)q.y * (double)q.y;
		v += (double)q.z * (double)q.z;
		v += (double)q.w * (double)q.w;
	}
	i = (n4 << 2) + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < n; i += stride)
		v += (double)x[i] * (double)x[i];
	for (k = 16; k > 0; k >>= 1)
		v += __shfl_down_sync(0xffffffffu, v, k);
	if (lane == 0)
		s[w] = v;
	__syncthreads();
	if (w == 0) {
		v = lane < (int)(blockDim.x >> 5) ? s[lane] : 0.0;
		for (k = 16; k > 0; k >>= 1)
			v += __shfl_down_sync(0xffffffffu, v, k);
		if (lane == 0 && part)	/* ANN_DET_SUM: summed in block order below */
			part[blockIdx.x] = v;
		else if (lane == 0)
			atomicAdd(acc, v);
	}
}

/* *acc += part[0] + part[1] + ... in block order (one thread: a few hundred adds). */
__global__ static void sumsq_part_reduce_kernel(double *acc, const double *part,
	int np)
{
	double a = *acc;
	int p;

	for (p = 0; p < np; p++)
		a += part[p];
	*acc = a;
}

/* ANN_SUMSQ_GS=0: back to the one-block-per-256-values sumsq_kernel. */
static int sumsq_gs_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_SUMSQ_GS");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

extern "C" int gpu_grad_sumsq(const ann_real *d_x, int n, double *out)
{
	OpScope _op(OP_ADAM);
	static double *s_acc;
	static double *s_part;	/* ANN_DET_SUM: one partial per block */

	if (!d_x || n <= 0 || !out || !gpu_available())
		return -1;
	if (!s_acc && cudaMalloc((void **)&s_acc, sizeof(double)) != cudaSuccess) {
		s_acc = NULL;
		return -1;
	}
	if (cudaMemsetAsync(s_acc, 0, sizeof(double), g_st) != cudaSuccess)
		return -1;
	if (sumsq_gs_on() && ((uintptr_t)d_x & 15) == 0) {
		static int s_blocks;
		int nb;

		if (!s_blocks) {
			int dev = 0;
			int sms = 0;

			cudaGetDevice(&dev);
			if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
			    dev) != cudaSuccess || sms <= 0)
				sms = 32;
			s_blocks = sms * 8;
		}
		nb = (int)(((size_t)n / 4 + 255) / 256);
		if (nb > s_blocks)
			nb = s_blocks;
		if (nb < 1)
			nb = 1;
		/*
		** ANN_DET_SUM: the clip norm in a fixed order too, so the clip
		** scale (and every clipped gradient) cannot change its last bit
		** from run to run.
		*/
		if (det_sum_on() && nb > 1 && !s_part
		    && cudaMalloc((void **)&s_part, (size_t)s_blocks
		    * sizeof(double)) != cudaSuccess)
			s_part = NULL;
		sumsq_gs_kernel<<<nb, 256, 0, g_st>>>(d_x, (size_t)n, s_acc,
		    det_sum_on() && nb > 1 ? s_part : NULL);
		if (det_sum_on() && nb > 1 && s_part)
			sumsq_part_reduce_kernel<<<1, 1, 0, g_st>>>(s_acc, s_part, nb);
	} else
		sumsq_kernel<<<(n + 255) / 256, 256, 0, g_st>>>(d_x, n, s_acc);
	if (cudaGetLastError() != cudaSuccess)
		return -1;
	return cudaMemcpy(out, s_acc, sizeof(double), cudaMemcpyDeviceToHost)
	    == cudaSuccess ? 0 : -1;
}

__global__ static void adam_kernel(ann_real *param, ann_real *grad, ann_real *m,
	ann_real *v, int n, ann_real lr, ann_real beta1, ann_real beta2, ann_real eps,
	ann_real bc1, ann_real bc2, ann_real gscale, ann_real wd)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	ann_real g;
	ann_real mhat;
	ann_real vhat;

	if (i >= n)
		return;
	g = grad[i] * gscale;
	m[i] = beta1 * m[i] + (1.0f - beta1) * g;
	v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
	mhat = m[i] / bc1;
	vhat = v[i] / bc2;
	param[i] -= lr * mhat / (sqrtf(vhat) + eps);
	param[i] -= lr * wd * param[i];
	grad[i] = 0.0f;
}

__global__ static void adam_kernel_bf16(ann_real *param, __nv_bfloat16 *param_bf16,
	ann_real *grad, ann_real *m, ann_real *v, int n, ann_real lr,
	ann_real beta1, ann_real beta2, ann_real eps, ann_real bc1, ann_real bc2,
	ann_real gscale, ann_real wd)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	ann_real g;
	ann_real mhat;
	ann_real vhat;
	ann_real p;

	if (i >= n)
		return;
	g = grad[i] * gscale;
	m[i] = beta1 * m[i] + (1.0f - beta1) * g;
	v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
	mhat = m[i] / bc1;
	vhat = v[i] / bc2;
	p = param[i] - lr * mhat / (sqrtf(vhat) + eps);
	p -= lr * wd * p;
	param[i] = p;
	param_bf16[i] = __float2bfloat16(p);
	grad[i] = 0.0f;
}

/*
** A3: the same per-element update as adam_kernel_bf16, four values per thread
** with float4 loads/stores and one 8-byte BF16 store. The statements are the
** kernel's own, in the same order, so every value is bit-identical; only the
** memory transactions get wider (34 bytes per parameter, bandwidth bound).
*/
__device__ __forceinline__ static ann_real adam_bf16_elem(ann_real p,
	ann_real *gr, ann_real *mm, ann_real *vv, ann_real lr, ann_real beta1,
	ann_real beta2, ann_real eps, ann_real bc1, ann_real bc2,
	ann_real gscale, ann_real wd)
{
	ann_real g;
	ann_real mhat;
	ann_real vhat;

	/*
	** The moment updates spell out the contraction nvcc picks for
	** adam_kernel_bf16 (fma(b1, m, (1-b1) g) and fma(g, (1-b2) g, b2 v));
	** left to itself it fuses the other product here and m drifts by an ulp.
	*/
	g = *gr * gscale;
	*mm = __fmaf_rn(beta1, *mm, __fmul_rn(1.0f - beta1, g));
	*vv = __fmaf_rn(g, __fmul_rn(1.0f - beta2, g), __fmul_rn(beta2, *vv));
	mhat = *mm / bc1;
	vhat = *vv / bc2;
	p = p - lr * mhat / (sqrtf(vhat) + eps);
	p -= lr * wd * p;
	*gr = 0.0f;
	return p;
}

__global__ static void adam_kernel_bf16_v4(float4 *param, uint2 *param_bf16,
	float4 *grad, float4 *m, float4 *v, int n4, ann_real lr,
	ann_real beta1, ann_real beta2, ann_real eps, ann_real bc1, ann_real bc2,
	ann_real gscale, ann_real wd)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float4 p;
	float4 g;
	float4 mm;
	float4 vv;
	__nv_bfloat162 lo;
	__nv_bfloat162 hi;
	uint2 out;

	if (i >= n4)
		return;
	p = param[i];
	g = grad[i];
	mm = m[i];
	vv = v[i];
	p.x = adam_bf16_elem(p.x, &g.x, &mm.x, &vv.x, lr, beta1, beta2, eps,
	    bc1, bc2, gscale, wd);
	p.y = adam_bf16_elem(p.y, &g.y, &mm.y, &vv.y, lr, beta1, beta2, eps,
	    bc1, bc2, gscale, wd);
	p.z = adam_bf16_elem(p.z, &g.z, &mm.z, &vv.z, lr, beta1, beta2, eps,
	    bc1, bc2, gscale, wd);
	p.w = adam_bf16_elem(p.w, &g.w, &mm.w, &vv.w, lr, beta1, beta2, eps,
	    bc1, bc2, gscale, wd);
	m[i] = mm;
	v[i] = vv;
	param[i] = p;
	lo.x = __float2bfloat16(p.x);
	lo.y = __float2bfloat16(p.y);
	hi.x = __float2bfloat16(p.z);
	hi.y = __float2bfloat16(p.w);
	out.x = *(unsigned int *)&lo;
	out.y = *(unsigned int *)&hi;
	param_bf16[i] = out;
	grad[i] = g;
}

/* ANN_ADAM_V4=0: back to the scalar adam_kernel_bf16. */
static int adam_v4_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_ADAM_V4");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

extern "C" int gpu_adam_step_dev_bf16(ann_real *d_p, void *d_p_bf16,
	ann_real *d_g, ann_real *d_m, ann_real *d_v, int n, double lr,
	double beta1, double beta2, double eps, int t, double grad_scale)
{
	OpScope _op(OP_ADAM);
	double bc1;
	double bc2;

	if (!d_p || !d_g || !d_m || !d_v || n <= 0 || lr <= 0.0 || t < 1)
		return -1;
	if (!gpu_available())
		return -1;
	bc1 = 1.0 - pow(beta1, (double)t);
	bc2 = 1.0 - pow(beta2, (double)t);
	if (bc1 <= 0.0 || bc2 <= 0.0)
		return -1;
#if CUDART_VERSION >= 12000 && defined(ANN_CUBLAS)
	if (g_fp8_fwd || g_fp8_dgrad)
		fp8_wt_invalidate(NULL);
#endif
	if (d_p_bf16 && g_compute_prec == GPU_PREC_BF16 && adam_v4_on()
	    && (((uintptr_t)d_p | (uintptr_t)d_g | (uintptr_t)d_m
	    | (uintptr_t)d_v) & 15) == 0 && ((uintptr_t)d_p_bf16 & 7) == 0) {
		int n4 = n >> 2;
		int done = n4 << 2;

		if (n4 > 0)
			adam_kernel_bf16_v4<<<(n4 + 255) / 256, 256, 0, g_st>>>(
			    (float4 *)d_p, (uint2 *)d_p_bf16, (float4 *)d_g,
			    (float4 *)d_m, (float4 *)d_v, n4, lr, beta1, beta2,
			    eps, bc1, bc2, (ann_real)grad_scale, g_adam_wd);
		/* The n % 4 tail (none at the production sizes) stays scalar. */
		if (done < n)
			adam_kernel_bf16<<<1, 256, 0, g_st>>>(d_p + done,
			    (__nv_bfloat16 *)d_p_bf16 + done, d_g + done,
			    d_m + done, d_v + done, n - done, lr, beta1, beta2,
			    eps, bc1, bc2, (ann_real)grad_scale, g_adam_wd);
	} else if (d_p_bf16 && g_compute_prec == GPU_PREC_BF16) {
		adam_kernel_bf16<<<(n + 255) / 256, 256, 0, g_st>>>(d_p,
		    (__nv_bfloat16 *)d_p_bf16, d_g, d_m, d_v, n, lr, beta1,
		    beta2, eps, bc1, bc2, (ann_real)grad_scale, g_adam_wd);
	} else
		adam_kernel<<<(n + 255) / 256, 256, 0, g_st>>>(d_p, d_g, d_m, d_v, n, lr,
		    beta1, beta2, eps, bc1, bc2, (ann_real)grad_scale, g_adam_wd);
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ static void axpy_kernel(ann_real *y, const ann_real *x, ann_real a, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		y[i] += a * x[i];
}

__global__ static void add_kernel(ann_real *y, const ann_real *x, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		y[i] += x[i];
}

__global__ static void add2_kernel(ann_real *dst, const ann_real *a,
	const ann_real *b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		dst[i] = a[i] + b[i];
}

static size_t mat_nbytes(const mat *m)
{
	return (size_t)m->rows * (size_t)m->cols * sizeof(ann_real);
}

static int mat_ensure_dev_auth(mat *m)
{
	if (!m)
		return -1;
	if (mat_device_alloc(m) != 0)
		return -1;
	if (m->on_device == MAT_DEV_HOST) {
		if (!m->data
		    || gpu_memcpy_h2d(m->d_data, m->data, mat_nbytes(m)) != 0)
			return -1;
		m->on_device = MAT_DEV_SYNCED;
	}
	return 0;
}

extern "C" int gpu_mat_copy(mat *dst, const mat *src)
{
	OpScope _op(OP_RESID);
	size_t n;

	if (!dst || !src)
		return -1;
	if (dst->rows != src->rows || dst->cols != src->cols)
		return -1;
	if (!gpu_available())
		return -1;
	n = mat_nbytes(src);
	if (mat_ensure_dev_auth((mat *)src) != 0)
		return -1;
	if (mat_device_alloc(dst) != 0)
		return -1;
	if (gpu_memcpy_d2d(dst->d_data, src->d_data, n) != 0)
		return -1;
	dst->on_device = MAT_DEV_DEVICE;
	return launch_done();
}

extern "C" int gpu_mat_add(mat *dst, const mat *src)
{
	OpScope _op(OP_RESID);
	int n;
	int bs;

	if (!dst || !src)
		return -1;
	if (dst->rows != src->rows || dst->cols != src->cols)
		return -1;
	if (!gpu_available())
		return -1;
	if (mat_ensure_dev_auth(dst) != 0 || mat_ensure_dev_auth((mat *)src) != 0)
		return -1;
	n = dst->rows * dst->cols;
	bs = 256;
	add_kernel<<<(n + bs - 1) / bs, bs, 0, g_st>>>(dst->d_data, src->d_data, n);
	if (launch_done() != 0)
		return -1;
	dst->on_device = MAT_DEV_DEVICE;
	return 0;
}

extern "C" int gpu_mat_add2(mat *dst, const mat *a, const mat *b)
{
	OpScope _op(OP_RESID);
	int n;
	int bs;

	if (!dst || !a || !b)
		return -1;
	if (dst->rows != a->rows || dst->cols != a->cols
	    || b->rows != a->rows || b->cols != a->cols)
		return -1;
	if (!gpu_available())
		return -1;
	if (mat_ensure_dev_auth((mat *)a) != 0
	    || mat_ensure_dev_auth((mat *)b) != 0)
		return -1;
	if (mat_device_alloc(dst) != 0)
		return -1;
	n = dst->rows * dst->cols;
	bs = 256;
	add2_kernel<<<(n + bs - 1) / bs, bs, 0, g_st>>>(dst->d_data, a->d_data, b->d_data,
	    n);
	if (launch_done() != 0)
		return -1;
	dst->on_device = MAT_DEV_DEVICE;
	return 0;
}

extern "C" int gpu_mat_axpy(mat *y, double a, const mat *x)
{
	OpScope _op(OP_RESID);
	int n;
	int bs;

	if (!y || !x)
		return -1;
	if (y->rows != x->rows || y->cols != x->cols)
		return -1;
	if (!gpu_available())
		return -1;
	if (mat_ensure_dev_auth(y) != 0 || mat_ensure_dev_auth((mat *)x) != 0)
		return -1;
	n = y->rows * y->cols;
	bs = 256;
	axpy_kernel<<<(n + bs - 1) / bs, bs, 0, g_st>>>(y->d_data, x->d_data,
	    (ann_real)a, n);
	if (launch_done() != 0)
		return -1;
	y->on_device = MAT_DEV_DEVICE;
	return 0;
}

__global__ static void embed_lookup_kernel(ann_real *out, const ann_real *table,
	const int *ids, int n, int d)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t;
	int j;
	int id;

	if (i >= n * d)
		return;
	t = i / d;
	j = i % d;
	id = ids[t];
	out[i] = table[id * d + j];
}

__global__ static void embed_scatter_kernel(ann_real *dtable, const ann_real *dout,
	const int *ids, int n, int d)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t;
	int j;
	int id;

	if (i >= n * d)
		return;
	t = i / d;
	j = i % d;
	id = ids[t];
	atomicAdd(&dtable[id * d + j], dout[i]);
}

/*
** B8: host index vectors (token ids, CE targets) staged through a pinned
** buffer into a persistent device buffer, async on the main stream, instead
** of cudaMalloc + a pageable copy + cudaFree per call (cudaFree and the
** pageable copy synchronise with the device at every microbatch boundary).
** The event guards the pinned buffer: the host refills it only once the
** previous copy out of it has completed, which in training happened a
** microbatch earlier. The device buffer is reused in stream order. Not
** while capturing a graph. ANN_MB_NOSYNC=0 restores the old paths.
*/
typedef struct {
	void		*dev;
	void		*pin;
	size_t		cap;
	cudaEvent_t	ev;
	int		ev_ok;
	int		pending;
} h2d_stage;

static int mb_nosync_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_MB_NOSYNC");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

/* Device copy of host[0..bytes) on the compute stream; NULL = use the old path. */
static void *h2d_stage_put(h2d_stage *s, const void *host, size_t bytes)
{
	if (!host || bytes == 0 || !mb_nosync_on() || stream_is_capturing())
		return NULL;
	if (s->pending) {
		if (cudaEventSynchronize(s->ev) != cudaSuccess)
			return NULL;
		s->pending = 0;
	}
	if (!s->ev_ok) {
		if (cudaEventCreateWithFlags(&s->ev, cudaEventDisableTiming)
		    != cudaSuccess)
			return NULL;
		s->ev_ok = 1;
	}
	if (s->cap < bytes) {
		if (s->dev)
			cudaFree(s->dev);
		if (s->pin)
			cudaFreeHost(s->pin);
		s->dev = NULL;
		s->pin = NULL;
		s->cap = 0;
		if (cudaMalloc(&s->dev, bytes) != cudaSuccess) {
			s->dev = NULL;
			return NULL;
		}
		if (cudaHostAlloc(&s->pin, bytes, cudaHostAllocDefault)
		    != cudaSuccess) {
			cudaFree(s->dev);
			s->dev = NULL;
			s->pin = NULL;
			return NULL;
		}
		s->cap = bytes;
	}
	memcpy(s->pin, host, bytes);
	if (cudaMemcpyAsync(s->dev, s->pin, bytes, cudaMemcpyHostToDevice, g_st)
	    != cudaSuccess)
		return NULL;
	if (cudaEventRecord(s->ev, g_st) == cudaSuccess)
		s->pending = 1;
	else if (cudaStreamSynchronize(g_st) != cudaSuccess)
		return NULL;
	return s->dev;
}

/*
** ANN_DET_SUM: the scatter in token order. next[t] = the next token with
** the same id (-1: none); first[t] = no earlier token has it. One thread
** per (first occurrence, column) then adds that id's rows in token order:
** the additions of the atomics, in a fixed order (a repeated token's rows
** used to land in whatever order the atomics ran).
*/
__global__ static void embed_chain_kernel(const int *ids, int n, int *next,
	unsigned char *first)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int id;
	int u;
	int nx = -1;
	unsigned char f = 1;

	if (t >= n)
		return;
	id = ids[t];
	for (u = t + 1; u < n; u++)
		if (ids[u] == id) {
			nx = u;
			break;
		}
	for (u = 0; u < t; u++)
		if (ids[u] == id) {
			f = 0;
			break;
		}
	next[t] = nx;
	first[t] = f;
}

__global__ static void embed_scatter_det_kernel(ann_real *dtable,
	const ann_real *dout, const int *ids, const int *next,
	const unsigned char *first, int n, int d)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t;
	int j;
	int u;
	ann_real *dst;
	ann_real a;

	if (i >= n * d)
		return;
	t = i / d;
	j = i % d;
	if (!first[t])
		return;
	dst = &dtable[(size_t)ids[t] * (size_t)d + (size_t)j];
	a = *dst;
	for (u = t; u >= 0; u = next[u])
		a += dout[(size_t)u * (size_t)d + (size_t)j];
	*dst = a;
}

/* Chain scratch of the deterministic scatter (grow-only, under its mutex). */
static pthread_mutex_t g_emb_det_mu = PTHREAD_MUTEX_INITIALIZER;
static void *g_emb_next;
static size_t g_emb_next_cap;
static void *g_emb_first;
static size_t g_emb_first_cap;

static void embed_scatter_launch(ann_real *dtable, const ann_real *dout,
	const int *d_ids, int n, int d)
{
	int nd = n * d;

	if (det_sum_on() && n > 1) {
		pthread_mutex_lock(&g_emb_det_mu);
		if (ensure_half_buf(&g_emb_next, &g_emb_next_cap,
		    (size_t)n * sizeof(int)) == 0
		    && ensure_half_buf(&g_emb_first, &g_emb_first_cap,
		    (size_t)n) == 0) {
			embed_chain_kernel<<<(n + 255) / 256, 256, 0, g_st>>>(d_ids,
			    n, (int *)g_emb_next, (unsigned char *)g_emb_first);
			embed_scatter_det_kernel<<<(nd + 255) / 256, 256, 0, g_st>>>(
			    dtable, dout, d_ids, (const int *)g_emb_next,
			    (const unsigned char *)g_emb_first, n, d);
			pthread_mutex_unlock(&g_emb_det_mu);
			return;
		}
		pthread_mutex_unlock(&g_emb_det_mu);
	}
	embed_scatter_kernel<<<(nd + 255) / 256, 256, 0, g_st>>>(dtable, dout,
	    d_ids, n, d);
}

/*
** Staged token ids (gpu_ids_stage): one fixed device buffer that a captured
** graph reads, refilled on the compute stream before every microbatch from
** a two-slot page-locked ring. The host waits for a slot's previous copy
** (two microbatches back) before overwriting it, never for the compute.
*/
static int *g_ids_dev;
static size_t g_ids_dev_cap;
static int *g_ids_pin[2];
static size_t g_ids_pin_cap;
static cudaEvent_t g_ids_ev[2];
static int g_ids_ev_used[2];
static int g_ids_k;
static const int *g_ids_key;
static const int *g_ids_alias;
static int g_ids_n;
static int *g_tgt_dev;
static size_t g_tgt_dev_cap;

extern "C" int gpu_ids_stage(const int *ids, const int *alias, int n)
{
	size_t ib;
	int k;

	gpu_ids_unstage();
	if (!ids || n <= 0 || !gpu_available() || capturing_now())
		return -1;
	ib = (size_t)n * sizeof(int);
	if (g_ids_dev_cap < ib) {
		if (g_ids_dev)
			cudaFree(g_ids_dev);
		g_ids_dev = NULL;
		g_ids_dev_cap = 0;
		if (cudaMalloc((void **)&g_ids_dev, ib) != cudaSuccess) {
			g_ids_dev = NULL;
			return -1;
		}
		g_ids_dev_cap = ib;
	}
	if (g_ids_pin_cap < ib) {
		for (k = 0; k < 2; k++) {
			if (g_ids_ev_used[k])
				(void)cudaEventSynchronize(g_ids_ev[k]);
			if (g_ids_pin[k])
				cudaFreeHost(g_ids_pin[k]);
			g_ids_pin[k] = NULL;
		}
		g_ids_pin_cap = 0;
		for (k = 0; k < 2; k++)
			if (cudaHostAlloc((void **)&g_ids_pin[k], ib,
			    cudaHostAllocDefault) != cudaSuccess) {
				g_ids_pin[k] = NULL;
				return -1;
			}
		g_ids_pin_cap = ib;
	}
	k = g_ids_k;
	g_ids_k ^= 1;
	if (!g_ids_ev[k] && cudaEventCreateWithFlags(&g_ids_ev[k],
	    cudaEventDisableTiming) != cudaSuccess) {
		g_ids_ev[k] = NULL;
		return -1;
	}
	if (g_ids_ev_used[k] && cudaEventSynchronize(g_ids_ev[k]) != cudaSuccess)
		return -1;
	memcpy(g_ids_pin[k], ids, ib);
	if (cudaMemcpyAsync(g_ids_dev, g_ids_pin[k], ib, cudaMemcpyHostToDevice,
	    g_st) != cudaSuccess
	    || cudaEventRecord(g_ids_ev[k], g_st) != cudaSuccess)
		return -1;
	g_ids_ev_used[k] = 1;
	g_ids_key = ids;
	g_ids_alias = alias;
	g_ids_n = n;
	return 0;
}

extern "C" void gpu_ids_unstage(void)
{
	g_ids_key = NULL;
	g_ids_alias = NULL;
	g_ids_n = 0;
}

/* The staged device copy of ids (same host array or its alias), else NULL. */
static const int *ids_staged(const int *ids, int n)
{
	if (!ids || n <= 0 || n != g_ids_n || !g_ids_dev)
		return NULL;
	return ids == g_ids_key || ids == g_ids_alias ? g_ids_dev : NULL;
}

__global__ static void next_token_targets_kernel(int *tgt, const int *ids,
	int B, int T)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int b;

	if (r >= B * (T - 1))
		return;
	b = r / (T - 1);
	tgt[r] = ids[b * T + r % (T - 1) + 1];
}

/*
** Row b*(T-1)+t predicts ids[b*T+t+1]: the packed CE targets built on the
** device from the staged ids (no loss mask, no varlen: those are host data).
*/
extern "C" const int *gpu_ce_targets_staged(const int *ids, int B, int T)
{
	const int *d_ids;
	size_t nb;
	int rows;

	if (B <= 0 || T < 2 || !(d_ids = ids_staged(ids, B * T)))
		return NULL;
	rows = B * (T - 1);
	nb = (size_t)rows * sizeof(int);
	if (g_tgt_dev_cap < nb) {
		if (g_tgt_dev)
			cudaFree(g_tgt_dev);
		g_tgt_dev = NULL;
		g_tgt_dev_cap = 0;
		if (cudaMalloc((void **)&g_tgt_dev, nb) != cudaSuccess) {
			g_tgt_dev = NULL;
			return NULL;
		}
		g_tgt_dev_cap = nb;
	}
	next_token_targets_kernel<<<(rows + 255) / 256, 256, 0, g_st>>>(
	    g_tgt_dev, d_ids, B, T);
	return cudaGetLastError() == cudaSuccess ? g_tgt_dev : NULL;
}

extern "C" int gpu_embed_lookup(mat *out, const mat *table, const int *ids, int n)
{
	OpScope _op(OP_EMBED);
	static h2d_stage s_ids;
	int *d_ids;
	const int *staged;
	int d;
	int nd;
	size_t ib;

	if (!out || !table || !ids || n <= 0 || !gpu_available())
		return -1;
	if (out->rows != n || out->cols != table->cols)
		return -1;
	d = table->cols;
	if (mat_ensure_dev_auth((mat *)table) != 0 || mat_device_alloc(out) != 0)
		return -1;
	if ((staged = ids_staged(ids, n)) != NULL) {
		nd = n * d;
		embed_lookup_kernel<<<(nd + 255) / 256, 256, 0, g_st>>>(
		    out->d_data, table->d_data, staged, n, d);
		if (launch_done() != 0)
			return -1;
		out->on_device = MAT_DEV_DEVICE;
		return 0;
	}
	ib = (size_t)n * sizeof(int);
	d_ids = (int *)h2d_stage_put(&s_ids, ids, ib);
	if (d_ids) {
		nd = n * d;
		embed_lookup_kernel<<<(nd + 255) / 256, 256, 0, g_st>>>(out->d_data,
		    table->d_data, d_ids, n, d);
		if (launch_done() != 0)
			return -1;
		out->on_device = MAT_DEV_DEVICE;
		return 0;
	}
	d_ids = (int *)gpu_malloc(ib);
	if (!d_ids)
		return -1;
	/* R8: async H2D on copy stream when available. */
	if (g_copy_stream) {
		if (cudaMemcpyAsync(d_ids, ids, ib, cudaMemcpyHostToDevice,
		    g_copy_stream) != cudaSuccess
		    || cudaStreamSynchronize(g_copy_stream) != cudaSuccess) {
			gpu_free(d_ids);
			return -1;
		}
	} else if (gpu_memcpy_h2d(d_ids, ids, ib) != 0) {
		gpu_free(d_ids);
		return -1;
	}
	nd = n * d;
	embed_lookup_kernel<<<(nd + 255) / 256, 256, 0, g_st>>>(out->d_data, table->d_data,
	    d_ids, n, d);
	gpu_free(d_ids);
	if (launch_done() != 0)
		return -1;
	out->on_device = MAT_DEV_DEVICE;
	return 0;
}

extern "C" int gpu_embed_scatter(mat *dtable, const mat *dout, const int *ids,
	int n)
{
	OpScope _op(OP_EMBED);
	static h2d_stage s_ids;
	int *d_ids;
	const int *staged;
	int d;
	size_t ib;

	if (!dtable || !dout || !ids || n <= 0 || !gpu_available())
		return -1;
	if (dout->rows != n || dout->cols != dtable->cols)
		return -1;
	d = dtable->cols;
	if (mat_ensure_dev_auth((mat *)dout) != 0 || mat_device_alloc(dtable) != 0)
		return -1;
	if (dtable->on_device == MAT_DEV_HOST) {
		if (gpu_memcpy_h2d(dtable->d_data, dtable->data,
		    mat_nbytes(dtable)) != 0)
			return -1;
		dtable->on_device = MAT_DEV_SYNCED;
	}
	if ((staged = ids_staged(ids, n)) != NULL) {
		embed_scatter_launch(dtable->d_data, dout->d_data, staged, n, d);
		if (launch_done() != 0)
			return -1;
		dtable->on_device = MAT_DEV_DEVICE;
		return 0;
	}
	ib = (size_t)n * sizeof(int);
	d_ids = (int *)h2d_stage_put(&s_ids, ids, ib);
	if (d_ids) {
		embed_scatter_launch(dtable->d_data, dout->d_data, d_ids, n, d);
		if (launch_done() != 0)
			return -1;
		dtable->on_device = MAT_DEV_DEVICE;
		return 0;
	}
	d_ids = (int *)gpu_malloc(ib);
	if (!d_ids)
		return -1;
	if (gpu_memcpy_h2d(d_ids, ids, ib) != 0) {
		gpu_free(d_ids);
		return -1;
	}
	embed_scatter_launch(dtable->d_data, dout->d_data, d_ids, n, d);
	gpu_free(d_ids);
	if (launch_done() != 0)
		return -1;
	dtable->on_device = MAT_DEV_DEVICE;
	return 0;
}

extern "C" void *gpu_host_alloc(size_t nbytes)
{
	void *p;

	if (nbytes == 0 || !gpu_available())
		return NULL;
	if (cudaHostAlloc(&p, nbytes, cudaHostAllocDefault) != cudaSuccess)
		return NULL;
	return p;
}

extern "C" void gpu_host_free(void *p)
{
	if (p)
		cudaFreeHost(p);
}

extern "C" void *gpu_copy_stream(void)
{
	if (!gpu_available())
		return NULL;
	if (!g_copy_stream
	    && cudaStreamCreateWithFlags(&g_copy_stream, cudaStreamNonBlocking)
	    != cudaSuccess)
		return NULL;
	return (void *)g_copy_stream;
}

/*
** S1/S2/S3: CE delta, packed posenc, column-sum bias grads (device-first).
*/
__global__ static void ce_onehot_ce_kernel(ann_real *probs, const int *tgt,
	int rows, int V, float *ce_sum)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int t;
	float p;

	if (r >= rows)
		return;
	t = tgt[r];
	if (t < 0 || t >= V) {
		/* Masked row (tgt -1): no loss and no gradient. */
		for (t = 0; t < V; t++)
			probs[(size_t)r * (size_t)V + (size_t)t] = 0.0f;
		return;
	}
	p = probs[(size_t)r * (size_t)V + (size_t)t];
	if (p < 1e-12f)
		p = 1e-12f;
	atomicAdd(ce_sum, -logf(p));
	probs[(size_t)r * (size_t)V + (size_t)t] -= 1.0f;
}

__global__ static void posenc_packed_kernel(ann_real *x, int B, int T, int D)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int n = B * T * D;
	int tok;
	int t;
	int d;
	int pair;
	float log_base;
	float angle;

	if (i >= n)
		return;
	tok = i / D;
	d = i % D;
	t = tok % T;
	pair = d / 2;
	log_base = -logf(10000.0f) / (float)D;
	angle = (float)t * expf((2.0f * (float)pair) * log_base);
	if ((d & 1) == 0)
		x[i] += sinf(angle);
	else
		x[i] += cosf(angle);
}

/*
** Head loss in one pass (BF16 training): for each row of FP32 logits,
** softmax, the CE term of the target, delta = p - onehot written as BF16 in
** place over the row's own first half (row stride 2 V in BF16 elements),
** and the db_head column sums of the FP32 delta. One block of ew_block_for(V)
** threads per row at a time, each thread owning columns tid + k * blockDim
** like softmax_rows_kernel, with the same max / sum reductions: the
** probabilities, the CE and the delta are bitwise those of softmax_rows_kernel
** + ce_onehot_ce_kernel. The row lives in registers between its single load
** and the BF16 store (the store overwrites FP32 values of the same row, read
** before the reductions' barriers). Blocks stride over the rows and keep
** their column sums in registers: one atomic per column and block at the end.
** bias (B2, may be NULL): the head bias added at the load, row[j] + bias[j]
** in FP32 exactly as bias_add_kernel did in a separate pass before.
*/
template <int NV>
__global__ static void __launch_bounds__(EW_BLOCK)
ce_softmax_delta_bf16_kernel(ann_real *x, const int *tgt, int rows, int V,
	const ann_real *bias, ann_real *db, float *ce_sum, ann_real *part)
{
	__shared__ ann_real sh[EW_BLOCK];
	float e[NV];
	float cs[NV];
	int k;
	int r;

#pragma unroll
	for (k = 0; k < NV; k++)
		cs[k] = 0.0f;
	for (r = blockIdx.x; r < rows; r += gridDim.x) {
		ann_real *row = x + (size_t)r * (size_t)V;
		__nv_bfloat16 *drow = (__nv_bfloat16 *)row;
		const int t = tgt[r];
		ann_real m = -INFINITY;
		ann_real s = 0.0f;
		ann_real inv;

		if (t < 0 || t >= V) {
			/* Masked row (tgt -1): no loss and no gradient. */
#pragma unroll
			for (k = 0; k < NV; k++) {
				const int j = threadIdx.x + k * blockDim.x;

				if (j < V)
					drow[j] = __float2bfloat16_rn(0.0f);
			}
			continue;	/* t is uniform: the whole block skips */
		}
#pragma unroll
		for (k = 0; k < NV; k++) {
			const int j = threadIdx.x + k * blockDim.x;

			e[k] = j < V ? (bias ? row[j] + bias[j] : row[j])
			    : -INFINITY;
			if (j < V && e[k] > m)
				m = e[k];
		}
		m = ew_block_max(m, sh);
#pragma unroll
		for (k = 0; k < NV; k++) {
			const int j = threadIdx.x + k * blockDim.x;

			if (j < V) {
				e[k] = expf(e[k] - m);
				s += e[k];
			}
		}
		s = ew_block_sum(s, sh);
		inv = 1.0f / s;
#pragma unroll
		for (k = 0; k < NV; k++) {
			const int j = threadIdx.x + k * blockDim.x;
			float d;

			if (j >= V)
				continue;
			d = e[k] * inv;
			if (j == t) {
				float p = d < 1e-12f ? 1e-12f : d;

				atomicAdd(ce_sum, -logf(p));
				d -= 1.0f;
			}
			cs[k] += d;
			drow[j] = __float2bfloat16_rn(d);
		}
	}
#pragma unroll
	for (k = 0; k < NV; k++) {
		const int j = threadIdx.x + k * blockDim.x;

		if (j < V && part)	/* ANN_DET_SUM: this block's row, reduced in a fixed order */
			part[(size_t)blockIdx.x * (size_t)V + (size_t)j] = cs[k];
		else if (j < V)
			atomicAdd(&db[j], cs[k]);
	}
}

/* Launch the instance for nv = ceil(V / bs); -1 = no instance that large. */
static int ce_softmax_delta_bf16_launch(ann_real *x, const int *tgt,
	int rows, int V, const ann_real *bias, ann_real *db, float *ce_sum)
{
	static int n_sm;
	const int bs = ew_block_for(V);
	const int nv = (V + bs - 1) / bs;
	ann_real *part;
	int grid;

	if (n_sm <= 0) {
		int dev = 0;

		if (cudaGetDevice(&dev) != cudaSuccess
		    || cudaDeviceGetAttribute(&n_sm,
		    cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
		    || n_sm <= 0)
			n_sm = 1;
	}
	/*
	** About 150 registers per thread at nv = 64: one block per SM. Two
	** waves of blocks so a slow SM does not hold the tail; each extra
	** block costs V atomics on db.
	*/
	grid = 2 * n_sm;
	if (grid > rows)
		grid = rows;
	part = grid > 1 ? cs_part((size_t)grid * (size_t)V) : NULL;
#define CE_BF16_CASE(NV) \
	case NV: \
		ce_softmax_delta_bf16_kernel<NV><<<grid, bs, 0, g_st>>>(x, tgt, rows, V, \
		    bias, db, ce_sum, part); \
		if (part) \
			colsum_reduce_launch(db, part, grid, V, V); \
		return 0;
	switch (nv <= 1 ? 1 : nv <= 2 ? 2 : nv <= 4 ? 4 : nv <= 8 ? 8
	    : nv <= 16 ? 16 : nv <= 32 ? 32 : nv <= 64 ? 64 : 0) {
	CE_BF16_CASE(1)
	CE_BF16_CASE(2)
	CE_BF16_CASE(4)
	CE_BF16_CASE(8)
	CE_BF16_CASE(16)
	CE_BF16_CASE(32)
	CE_BF16_CASE(64)
	default:
		return -1;
	}
#undef CE_BF16_CASE
}

/* acc[j] += x[j]: one FP32 add per element, as a host loop would do. */
__global__ static void vec_acc_kernel(ann_real *acc, const ann_real *x, int n)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < n)
		acc[j] += x[j];
}

/* Device CE sum + pinned host mirror (captured graph D2H updates host). */
static float *g_ce_dev;
static float *g_ce_host_pinned;
static int g_ce_last_rows;

/*
** fused = 0: probs holds the softmax rows; the one-hot is subtracted in
** place (FP32) and db is summed from it. fused = 1: probs holds the logits;
** ce_softmax_delta_bf16_kernel does the softmax as well and leaves the BF16
** delta in place (the caller registers the twin). bias (fused only, may be
** NULL): the head bias, not yet added to the logits.
*/
static int ce_delta_run(ann_real *probs, const int *tgt, int rows, int V,
	const ann_real *bias, ann_real *db_host, double *mean_ce, int fused,
	int *launched)
{
	ew_ref rp;
	ew_ref rt;
	ew_ref rb;
	static h2d_stage s_tgt;	/* B8: pinned staging of the targets */
	static ann_real *s_db_dev;
	static ann_real *s_db_host;
	static ann_real *s_dbg_dev;	/* graph mode: this microbatch's sums */
	static size_t s_dbg_cap;
	static size_t s_db_cap;
	size_t nb;
	size_t nbt;
	size_t need_db;
	int rc;
	int dn;
	int j;

	if (!probs || !tgt || !db_host || rows <= 0 || V <= 0 || !gpu_available())
		return -1;
	nb = (size_t)rows * (size_t)V * sizeof(ann_real);
	nbt = (size_t)rows * sizeof(int);
	need_db = (size_t)V * sizeof(ann_real);
	pthread_mutex_lock(&g_ew_mu);
	if (ew_acquire(&rp, 0, probs, nb, 1, 1) != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	rt.dev = ew_is_dev(tgt) ? NULL
	    : (ann_real *)h2d_stage_put(&s_tgt, tgt, nbt);
	if (!rt.dev && ew_acquire(&rt, 1, tgt, nbt, 1, 0) != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	rb.dev = NULL;
	if (fused && bias && ew_acquire_param(&rb, 2, bias, need_db) != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	if (!g_ce_dev
	    && cudaMalloc((void **)&g_ce_dev, sizeof(float)) != cudaSuccess) {
		g_ce_dev = NULL;
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	if (!g_ce_host_pinned
	    && cudaHostAlloc((void **)&g_ce_host_pinned, sizeof(float),
	    cudaHostAllocDefault) != cudaSuccess) {
		g_ce_host_pinned = NULL;
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	/*
	** Graph mode: db_head accumulates on the device (flushed into the
	** host vector by gpu_device_sync), so the microbatch has no host step
	** left and can be captured. Each microbatch's column sums still go to
	** their own zeroed buffer first and are then added to the accumulator,
	** element by element in microbatch order: the same FP32 additions, in
	** the same order, as the host loop below (summing straight into the
	** accumulator would reassociate them). The mean CE is read from the
	** pinned mirror after a stream sync, only when asked for and never
	** inside a capture (gpu_ce_fetch_mean after the replay).
	*/
	if (g_graph_mode) {
		colsum_acc *acc = colsum_acc_get(db_host, V);

		if (!acc)
			goto ce_fail;
		if (s_dbg_cap < need_db) {
			if (s_dbg_dev)
				cudaFree(s_dbg_dev);
			s_dbg_dev = NULL;
			s_dbg_cap = 0;
			if (cudaMalloc((void **)&s_dbg_dev, need_db) != cudaSuccess) {
				s_dbg_dev = NULL;
				goto ce_fail;
			}
			s_dbg_cap = need_db;
		}
		cudaMemsetAsync(g_ce_dev, 0, sizeof(float), g_st);
		cudaMemsetAsync(s_dbg_dev, 0, need_db, g_st);
		if (fused) {
#if CUDART_VERSION >= 11000
			/* The fused head loss (softmax + CE + BF16 delta + column sums) on the compute stream. */
			if (rp.dev != probs
			    || ce_softmax_delta_bf16_launch(rp.dev, (const int *)rt.dev,
			    rows, V, rb.dev, s_dbg_dev, g_ce_dev) != 0)
				goto ce_fail;
			*launched = 1;
			if (ew_launch_ok() != 0)
				goto ce_fail;
#else
			goto ce_fail;
#endif
		} else {
			ce_onehot_ce_kernel<<<(rows + EW_BLOCK - 1) / EW_BLOCK, EW_BLOCK, 0, g_st>>>(
			    rp.dev, (const int *)rt.dev, rows, V, g_ce_dev);
			if (ew_launch_ok() != 0)
				goto ce_fail;
			col_sum_tiled_run(s_dbg_dev, rp.dev, NULL, rows, V);
		}
		vec_acc_kernel<<<(V + EW_BLOCK - 1) / EW_BLOCK, EW_BLOCK, 0, g_st>>>(
		    acc->dev, s_dbg_dev, V);
		if (ew_launch_ok() != 0)
			goto ce_fail;
		acc->dirty = 1;
		if (cudaMemcpyAsync(g_ce_host_pinned, g_ce_dev, sizeof(float),
		    cudaMemcpyDeviceToHost, g_st) != cudaSuccess)
			goto ce_fail;
		g_ce_last_rows = rows;
		if (ew_release(&rp) < 0)
			goto ce_fail;
		if (mean_ce && !stream_is_capturing()) {
			if (cudaStreamSynchronize(g_st) != cudaSuccess)
				goto ce_fail;
			*mean_ce = (double)(*g_ce_host_pinned) / (double)rows;
		}
		pthread_mutex_unlock(&g_ew_mu);
		return 0;
	}
	if (s_db_cap < need_db) {
		if (s_db_dev)
			cudaFree(s_db_dev);
		s_db_dev = NULL;
		free(s_db_host);
		s_db_host = NULL;
		s_db_cap = 0;
		if (cudaMalloc((void **)&s_db_dev, need_db) != cudaSuccess) {
			s_db_dev = NULL;
			pthread_mutex_unlock(&g_ew_mu);
			return -1;
		}
		s_db_host = (ann_real *)malloc(need_db);
		if (!s_db_host) {
			cudaFree(s_db_dev);
			s_db_dev = NULL;
			pthread_mutex_unlock(&g_ew_mu);
			return -1;
		}
		s_db_cap = need_db;
	}
	cudaMemsetAsync(g_ce_dev, 0, sizeof(float), g_st);
	cudaMemsetAsync(s_db_dev, 0, need_db, g_st);
	if (fused) {
		if (rp.dev != probs
		    || ce_softmax_delta_bf16_launch(rp.dev, (const int *)rt.dev,
		    rows, V, rb.dev, s_db_dev, g_ce_dev) != 0)
			goto ce_fail;
		*launched = 1;
		if (ew_launch_ok() != 0)
			goto ce_fail;
	} else {
		ce_onehot_ce_kernel<<<(rows + EW_BLOCK - 1) / EW_BLOCK,
		    EW_BLOCK, 0, g_st>>>(rp.dev, (const int *)rt.dev, rows, V, g_ce_dev);
		if (ew_launch_ok() != 0)
			goto ce_fail;
		col_sum_tiled_run(s_db_dev, rp.dev, NULL, rows, V);
		if (ew_launch_ok() != 0)
			goto ce_fail;
	}
	/* Always mirror CE to pinned host so graph replay can report NLL. */
	if (cudaMemcpyAsync(g_ce_host_pinned, g_ce_dev, sizeof(float),
	    cudaMemcpyDeviceToHost, g_st) != cudaSuccess)
		goto ce_fail;
	g_ce_last_rows = rows;
	rc = ew_release(&rp);
	if (rc < 0)
		goto ce_fail;
	dn = rc;
	if (ew_finish_sync(dn) != 0)
		goto ce_fail;
	/*
	** During graph capture, skip host db accumulate (host loops are not
	** replayed). CE scalar still lands in pinned g_ce_host_pinned.
	*/
	if (!stream_is_capturing()) {
		if (cudaMemcpy(s_db_host, s_db_dev, need_db,
		    cudaMemcpyDeviceToHost) != cudaSuccess)
			goto ce_fail;
		for (j = 0; j < V; j++)
			db_host[j] += s_db_host[j];
		if (mean_ce)
			*mean_ce = rows > 0
			    ? (double)(*g_ce_host_pinned) / (double)rows
			    : -1.0;
	}
	pthread_mutex_unlock(&g_ew_mu);
	return 0;
ce_fail:
	pthread_mutex_unlock(&g_ew_mu);
	return -1;
}

extern "C" int gpu_ce_delta_from_softmax(ann_real *probs, const int *tgt,
	int rows, int V, ann_real *db_host, double *mean_ce)
{
	OpScope _op(OP_CE);

	int launched = 0;

	return ce_delta_run(probs, tgt, rows, V, NULL, db_host, mean_ce, 0,
	    &launched);
}

/*
** Head loss fused for BF16 batch training (ANN_CE_BF16=0 disables): the
** logits rows become softmax, CE and the BF16 delta in one kernel instead of
** softmax_rows_kernel, ce_onehot_ce_kernel, col_sum_tiled_kernel and the two
** FP32 -> BF16 conversions of the delta by the head wgrad and dgrad. The
** delta is stored in place over the logits (no extra VRAM) and registered as
** the in-place gradient twin of the logits buffer; the FP32 contents are
** destroyed, so the head's wgrad (peek) and dgrad (take) must both run on
** the twin, which gpu_ce_bf16_delta_done checks afterwards. -1 before any
** write (logits intact): run the FP32 steps; -2: failed after the kernel
** was launched, the logits may be overwritten. bias != NULL: the logits
** are the bare GEMM output and the kernel adds the head bias itself (B2,
** see gpu_ce_bias_fused); on -1 the caller still has to add it.
*/
extern "C" int gpu_ce_bias_fused(void)
{
	static int on = -1;

	if (on < 0) {
		const char *ev = getenv("ANN_CE_BIAS");

		on = (ev && ev[0] == '0') ? 0 : 1;
	}
	return on;
}

extern "C" int gpu_ce_softmax_delta_bf16(ann_real *logits, const int *tgt,
	int rows, int V, const ann_real *bias, ann_real *db_host,
	double *mean_ce)
{
	OpScope _op(OP_CE);
	static int on = -1;
	int launched = 0;
	size_t n;

	if (on < 0) {
		const char *ev = getenv("ANN_CE_BF16");

		on = (ev && ev[0] == '0') ? 0 : 1;
	}
	/*
	** BF16 GEMMs only: the FP8 dgrad and the FP32 / FP16 paths read the
	** FP32 delta. Batch mode only: eager evaluation keeps the FP32 steps.
	*/
	if (!on || !logits || !tgt || rows <= 0 || V <= 0 || !gpu_available()
	    || g_compute_prec != GPU_PREC_BF16 || g_fp8_dgrad
	    || g_sync_mode == GPU_SYNC_EAGER || !ew_is_dev(logits)
	    || (V + ew_block_for(V) - 1) / ew_block_for(V) > 64)
		return -1;
	n = (size_t)rows * (size_t)V;
	if (n > (size_t)0x7fffffff)
		return -1;
	if (ce_delta_run(logits, tgt, rows, V, bias, db_host, mean_ce, 1,
	    &launched) != 0)
		return launched ? -2 : -1;	/* -2: the logits are lost */
	pthread_mutex_lock(&g_gemm_mu);
	g_grad_twin[GRAD_TWIN_INPLACE].half = (void *)logits;
	g_grad_twin[GRAD_TWIN_INPLACE].cap = 0;
	g_grad_twin[GRAD_TWIN_INPLACE].src = logits;
	g_grad_twin[GRAD_TWIN_INPLACE].n = n;
	g_grad_twin[GRAD_TWIN_INPLACE].ld = 2 * V;
	g_ce_inplace_used = 0;
	pthread_mutex_unlock(&g_gemm_mu);
	{
		static int reported;

		if (!reported++)
			fprintf(stderr, "[lm] head softmax, CE and BF16 delta in one kernel\n");
	}
	return 0;
}

/*
** After the head's wgrad and dgrad: 0 when both ran on the in-place BF16
** delta (the FP32 buffer holds no delta any more, so any other consumer
** computed garbage); the twin is dropped either way.
*/
extern "C" int gpu_ce_bf16_delta_done(const ann_real *logits)
{
	int ok;

	pthread_mutex_lock(&g_gemm_mu);
	ok = g_ce_inplace_used == 3
	    && g_grad_twin[GRAD_TWIN_INPLACE].half == (const void *)logits
	    && g_grad_twin[GRAD_TWIN_INPLACE].src == NULL;
	g_grad_twin[GRAD_TWIN_INPLACE].src = NULL;
	g_grad_twin[GRAD_TWIN_INPLACE].n = 0;
	g_grad_twin[GRAD_TWIN_INPLACE].half = NULL;
	g_ce_inplace_used = 0;
	pthread_mutex_unlock(&g_gemm_mu);
	if (!ok)
		fprintf(stderr, "[lm] head BF16 delta not consumed by both GEMMs\n");
	return ok ? 0 : -1;
}

/*
** Probe loss: one block per row, FP32 log-sum-exp minus the target logit.
** Read-only on the logits (no softmax in place, no db_head, no delta), so the
** validation probe batches B windows through one head GEMM.
*/
__global__ static void ce_nll_rows_kernel(const ann_real *logits,
	const int *tgt, int V, float *out)
{
	__shared__ ann_real sh[EW_BLOCK];
	const ann_real *row = logits + (size_t)blockIdx.x * (size_t)V;
	ann_real m = -INFINITY;
	ann_real s = 0.0f;
	int j;

	for (j = threadIdx.x; j < V; j += blockDim.x)
		if (row[j] > m)
			m = row[j];
	m = ew_block_max(m, sh);
	for (j = threadIdx.x; j < V; j += blockDim.x)
		s += expf(row[j] - m);
	s = ew_block_sum(s, sh);
	if (threadIdx.x == 0)
		out[blockIdx.x] = (m + logf(s)) - row[tgt[blockIdx.x]];
}

extern "C" int gpu_ce_nll_rows(const ann_real *logits, const int *tgt,
	int rows, int V, float *out)
{
	OpScope _op(OP_CE);
	ew_ref rl;
	ew_ref rt;
	ew_ref ro;
	int rc;

	if (!logits || !tgt || !out || rows <= 0 || V <= 0 || !gpu_available()
	    || !ew_is_dev(logits))
		return -1;
	pthread_mutex_lock(&g_ew_mu);
	if (ew_acquire(&rl, 0, logits, (size_t)rows * (size_t)V
	    * sizeof(ann_real), 0, 0) != 0
	    || ew_acquire(&rt, 1, tgt, (size_t)rows * sizeof(int), 1, 0) != 0
	    || ew_acquire(&ro, 2, out, (size_t)rows * sizeof(float), 0, 1) != 0)
		goto fail;
	ce_nll_rows_kernel<<<rows, ew_block_for(V), 0, g_st>>>(rl.dev,
	    (const int *)rt.dev, V, (float *)ro.dev);
	if (ew_launch_ok() != 0)
		goto fail;
	rc = ew_release(&ro);	/* synchronising D2H of the per-row losses */
	if (rc < 0)
		goto fail;
	pthread_mutex_unlock(&g_ew_mu);
	return 0;
fail:
	pthread_mutex_unlock(&g_ew_mu);
	return -1;
}

extern "C" int gpu_ce_fetch_mean(double *mean_ce, int rows)
{
	if (!mean_ce || rows <= 0 || !g_ce_host_pinned || g_ce_last_rows <= 0)
		return -1;
	if (cudaStreamSynchronize(g_st) != cudaSuccess)
		return -1;
	*mean_ce = (double)(*g_ce_host_pinned) / (double)rows;
	return 0;
}

extern "C" int gpu_posenc_add_packed(ann_real *x, int B, int T, int D)
{
	OpScope _op(OP_POSENC);
	ew_ref rx;
	size_t nb;
	int n;
	int rc;
	int dn;

	if (!x || B <= 0 || T <= 0 || D <= 0 || !gpu_available())
		return -1;
	n = B * T * D;
	nb = (size_t)n * sizeof(ann_real);
	pthread_mutex_lock(&g_ew_mu);
	if (ew_acquire(&rx, 0, x, nb, 1, 1) != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	posenc_packed_kernel<<<(n + EW_BLOCK - 1) / EW_BLOCK, EW_BLOCK, 0, g_st>>>(rx.dev,
	    B, T, D);
	if (ew_launch_ok() != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	rc = ew_release(&rx);
	if (rc < 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	dn = rc;
	rc = ew_finish_sync(dn) == 0 ? 0 : -1;
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}

extern "C" int gpu_mat_col_sum_add(ann_real *db_host, const ann_real *x,
	int rows, int cols)
{
	OpScope _op(OP_COLSUM);
	ew_ref rx;
	colsum_acc *acc;
	size_t nb;
	int j;

	if (!db_host || !x || rows <= 0 || cols <= 0 || !gpu_available())
		return -1;
	nb = (size_t)rows * (size_t)cols * sizeof(ann_real);
	pthread_mutex_lock(&g_ew_mu);
	acc = colsum_acc_get(db_host, cols);
	if (!acc) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	if (ew_acquire(&rx, 0, x, nb, 1, 0) != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	col_sum_tiled_run(acc->dev, rx.dev, NULL, rows, cols);
	if (ew_launch_ok() != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	acc->dirty = 1;
	/*
	** BATCH / capture: leave sums on device; gpu_device_sync flushes.
	** Eager: sync this op so callers see host db immediately.
	*/
	if (g_sync_mode == GPU_SYNC_EAGER && !stream_is_capturing()) {
		ann_real *db_tmp;

		if (cudaDeviceSynchronize() != cudaSuccess) {
			pthread_mutex_unlock(&g_ew_mu);
			return -1;
		}
		db_tmp = (ann_real *)malloc((size_t)cols * sizeof(ann_real));
		if (!db_tmp
		    || cudaMemcpy(db_tmp, acc->dev,
		    (size_t)cols * sizeof(ann_real),
		    cudaMemcpyDeviceToHost) != cudaSuccess) {
			free(db_tmp);
			pthread_mutex_unlock(&g_ew_mu);
			return -1;
		}
		for (j = 0; j < cols; j++)
			db_host[j] += db_tmp[j];
		free(db_tmp);
		(void)cudaMemsetAsync(acc->dev, 0, (size_t)cols * sizeof(ann_real), g_st);
		acc->dirty = 0;
	}
	pthread_mutex_unlock(&g_ew_mu);
	return 0;
}

extern "C" void gpu_graph_set_mode(int on)
{
	on = on ? 1 : 0;
	if (on == g_graph_mode)
		return;
	gpu_graph_destroy();
	gpu_ids_unstage();
	if (!on && gpu_available()) {
		(void)cudaStreamSynchronize(g_st);
		pthread_mutex_lock(&g_pool_mu);
		pool_release_all();
		pthread_mutex_unlock(&g_pool_mu);
	}
	g_graph_mode = on;
}

/* Graphs need the capturable compute stream; FP8 keeps host-side state. */
extern "C" int gpu_graph_supported(void)
{
	return gpu_available() && g_st && !g_fp8_fwd && !g_fp8_dgrad;
}

extern "C" unsigned gpu_param_epoch(void)
{
	return g_wt_epoch;
}

extern "C" int gpu_graph_begin(void)
{
	if (!gpu_graph_supported() || !g_graph_mode)
		return -1;
	gpu_graph_destroy();
	if (g_copy_stream)
		(void)cudaStreamSynchronize(g_copy_stream);
	if (cudaStreamSynchronize(g_st) != cudaSuccess)
		return -1;
	g_capture_bad = 0;
	g_graph_gen = g_dev_free_gen;
	/* ThreadLocal: an illegal call from this thread fails the capture. */
	if (cudaStreamBeginCapture(g_st, cudaStreamCaptureModeThreadLocal)
	    != cudaSuccess) {
		(void)cudaGetLastError();
		return -1;
	}
	return 0;
}

extern "C" int gpu_graph_end(void)
{
	cudaError_t err;
	int i;

	if (!g_st)
		return -1;
	err = cudaStreamEndCapture(g_st, &g_graph);
	for (i = 0; i < g_defer_free_n; i++)
		cudaFree(g_defer_free[i]);
	g_defer_free_n = 0;
	/* Blocks the capture took and kept are its owners' now, not temporaries. */
	pthread_mutex_lock(&g_pool_mu);
	for (i = 0; i < g_pool_n; i++)
		if (g_pool[i].graph && !g_pool[i].free)
			g_pool[i].graph = 0;
	pthread_mutex_unlock(&g_pool_mu);
	if (err != cudaSuccess || g_capture_bad || !g_graph
	    || g_dev_free_gen != g_graph_gen) {
		if (graph_debug_on())
			fprintf(stderr, "[graph] capture dropped: %s%s%s\n",
			    cudaGetErrorString(err),
			    g_capture_bad ? ", refused call inside" : "",
			    g_dev_free_gen != g_graph_gen ? ", buffer freed" : "");
		(void)cudaGetLastError();
		gpu_graph_destroy();
		return -1;
	}
	if (cudaGraphInstantiate(&g_graph_exec, g_graph, NULL, NULL, 0)
	    != cudaSuccess) {
		(void)cudaGetLastError();
		gpu_graph_destroy();
		return -1;
	}
	g_graph_ok = 1;
	return 0;
}

/*
** A graph may run only while every address it holds is still what it was:
** no buffer freed since the capture and none of its temporaries lent out.
*/
extern "C" int gpu_graph_replayable(void)
{
	return g_graph_ok && g_graph_exec && g_dev_free_gen == g_graph_gen
	    && g_pool_graph_out == 0;
}

extern "C" int gpu_graph_replay(void)
{
	if (!gpu_graph_replayable()) {
		static int said;

		if (graph_debug_on() && g_graph_ok && said++ < 20)
			fprintf(stderr, "[graph] not replayed:%s%s\n",
			    g_dev_free_gen != g_graph_gen ? " a buffer was freed" : "",
			    g_pool_graph_out ? " a graph temporary is lent out" : "");
		return -1;
	}
	if (cudaGraphLaunch(g_graph_exec, g_st) != cudaSuccess) {
		(void)cudaGetLastError();
		return -1;
	}
	return launch_done();
}

extern "C" void gpu_graph_destroy(void)
{
	int i;

	if (g_graph_exec) {
		cudaGraphExecDestroy(g_graph_exec);
		g_graph_exec = NULL;
	}
	if (g_graph) {
		cudaGraphDestroy(g_graph);
		g_graph = NULL;
	}
	g_graph_ok = 0;
	/* Its temporaries are ordinary pool blocks again. */
	pthread_mutex_lock(&g_pool_mu);
	for (i = 0; i < g_pool_n; i++) {
		g_pool[i].graph = 0;
		g_pool[i].out = 0;
	}
	g_pool_graph_out = 0;
	pthread_mutex_unlock(&g_pool_mu);
}

/*
** ---- Single-stream decode (generate) ------------------------------------
**
** Weights are the persistent device twins; LN / bias vectors are copied
** once. K/V cache, activation rows and the step scalars stay on the device.
** One step: embed + posenc, n_layers blocks, head; only logits come back.
*/
typedef struct {
	const ann_real	*wq;
	const ann_real	*wk;
	const ann_real	*wv;
	const ann_real	*wo;
	const ann_real	*w1;
	const ann_real	*w2;
	ann_real	*ln1_g;
	ann_real	*ln1_b;
	ann_real	*ln2_g;
	ann_real	*ln2_b;
	ann_real	*b1;
	ann_real	*b2;
	float		eps1;
	float		eps2;
} dec_layer;

struct gpu_dec {
	int		L;
	int		D;
	int		F;
	int		V;
	int		max_T;
	int		H;
	int		dh;
	dec_layer	*lay;
	const ann_real	*embed;
	const ann_real	*w_head;
	ann_real	*b_head;
	ann_real	*kc;
	ann_real	*vc;
	ann_real	*x;
	ann_real	*ln;
	ann_real	*q;
	ann_real	*o;
	ann_real	*hid;
	ann_real	*logits;
	int		*d_step;	/* [0] = id, [1] = pos */
	int		*h_step;	/* pinned */
	ann_real	*part;		/* split-K partial sums */
	ann_real	*h_logits;	/* pinned */
	cudaStream_t	st;
	cudaGraph_t	graph;
	cudaGraphExec_t	graph_exec;
	int		graph_state;	/* 0 untried, 1 replay, -1 plain */
};

/*
** Split-K matrix-vector: block (x, y, z) owns DEC_MV_TPB*4 columns of
** matrix z over DEC_MV_ROWS input rows and writes one partial row. The
** reduce kernel sums partials in a fixed order (no atomics, same bits every
** run), then bias / ReLU / residual / K-V row offset.
*/
#define DEC_MV_ROWS	32
#define DEC_MV_TPB	64
#define DEC_MV_MAX	3

typedef struct {
	const ann_real	*W[DEC_MV_MAX];
	ann_real	*y[DEC_MV_MAX];
	int		at_pos[DEC_MV_MAX];	/* y += step[1] * Dout */
} dec_mv;

static int dec_mv_splits(int Din)
{
	return (Din + DEC_MV_ROWS - 1) / DEC_MV_ROWS;
}

__global__ static void dec_mv_part_kernel(ann_real *part, const ann_real *x,
	dec_mv m, int Din, int Dout)
{
	__shared__ ann_real sx[DEC_MV_ROWS];
	const ann_real *W = m.W[blockIdx.z];
	int col = (blockIdx.x * DEC_MV_TPB + threadIdx.x) * 4;
	int r0 = blockIdx.y * DEC_MV_ROWS;
	int nr = min(DEC_MV_ROWS, Din - r0);
	float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
	int i;

	for (i = threadIdx.x; i < nr; i += DEC_MV_TPB)
		sx[i] = x[r0 + i];
	__syncthreads();
	if (col >= Dout)
		return;
#pragma unroll 8
	for (i = 0; i < nr; i++) {
		float4 w = *(const float4 *)(W + (size_t)(r0 + i) * Dout + col);
		float xi = sx[i];

		acc.x += xi * w.x;
		acc.y += xi * w.y;
		acc.z += xi * w.z;
		acc.w += xi * w.w;
	}
	*(float4 *)(part + ((size_t)blockIdx.z * gridDim.y + blockIdx.y)
	    * Dout + col) = acc;
}

__global__ static void dec_mv_reduce_kernel(dec_mv m, const ann_real *part,
	int S, int Dout, const ann_real *bias, const ann_real *resid,
	const int *step, int relu)
{
	int z = blockIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	const ann_real *p;
	ann_real *y;
	float s;
	int k;

	if (j >= Dout)
		return;
	p = part + (size_t)z * S * Dout + j;
	s = 0.0f;
	for (k = 0; k < S; k++)
		s += p[(size_t)k * Dout];
	if (bias)
		s += bias[j];
	if (relu && s < 0.0f)
		s = 0.0f;
	if (resid)
		s += resid[j];
	y = m.y[z];
	if (m.at_pos[z])
		y += (size_t)step[1] * (size_t)Dout;
	y[j] = s;
}

static ann_real *dec_upload(const ann_real *host, int n)
{
	ann_real *d;

	if (!host || n <= 0)
		return NULL;
	if (cudaMalloc((void **)&d, (size_t)n * sizeof(ann_real)) != cudaSuccess)
		return NULL;
	if (cudaMemcpy(d, host, (size_t)n * sizeof(ann_real),
	    cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(d);
		return NULL;
	}
	return d;
}

__global__ static void dec_embed_kernel(ann_real *x, const ann_real *table,
	const int *step, int D)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int pair;
	float log_base;
	float angle;

	if (d >= D)
		return;
	/* Same float math as posenc_packed_kernel. */
	pair = d / 2;
	log_base = -logf(10000.0f) / (float)D;
	angle = (float)step[1] * expf((2.0f * (float)pair) * log_base);
	x[d] = table[(size_t)step[0] * (size_t)D + d]
	    + ((d & 1) == 0 ? sinf(angle) : cosf(angle));
}

/*
** y[j] = act(x W[:, j] + bias[j]) + resid[j], W row-major Din x Dout.
** step != NULL offsets y by step[1] rows of Dout (K/V cache append).
*/
__global__ static void dec_gemv_kernel(ann_real *y, const ann_real *x,
	const ann_real *W, const ann_real *bias, const ann_real *resid,
	const int *step, int Din, int Dout, int relu)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	float s;

	if (j >= Dout)
		return;
	s = 0.0f;
	for (i = 0; i < Din; i++)
		s += x[i] * W[(size_t)i * (size_t)Dout + j];
	if (bias)
		s += bias[j];
	if (relu && s < 0.0f)
		s = 0.0f;
	if (resid)
		s += resid[j];
	if (step)
		y += (size_t)step[1] * (size_t)Dout;
	y[j] = s;
}

/* One block per head over cache rows [0, pos]; scores live in smem. */
__global__ static void dec_attn_kernel(ann_real *o, const ann_real *q,
	const ann_real *kc, const ann_real *vc, const int *step, int D, int dh,
	float scale)
{
	extern __shared__ ann_real sc[];
	__shared__ ann_real sh[EW_BLOCK];
	int off = blockIdx.x * dh;
	int len = step[1] + 1;
	int j;
	int t;
	float s;
	float mx;
	float sum;

	mx = -1e30f;
	for (j = threadIdx.x; j < len; j += blockDim.x) {
		s = 0.0f;
		for (t = 0; t < dh; t++)
			s += q[off + t] * kc[(size_t)j * D + off + t];
		s *= scale;
		sc[j] = s;
		mx = fmaxf(mx, s);
	}
	mx = ew_block_max(mx, sh);
	sum = 0.0f;
	for (j = threadIdx.x; j < len; j += blockDim.x) {
		s = expf(sc[j] - mx);
		sc[j] = s;
		sum += s;
	}
	sum = ew_block_sum(sum, sh);
	__syncthreads();
	for (t = threadIdx.x; t < dh; t += blockDim.x) {
		s = 0.0f;
		for (j = 0; j < len; j++)
			s += sc[j] * vc[(size_t)j * D + off + t];
		o[off + t] = s / sum;
	}
}

extern "C" gpu_dec *gpu_dec_create(int n_layers, int d_model, int d_ff,
	int vocab, int max_T, int n_heads)
{
	gpu_dec *g;
	size_t kvb;

	if (!gpu_available() || n_layers <= 0 || d_model <= 0 || d_ff <= 0
	    || vocab <= 0 || max_T <= 0 || n_heads <= 0
	    || d_model % n_heads != 0 || d_model > 4096)
		return NULL;
	g = (gpu_dec *)calloc(1, sizeof(*g));
	if (!g)
		return NULL;
	g->L = n_layers;
	g->D = d_model;
	g->F = d_ff;
	g->V = vocab;
	g->max_T = max_T;
	g->H = n_heads;
	g->dh = d_model / n_heads;
	g->lay = (dec_layer *)calloc((size_t)n_layers, sizeof(dec_layer));
	kvb = (size_t)n_layers * (size_t)max_T * (size_t)d_model
	    * sizeof(ann_real);
	{
		size_t pn;
		size_t t;

		pn = (size_t)DEC_MV_MAX * dec_mv_splits(d_model) * d_model;
		t = (size_t)dec_mv_splits(d_model) * d_ff;
		pn = t > pn ? t : pn;
		t = (size_t)dec_mv_splits(d_ff) * d_model;
		pn = t > pn ? t : pn;
		t = (size_t)dec_mv_splits(d_model) * vocab;
		pn = t > pn ? t : pn;
		if (cudaMalloc((void **)&g->part, pn * sizeof(ann_real))
		    != cudaSuccess) {
			g->part = NULL;
			gpu_dec_free(&g);
			return NULL;
		}
	}
	if (!g->lay
	    || cudaMalloc((void **)&g->kc, kvb) != cudaSuccess
	    || cudaMalloc((void **)&g->vc, kvb) != cudaSuccess
	    || cudaMalloc((void **)&g->x, (size_t)d_model * sizeof(ann_real))
	    != cudaSuccess
	    || cudaMalloc((void **)&g->ln, (size_t)d_model * sizeof(ann_real))
	    != cudaSuccess
	    || cudaMalloc((void **)&g->q, (size_t)d_model * sizeof(ann_real))
	    != cudaSuccess
	    || cudaMalloc((void **)&g->o, (size_t)d_model * sizeof(ann_real))
	    != cudaSuccess
	    || cudaMalloc((void **)&g->hid, (size_t)d_ff * sizeof(ann_real))
	    != cudaSuccess
	    || cudaMalloc((void **)&g->logits, (size_t)vocab * sizeof(ann_real))
	    != cudaSuccess
	    || cudaMalloc((void **)&g->d_step, 2 * sizeof(int)) != cudaSuccess
	    || cudaHostAlloc((void **)&g->h_step, 2 * sizeof(int),
	    cudaHostAllocDefault) != cudaSuccess
	    || cudaHostAlloc((void **)&g->h_logits,
	    (size_t)vocab * sizeof(ann_real), cudaHostAllocDefault)
	    != cudaSuccess
	    || cudaStreamCreate(&g->st) != cudaSuccess) {
		gpu_dec_free(&g);
		return NULL;
	}
	return g;
}

extern "C" int gpu_dec_set_layer(gpu_dec *g, int i, const gpu_dec_layer_w *w)
{
	dec_layer *l;

	if (!g || !w || i < 0 || i >= g->L)
		return -1;
	if (!w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2)
		return -1;
	l = &g->lay[i];
	l->wq = w->wq;
	l->wk = w->wk;
	l->wv = w->wv;
	l->wo = w->wo;
	l->w1 = w->w1;
	l->w2 = w->w2;
	l->eps1 = (float)w->eps1;
	l->eps2 = (float)w->eps2;
	if (!(l->ln1_g = dec_upload(w->ln1_g, g->D))
	    || !(l->ln1_b = dec_upload(w->ln1_b, g->D))
	    || !(l->ln2_g = dec_upload(w->ln2_g, g->D))
	    || !(l->ln2_b = dec_upload(w->ln2_b, g->D))
	    || !(l->b1 = dec_upload(w->b1, g->F))
	    || !(l->b2 = dec_upload(w->b2, g->D)))
		return -1;
	return 0;
}

extern "C" int gpu_dec_set_io(gpu_dec *g, const ann_real *embed_dev,
	const ann_real *w_head_dev, const ann_real *b_head_host)
{
	if (!g || !embed_dev || !w_head_dev || !b_head_host)
		return -1;
	g->embed = embed_dev;
	g->w_head = w_head_dev;
	g->b_head = dec_upload(b_head_host, g->V);
	return g->b_head ? 0 : -1;
}

extern "C" int gpu_dec_kv_load(gpu_dec *g, int layer, const ann_real *k_host,
	const ann_real *v_host, int n)
{
	size_t off;
	size_t nb;

	if (!g || !k_host || !v_host || layer < 0 || layer >= g->L || n <= 0
	    || n > g->max_T)
		return -1;
	off = (size_t)layer * (size_t)g->max_T * (size_t)g->D;
	nb = (size_t)n * (size_t)g->D * sizeof(ann_real);
	if (cudaMemcpy(g->kc + off, k_host, nb, cudaMemcpyHostToDevice)
	    != cudaSuccess
	    || cudaMemcpy(g->vc + off, v_host, nb, cudaMemcpyHostToDevice)
	    != cudaSuccess)
		return -1;
	return 0;
}

/* n matrices sharing x (n > 1: no bias / resid). Falls back if Dout % 4. */
static void dec_mv_run(gpu_dec *g, const dec_mv *m, int n, const ann_real *x,
	const ann_real *bias, const ann_real *resid, int Din, int Dout,
	int relu)
{
	int S = dec_mv_splits(Din);
	dim3 pg((Dout / 4 + DEC_MV_TPB - 1) / DEC_MV_TPB, S, n);
	dim3 rg((Dout + 255) / 256, n);
	int vec4 = (Dout & 3) == 0;
	int i;

	for (i = 0; i < n; i++)
		if (((size_t)m->W[i] & 15) != 0)
			vec4 = 0;
	if (!vec4) {
		for (i = 0; i < n; i++)
			dec_gemv_kernel<<<(Dout + 255) / 256, 256, 0, g->st>>>(
			    m->y[i], x, m->W[i], bias, resid,
			    m->at_pos[i] ? g->d_step : NULL, Din, Dout, relu);
		return;
	}
	dec_mv_part_kernel<<<pg, DEC_MV_TPB, 0, g->st>>>(g->part, x, *m, Din,
	    Dout);
	dec_mv_reduce_kernel<<<rg, 256, 0, g->st>>>(*m, g->part, S, Dout, bias,
	    resid, g->d_step, relu);
}

static void dec_gemv(gpu_dec *g, ann_real *y, const ann_real *x,
	const ann_real *W, const ann_real *bias, const ann_real *resid,
	int Din, int Dout, int relu)
{
	dec_mv m;

	memset(&m, 0, sizeof(m));
	m.W[0] = W;
	m.y[0] = y;
	dec_mv_run(g, &m, 1, x, bias, resid, Din, Dout, relu);
}

static void dec_launch_step(gpu_dec *g)
{
	const int D = g->D;
	const size_t kvl = (size_t)g->max_T * (size_t)D;
	const int lnb = ew_block_for(D);
	const size_t lns = (size_t)D * sizeof(ann_real);
	const float scale = 1.0f / sqrtf((float)g->dh);
	int i;

	dec_embed_kernel<<<(D + 255) / 256, 256, 0, g->st>>>(g->x, g->embed,
	    g->d_step, D);
	for (i = 0; i < g->L; i++) {
		dec_layer *l = &g->lay[i];
		ann_real *kc = g->kc + (size_t)i * kvl;
		ann_real *vc = g->vc + (size_t)i * kvl;
		dec_mv qkv;

		ln_fwd_kernel<<<1, lnb, lns, g->st>>>(g->ln, g->x, l->ln1_g,
		    l->ln1_b, NULL, NULL, NULL, D, (ann_real)l->eps1, NULL, NULL,
		    NULL, NULL);
		qkv.W[0] = l->wq;
		qkv.W[1] = l->wk;
		qkv.W[2] = l->wv;
		qkv.y[0] = g->q;
		qkv.y[1] = kc;
		qkv.y[2] = vc;
		qkv.at_pos[0] = 0;
		qkv.at_pos[1] = 1;
		qkv.at_pos[2] = 1;
		dec_mv_run(g, &qkv, 3, g->ln, NULL, NULL, D, D, 0);
		dec_attn_kernel<<<g->H, 128, (size_t)g->max_T
		    * sizeof(ann_real), g->st>>>(g->o, g->q, kc, vc, g->d_step,
		    D, g->dh, scale);
		dec_gemv(g, g->x, g->o, l->wo, NULL, g->x, D, D, 0);
		ln_fwd_kernel<<<1, lnb, lns, g->st>>>(g->ln, g->x, l->ln2_g,
		    l->ln2_b, NULL, NULL, NULL, D, (ann_real)l->eps2, NULL, NULL,
		    NULL, NULL);
		dec_gemv(g, g->hid, g->ln, l->w1, l->b1, NULL, D, g->F, 1);
		dec_gemv(g, g->x, g->hid, l->w2, l->b2, g->x, g->F, D, 0);
	}
	dec_gemv(g, g->logits, g->x, g->w_head, g->b_head, NULL, D, g->V, 0);
}

extern "C" int gpu_dec_step(gpu_dec *g, int id, int pos, ann_real *logits)
{
	if (!g || !logits || !g->embed || !g->w_head || id < 0 || id >= g->V
	    || pos < 0 || pos >= g->max_T)
		return -1;
	g->h_step[0] = id;
	g->h_step[1] = pos;
	if (cudaMemcpyAsync(g->d_step, g->h_step, 2 * sizeof(int),
	    cudaMemcpyHostToDevice, g->st) != cudaSuccess)
		return -1;
	/* Pointers never move and pos is read on the device: one graph. */
	if (g->graph_state == 0) {
		const char *env = getenv("ANN_DECODE_GRAPH");

		g->graph_state = -1;
		if ((!env || atoi(env) != 0)
		    && cudaStreamBeginCapture(g->st,
		    cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
			dec_launch_step(g);
			if (cudaStreamEndCapture(g->st, &g->graph) == cudaSuccess
			    && cudaGraphInstantiate(&g->graph_exec, g->graph,
			    NULL, NULL, 0) == cudaSuccess)
				g->graph_state = 1;
		}
		(void)cudaGetLastError();
	}
	if (g->graph_state == 1) {
		if (cudaGraphLaunch(g->graph_exec, g->st) != cudaSuccess)
			return -1;
	} else
		dec_launch_step(g);
	if (cudaGetLastError() != cudaSuccess)
		return -1;
	if (cudaMemcpyAsync(g->h_logits, g->logits,
	    (size_t)g->V * sizeof(ann_real), cudaMemcpyDeviceToHost, g->st)
	    != cudaSuccess || cudaStreamSynchronize(g->st) != cudaSuccess)
		return -1;
	memcpy(logits, g->h_logits, (size_t)g->V * sizeof(ann_real));
	return 0;
}

extern "C" void gpu_dec_free(gpu_dec **gp)
{
	gpu_dec *g;
	int i;

	if (!gp || !*gp)
		return;
	g = *gp;
	if (g->lay) {
		for (i = 0; i < g->L; i++) {
			cudaFree(g->lay[i].ln1_g);
			cudaFree(g->lay[i].ln1_b);
			cudaFree(g->lay[i].ln2_g);
			cudaFree(g->lay[i].ln2_b);
			cudaFree(g->lay[i].b1);
			cudaFree(g->lay[i].b2);
		}
		free(g->lay);
	}
	cudaFree(g->b_head);
	cudaFree(g->kc);
	cudaFree(g->vc);
	cudaFree(g->x);
	cudaFree(g->ln);
	cudaFree(g->q);
	cudaFree(g->o);
	cudaFree(g->hid);
	cudaFree(g->logits);
	cudaFree(g->d_step);
	cudaFree(g->part);
	if (g->graph_exec)
		cudaGraphExecDestroy(g->graph_exec);
	if (g->graph)
		cudaGraphDestroy(g->graph);
	if (g->st)
		cudaStreamDestroy(g->st);
	if (g->h_step)
		cudaFreeHost(g->h_step);
	if (g->h_logits)
		cudaFreeHost(g->h_logits);
	free(g);
	*gp = NULL;
}
