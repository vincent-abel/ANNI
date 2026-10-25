/*
**	Optional CUDA backend (build with: make cuda)
*/
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
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
static int g_compute_prec;
static int g_fp8_fwd;
static int g_fp8_dgrad;
static cudaStream_t g_copy_stream;
static cudaGraph_t g_graph;
static cudaGraphExec_t g_graph_exec;
static int g_graph_ok;

extern "C" void gpu_set_compute_prec(int prec)
{
	if (prec != GPU_PREC_BF16 && prec != GPU_PREC_FP16)
		prec = GPU_PREC_FP32;
	g_compute_prec = prec;
}

extern "C" int gpu_compute_prec(void)
{
	return g_compute_prec;
}

extern "C" void gpu_set_fp8(int on)
{
	g_fp8_fwd = on ? 1 : 0;
	/*
	** --fp8: E4M3 forward and hot LayerNorm, E4M3×E5M2 dgrad,
	** E4M3 softmax probabilities, group-128 E4M3 Adam moments.
	** Wgrad stays BF16 unless ANN_FP8_WGRAD=1 (two transposes lose).
	** NVFP4 GEMM is not in this cuBLAS (probe status 7).
	*/
	if (on)
		g_fp8_dgrad = 1;
}

extern "C" int gpu_fp8(void)
{
	return g_fp8_fwd;
}

extern "C" void gpu_set_fp8_dgrad(int on)
{
	g_fp8_dgrad = on ? 1 : 0;
}

extern "C" int gpu_fp8_dgrad(void)
{
	return g_fp8_dgrad;
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

#if CUDART_VERSION >= 11000
static int cublas_gemm_ex_bf16_h(cublasHandle_t h, int N, int M, int K,
	const float *alpha, const void *B, int ldb, const void *A, int lda,
	const float *beta, float *C, int ldc)
{
	return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B,
	    CUDA_R_16BF, ldb, A, CUDA_R_16BF, lda, beta, C, CUDA_R_32F, ldc,
	    CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
	    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}
#endif

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
** cuBLASLt BF16/FP16 GEMM with cached heuristic. Falls back to GemmEx.
** Column-major dims match existing GemmEx callers (N,M,K with opa/opb).
** epi/bias_dev: optional Lt epilogue (BIAS / RELU_BIAS); bias length = N.
*/
static int cublas_lt_gemm_half(cublasOperation_t opa, cublasOperation_t opb,
	int N, int M, int K, const float *alpha, const void *B, int ldb,
	const void *A, int lda, const float *beta, float *C, int ldc, int bf16,
	cublasLtEpilogue_t epi, const float *bias_dev)
{
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
	dt = bf16 ? CUDA_R_16BF : CUDA_R_16F;
	for (i = 0; i < LT_ALGO_CACHE; i++) {
		e = &g_lt_algo[i];
		if (e->valid && e->M == M && e->N == N && e->K == K
		    && e->opa == (int)opa && e->opb == (int)opb
		    && e->bf16 == bf16 && e->fp8 == 0 && e->epi == epi_i) {
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
	    && e->fp8 == 0 && e->epi == epi_i && e->lda == lda && e->ldb == ldb
	    && e->ldc == ldc) {
		if (bias_dev != e->bias) {
			(void)cublasLtMatmulDescSetAttribute(e->desc,
			    CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_dev,
			    sizeof(bias_dev));
			e->bias = bias_dev;
		}
		st = cublasLtMatmul(g_cublaslt, e->desc, alpha, B, e->layoutA,
		    A, e->layoutB, beta, C, e->layoutC, C, e->layoutC,
		    &e->algo, g_lt_workspace,
		    e->ws > 0 ? e->ws : g_lt_workspace_bytes, 0);
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
		cublasComputeType_t cterm = bf16
		    ? CUBLAS_COMPUTE_32F_FAST_16BF
		    : CUBLAS_COMPUTE_32F_FAST_16F;

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
		cudaDataType_t bdt = CUDA_R_32F;

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
	    || e->opa != (int)opa || e->opb != (int)opb || e->bf16 != bf16
	    || e->fp8 != 0 || e->epi != epi_i) {
		cublasLtMatmulHeuristicResult_t heurs[16];
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
		time_D = C;
		time_D_owned = 0;
		if (*beta != 0.0f) {
			size_t d_bytes = (size_t)M * (size_t)N * sizeof(float);

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
				(void)cudaEventRecord(ev0, 0);
				r = cublasLtMatmul(g_cublaslt, op, alpha, B,
				    layoutA, A, layoutB, beta, C, layoutC,
				    time_D, layoutC, &heurs[hi].algo,
				    g_lt_workspace,
				    heurs[hi].workspaceSize > 0
				    ? heurs[hi].workspaceSize
				    : g_lt_workspace_bytes, 0);
				(void)cudaEventRecord(ev1, 0);
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
	    C, layoutC, C, layoutC, &e->algo,
	    g_lt_workspace, e->ws > 0 ? e->ws : g_lt_workspace_bytes, 0);
	cublasLtMatmulPreferenceDestroy(pref);
	return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

#if CUDART_VERSION >= 12000 && defined(ANN_CUBLAS)
/*
** FP8 mixed stack: E4M3 forward, E4M3×E5M2 dgrad, E4M3×E5M2 wgrad, FP32 accum.
** Note: on sm_89/sm_120 Tensor Cores, cuBLASLt FP8 ONLY supports (OP_T, OP_N)
** and rejects E5M2×E5M2. So:
**   - Forward (C = A * B): B [K, N] is cached transposed as [N, K] E4M3 so
**     cuBLASLt sees layoutA=[K,N] col-major with OP_T and layoutB=A [K,M] OP_N.
**   - Dgrad (dX = dY * W^T): W [N, K] is cached as [N, K] E4M3 (layoutA=[K,N]
**     col-major with OP_T) and dY [M, K] is quantized to E5M2 (OP_N).
**   - Wgrad (dW += X^T * dY): dY [T, N] is quantized to E5M2 (layoutA=[N,T]
**     col-major -> transposed to [T,N] col-major for OP_T, or X transposed).
*/
static void *g_fp8_A;
static void *g_fp8_B;
static size_t g_fp8_A_cap;
static size_t g_fp8_B_cap;
static float *g_fp8_scale_A;
static float *g_fp8_scale_B;
static float *g_fp8_scale_A_next;	/* delayed act scale (fwd) */
static float *g_fp8_scale_B_next;	/* delayed grad scale (bwd) */
static float *g_fp8_scale_WA;		/* wgrad A, not the fwd act scale */
static float *g_fp8_scale_WB;
static float *g_fp8_scale_WA_next;
static float *g_fp8_scale_WB_next;
static float *g_fp8_amax_buf;
/* LN one-pass writes E4M3 here; the next GEMM on that pointer skips quant. */
static void *g_fp8_hot_buf;
static size_t g_fp8_hot_cap;
static float *g_fp8_hot_quant;		/* scale the next LN quantizes with */
static float *g_fp8_hot_buf_scale;	/* scale that matches g_fp8_hot_buf */
static float *g_fp8_hot_amax;
static const ann_real *g_fp8_hot_src;
static int g_fp8_hot_n;
/* FP8 Adam master + E4M3 moments (group 128). FP32 param stays the numeric copy. */
static __nv_fp8_storage_t *g_fp8_master;
static __nv_fp8_storage_t *g_fp8_adam_m;
static __nv_fp8_storage_t *g_fp8_adam_v;
static float *g_fp8_master_scale;
static float *g_fp8_adam_m_scale;
static float *g_fp8_adam_v_scale;
static int g_fp8_adam_n;
static int g_fp8_mv_live;
static int g_fp8_scale_ready;
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

/* Fused 32x32 transpose + FP8 quantize + block amax for delayed scaling. */
__global__ static void f32_transpose_to_fp8_delayed_kernel(
	__nv_fp8_storage_t *dst, float *amax, const float *src, int rows,
	int cols, const float *scale_dev, int e5m2)
{
	__shared__ __nv_fp8_storage_t tile[32][33];
	__shared__ float smax[8];
	int x = blockIdx.x * 32 + threadIdx.x;
	int y = blockIdx.y * 32 + threadIdx.y;
	int tid = threadIdx.y * 32 + threadIdx.x;
	float scale = scale_dev ? scale_dev[0] : 1.f;
	float inv = (scale > 0.f) ? (1.f / scale) : 0.f;
	float vmax = 0.0f;
	__nv_fp8_interpretation_t fmt = e5m2 ? __NV_E5M2 : __NV_E4M3;

	for (int j = 0; j < 32; j += 8) {
		if (x < cols && (y + j) < rows) {
			float v = src[(size_t)(y + j) * (size_t)cols + (size_t)x];
			vmax = fmaxf(vmax, fabsf(v));
			tile[threadIdx.y + j][threadIdx.x] =
			    __nv_cvt_float_to_fp8(v * inv, __NV_SATFINITE, fmt);
		}
	}
	for (int mask = 16; mask > 0; mask >>= 1)
		vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, mask));
	if ((tid & 31) == 0)
		smax[tid >> 5] = vmax;
	__syncthreads();
	if (tid < 8) {
		vmax = smax[tid];
		for (int mask = 4; mask > 0; mask >>= 1)
			vmax = fmaxf(vmax, __shfl_xor_sync(0xffu, vmax, mask));
		if (tid == 0 && vmax > 0.0f)
			atomicMax((int *)amax, __float_as_int(vmax));
	}
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
	float one = 1.f;

	if (!g_fp8_amax_buf
	    && cudaMalloc((void **)&g_fp8_amax_buf, sizeof(float)) != cudaSuccess)
		return -1;
	if (!g_fp8_scale_A
	    && cudaMalloc((void **)&g_fp8_scale_A, sizeof(float)) != cudaSuccess)
		return -1;
	if (!g_fp8_scale_B
	    && cudaMalloc((void **)&g_fp8_scale_B, sizeof(float)) != cudaSuccess)
		return -1;
	if (!g_fp8_scale_A_next
	    && cudaMalloc((void **)&g_fp8_scale_A_next, sizeof(float))
	    != cudaSuccess)
		return -1;
	if (!g_fp8_scale_B_next
	    && cudaMalloc((void **)&g_fp8_scale_B_next, sizeof(float))
	    != cudaSuccess)
		return -1;
	if (!g_fp8_scale_WA
	    && cudaMalloc((void **)&g_fp8_scale_WA, sizeof(float)) != cudaSuccess)
		return -1;
	if (!g_fp8_scale_WB
	    && cudaMalloc((void **)&g_fp8_scale_WB, sizeof(float)) != cudaSuccess)
		return -1;
	if (!g_fp8_scale_WA_next
	    && cudaMalloc((void **)&g_fp8_scale_WA_next, sizeof(float))
	    != cudaSuccess)
		return -1;
	if (!g_fp8_scale_WB_next
	    && cudaMalloc((void **)&g_fp8_scale_WB_next, sizeof(float))
	    != cudaSuccess)
		return -1;
	if (!g_fp8_hot_quant
	    && cudaMalloc((void **)&g_fp8_hot_quant, sizeof(float)) != cudaSuccess)
		return -1;
	if (!g_fp8_hot_buf_scale
	    && cudaMalloc((void **)&g_fp8_hot_buf_scale, sizeof(float))
	    != cudaSuccess)
		return -1;
	if (!g_fp8_hot_amax
	    && cudaMalloc((void **)&g_fp8_hot_amax, sizeof(float)) != cudaSuccess)
		return -1;
	if (!g_fp8_scale_ready) {
		(void)cudaMemcpy(g_fp8_scale_A, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		(void)cudaMemcpy(g_fp8_scale_B, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		(void)cudaMemcpy(g_fp8_scale_A_next, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		(void)cudaMemcpy(g_fp8_scale_B_next, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		(void)cudaMemcpy(g_fp8_scale_WA, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		(void)cudaMemcpy(g_fp8_scale_WB, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		(void)cudaMemcpy(g_fp8_scale_WA_next, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		(void)cudaMemcpy(g_fp8_scale_WB_next, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		(void)cudaMemcpy(g_fp8_hot_quant, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		(void)cudaMemcpy(g_fp8_hot_buf_scale, &one, sizeof(float),
		    cudaMemcpyHostToDevice);
		g_fp8_scale_ready = 1;
	}
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
	if (cudaMemsetAsync(g_fp8_hot_amax, 0, sizeof(float), 0) != cudaSuccess)
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
	amax_to_scale_kernel<<<1, 1>>>(g_fp8_hot_scale, g_fp8_hot_amax, 448.f);
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
	cudaMemsetAsync(g_fp8_amax_buf, 0, sizeof(float), 0);
	blocks = (n + threads * 4 - 1) / (threads * 4);
	if (blocks > 1024)
		blocks = 1024;
	if (blocks < 1)
		blocks = 1;
	f32_amax_block_kernel<<<blocks, threads>>>(g_fp8_amax_buf, src, n);
	amax_to_scale_kernel<<<1, 1>>>(scale_dev, g_fp8_amax_buf,
	    fp8_max_finite(e5m2));
	blocks = (n + threads * 4 - 1) / (threads * 4);
	if (blocks < 1)
		blocks = 1;
	f32_to_fp8_vec_kernel<<<blocks, threads>>>(
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
	cudaMemsetAsync(g_fp8_amax_buf, 0, sizeof(float), 0);
	blocks = (n + threads * 4 - 1) / (threads * 4);
	if (blocks > 1024)
		blocks = 1024;
	if (blocks < 1)
		blocks = 1;
	f32_amax_block_kernel<<<blocks, threads>>>(g_fp8_amax_buf, src, n);
	amax_to_scale_kernel<<<1, 1>>>(scale_dev, g_fp8_amax_buf,
	    fp8_max_finite(e5m2));
	f32_transpose_to_fp8_kernel<<<grid, block>>>(
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
	cudaMemsetAsync(g_fp8_amax_buf, 0, sizeof(float), 0);
	blocks = (n + threads * 4 - 1) / (threads * 4);
	if (blocks < 1)
		blocks = 1;
	f32_to_fp8_delayed_kernel<<<blocks, threads>>>(
	    (__nv_fp8_storage_t *)dst_fp8, g_fp8_amax_buf, src, n, scale_dev,
	    e5m2);
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static void fp8_act_scale_commit(float *scale, int e5m2)
{
	if (!scale || !g_fp8_amax_buf)
		return;
	amax_to_scale_kernel<<<1, 1>>>(scale, g_fp8_amax_buf,
	    fp8_max_finite(e5m2));
}

static int fp8_quant_transpose_tensor(void *dst_fp8, float *scale_dev,
	float *scale_next, const float *src, int rows, int cols, int e5m2)
{
	dim3 grid((cols + 31) / 32, (rows + 31) / 32, 1);
	dim3 block(32, 8, 1);

	if (!dst_fp8 || !scale_dev || !scale_next || !src || rows <= 0 || cols <= 0)
		return -1;
	if (fp8_scales_ensure() != 0)
		return -1;
	cudaMemsetAsync(g_fp8_amax_buf, 0, sizeof(float), 0);
	f32_transpose_to_fp8_delayed_kernel<<<grid, block>>>(
	    (__nv_fp8_storage_t *)dst_fp8, g_fp8_amax_buf, src, rows, cols,
	    scale_dev, e5m2);
	amax_to_scale_kernel<<<1, 1>>>(scale_next, g_fp8_amax_buf,
	    fp8_max_finite(e5m2));
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
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

static int fp8_wt_get(const ann_real *fp32, size_t n, int e5m2, void **out_fp8,
	float **out_scale)
{
	return fp8_wt_get_ex(fp32, (int)n, 1, 0, e5m2, out_fp8, out_scale);
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
		    e->ws > 0 ? e->ws : g_lt_workspace_bytes, 0);
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
			if (time_D && cudaStreamIsCapturing(0, &cs) == cudaSuccess
			    && cs == cudaStreamCaptureStatusNone
			    && cudaEventCreate(&ev0) == cudaSuccess
			    && cudaEventCreate(&ev1) == cudaSuccess) {
				for (hi = 0; hi < returned; hi++) {
					float ms = 0.0f;
					cublasStatus_t r;

					if (heurs[hi].workspaceSize
					    > g_lt_workspace_bytes)
						continue;
					(void)cudaEventRecord(ev0, 0);
					r = cublasLtMatmul(g_cublaslt, op,
					    alpha, B, layoutA, A, layoutB, beta,
					    C, layoutC, time_D, layoutC,
					    &heurs[hi].algo, g_lt_workspace,
					    heurs[hi].workspaceSize > 0
					    ? heurs[hi].workspaceSize
					    : g_lt_workspace_bytes, 0);
					(void)cudaEventRecord(ev1, 0);
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
	    g_lt_workspace, e->ws > 0 ? e->ws : g_lt_workspace_bytes, 0);
	cublasLtMatmulPreferenceDestroy(pref);
	/* desc and layouts stay on the shape slot. */
	return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}
#endif /* CUDART>=12000 FP8 */

__global__ static void f32_to_f16_kernel(half *dst, const float *src, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		dst[i] = __float2half(src[i]);
}

#if CUDART_VERSION >= 11000
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
#endif

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
#define ACT_BF16_SLOTS	2
typedef struct {
	const ann_real	*fp32;
	void		*half;
	size_t		cap;
	int		n;
	int		live;
} act_bf16_slot;
static act_bf16_slot g_act_bf16[ACT_BF16_SLOTS];
static int g_act_bf16_rr;

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

#if CUDART_VERSION >= 11000
	if (bf16) {
		/* Vectorized kernel: 4 elems/thread. */
		blocks = (n + threads * 4 - 1) / (threads * 4);
		if (blocks < 1)
			blocks = 1;
		f32_to_bf16_kernel<<<blocks, threads>>>(
		    (__nv_bfloat16 *)dst, src, n);
		rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
		op_end(prof);
		return rc;
	}
#else
	(void)bf16;
#endif
	f32_to_f16_kernel<<<(n + threads - 1) / threads, threads>>>(
	    (half *)dst, src, n);
	rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
	op_end(prof);
	return rc;
}

extern "C" int gpu_adam_seed_bf16_flat(void *bf16, const ann_real *fp32, int n)
{
	if (!bf16 || !fp32 || n <= 0)
		return -1;
	return half_wt_convert(bf16, fp32, n, 1);
}

static void act_bf16_publish(const ann_real *fp32, int n)
{
	act_bf16_slot *e;
	int i;

	if (!fp32 || n <= 0 || g_compute_prec != GPU_PREC_BF16)
		return;
	e = NULL;
	for (i = 0; i < ACT_BF16_SLOTS; i++) {
		if (g_act_bf16[i].fp32 == fp32) {
			e = &g_act_bf16[i];
			break;
		}
	}
	if (!e) {
		e = &g_act_bf16[g_act_bf16_rr % ACT_BF16_SLOTS];
		g_act_bf16_rr++;
	}
	if (ensure_half_buf(&e->half, &e->cap, (size_t)n * 2) != 0) {
		e->live = 0;
		e->fp32 = NULL;
		return;
	}
	if (half_wt_convert(e->half, fp32, n, 1) != 0) {
		e->live = 0;
		return;
	}
	e->fp32 = fp32;
	e->n = n;
	e->live = 1;
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
			twin_check("act", g_act_bf16[i].half, fp32, (size_t)n, 1);
			return g_act_bf16[i].half;
		}
	}
	return NULL;
}

/* Compat wrappers for LN→GEMM call sites. */
static void ln_bf16_publish(const ann_real *fp32, int n, int bf16)
{
	if (bf16)
		act_bf16_publish(fp32, n);
}

static const void *ln_bf16_consume(const ann_real *fp32, int n)
{
	return act_bf16_get(fp32, n);
}

extern "C" void gpu_half_wt_refresh_all(void)
{
	int i;
	half_wt *e;

	for (i = 0; i < HALF_WT_SLOTS; i++) {
		e = &g_half_wt[i];
		if (!e->fp32 || !e->half || e->n == 0)
			continue;
		if (half_wt_convert(e->half, e->fp32, (int)e->n, e->bf16) != 0)
			e->valid = 0;
		else
			e->valid = 1;
	}
	g_wt_epoch++;
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

static int cublas_gemm_ex_f16(cublasOperation_t opa, cublasOperation_t opb,
	int N, int M, int K, const float *alpha, const void *B, int ldb,
	const void *A, int lda, const float *beta, float *C, int ldc)
{
	if (cublas_lt_gemm_half(opa, opb, N, M, K, alpha, B, ldb, A, lda, beta,
	    C, ldc, 0, CUBLASLT_EPILOGUE_DEFAULT, NULL) == 0)
		return 0;
#if CUDART_VERSION >= 11000
	return cublasGemmEx(g_cublas, opa, opb, N, M, K, alpha, B, CUDA_R_16F,
	    ldb, A, CUDA_R_16F, lda, beta, C, CUDA_R_32F, ldc,
	    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
	    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
#else
	return cublasGemmEx(g_cublas, opa, opb, N, M, K, alpha, B, CUDA_R_16F,
	    ldb, A, CUDA_R_16F, lda, beta, C, CUDA_R_32F, ldc, CUDA_R_32F,
	    CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
#endif
}

#if CUDART_VERSION >= 11000
static int cublas_gemm_ex_bf16(cublasOperation_t opa, cublasOperation_t opb,
	int N, int M, int K, const float *alpha, const void *B, int ldb,
	const void *A, int lda, const float *beta, float *C, int ldc)
{
	if (cublas_lt_gemm_half(opa, opb, N, M, K, alpha, B, ldb, A, lda, beta,
	    C, ldc, 1, CUBLASLT_EPILOGUE_DEFAULT, NULL) == 0)
		return 0;
	/* FAST_16BF: plain COMPUTE_32F can pick SIMT on Blackwell. */
	return cublasGemmEx(g_cublas, opa, opb, N, M, K, alpha, B, CUDA_R_16BF,
	    ldb, A, CUDA_R_16BF, lda, beta, C, CUDA_R_32F, ldc,
	    CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
	    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}
#endif

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
#if CUDART_VERSION < 11000
	if (use_bf16)
		use_bf16 = 0;
#endif
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
#if CUDART_VERSION >= 11000
	if (use_bf16)
		rc = cublas_gemm_ex_bf16(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
		    &alpha, hB_use, N, hA_use, K, &beta, dC, N);
	else
#endif
		rc = cublas_gemm_ex_f16(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
		    &alpha, hB_use, N, hA_use, K, &beta, dC, N);
	pthread_mutex_unlock(&g_gemm_mu);
	return rc;
}

static int cublas_gemm_rm(ann_real *dC, const ann_real *dA, const ann_real *dB,
	int M, int N, int K)
{
	return cublas_gemm_rm_ha(dC, dA, dB, M, N, K, NULL);
}

/* Upload host bias into persistent device scratch (length N). */
static int lt_bias_upload(const ann_real *bias_host, int N)
{
	size_t nb;

	if (!bias_host || N <= 0)
		return -1;
	nb = (size_t)N * sizeof(float);
	if (g_lt_bias_cap < nb) {
		if (g_lt_bias_dev)
			cudaFree(g_lt_bias_dev);
		g_lt_bias_dev = NULL;
		g_lt_bias_cap = 0;
		if (cudaMalloc(&g_lt_bias_dev, nb) != cudaSuccess)
			return -1;
		g_lt_bias_cap = nb;
	}
	return cudaMemcpy(g_lt_bias_dev, bias_host, nb, cudaMemcpyHostToDevice)
	    == cudaSuccess ? 0 : -1;
}

/*
** Row-major C = epi(A B + bias). epi is BIAS or RELU_BIAS. Bias length N.
*/
static int cublas_gemm_rm_epi(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int M, int N, int K, const ann_real *bias_host,
	cublasLtEpilogue_t epi)
{
	const ann_real alpha = 1.0f;
	const ann_real beta = 0.0f;
	size_t nA;
	size_t nB;
	int use_bf16;
	void *hA_use;
	void *hB_use;
	int rc;

	if (!bias_host || cublas_ensure() != 0)
		return -1;
	if (lt_bias_upload(bias_host, N) != 0)
		return -1;
	pthread_mutex_lock(&g_gemm_mu);
	half_scratch_warm();
	use_bf16 = (g_compute_prec == GPU_PREC_BF16);
#if CUDART_VERSION < 11000
	use_bf16 = 0;
#endif
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
	    hB_use, N, hA_use, K, &beta, dC, N, 1, epi, g_lt_bias_dev);
	pthread_mutex_unlock(&g_gemm_mu);
	return rc;
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
static int cublas_gemm_rm_bt(ann_real *dC, const ann_real *dA,
	const ann_real *dB, int M, int N, int K)
{
	const ann_real alpha = 1.0f;
	const ann_real beta = 0.0f;
	size_t nA;
	size_t nB;
	int threads = 256;
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
	** scale on dY via g_fp8_scale_B / g_fp8_scale_B_next.
	*/
	if (g_fp8_dgrad && g_fp8_dgrad_fail < 8
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
			pthread_mutex_unlock(&g_gemm_mu);
			return 0;
		}
		g_fp8_dgrad_fail++;
	}
#endif
	if (ensure_half_buf(&g_half_A, &g_half_A_cap, nA * 2) != 0) {
		pthread_mutex_unlock(&g_gemm_mu);
		return -1;
	}
	{
		int use_bf16 = (g_compute_prec == GPU_PREC_BF16);
		void *hB_use;

#if CUDART_VERSION < 11000
		use_bf16 = 0;
#endif
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
#if CUDART_VERSION >= 11000
		if (use_bf16) {
			if (half_wt_convert(g_half_A, dA, (int)nA, 1) != 0) {
				pthread_mutex_unlock(&g_gemm_mu);
				return -1;
			}
			rc = cublas_gemm_ex_bf16(CUBLAS_OP_T, CUBLAS_OP_N, N,
			    M, K, &alpha, hB_use, K, g_half_A, K, &beta, dC, N);
			pthread_mutex_unlock(&g_gemm_mu);
			return rc;
		}
#endif
		f32_to_f16_kernel<<<(int)((nA + threads - 1) / threads),
		    threads>>>((half *)g_half_A, dA, (int)nA);
		rc = cublas_gemm_ex_f16(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
		    &alpha, hB_use, K, g_half_A, K, &beta, dC, N);
		pthread_mutex_unlock(&g_gemm_mu);
		return rc;
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
	int threads = 256;
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
#if CUDART_VERSION >= 12000
	/*
	** FP8 wgrad: C [M, N] (+)= A^T [M, T] * B [T, N].
	** In col-major, C^T [N, M] (+)= B^T [N, T] * A [T, M].
	** To satisfy sm_89/sm_120 (OP_T, OP_N) with K=T (must be %16==0):
	**   - Transpose+quantize B [T, N] row-major -> g_fp8_B [N, T] row-major
	**     (= [T, N] col-major, ldb = T) so OP_T yields [N, T].
	**   - Transpose+quantize A [T, M] row-major -> g_fp8_A [M, T] row-major
	**     (= [T, M] col-major, lda = T) so OP_N yields [T, M].
	*/
	/*
	** FP8 wgrad needs two layout transposes. On sm_120 that costs more than
	** the GEMM saves, so it stays off unless ANN_FP8_WGRAD=1.
	*/
	{
		static int s_wgrad = -1;

		if (s_wgrad < 0) {
			const char *e = getenv("ANN_FP8_WGRAD");

			s_wgrad = (e && e[0] == '1') ? 1 : 0;
		}
		if (!s_wgrad)
			goto fp8_wgrad_skip;
	}
	if (g_fp8_dgrad && (T & 15) == 0 && (M & 15) == 0 && (N & 15) == 0) {
		fp8_act_slot *wa = fp8_act_scale(dA, 0);
		fp8_act_slot *wb = fp8_act_scale(dB, 1);

		if (wa && wb && fp8_scales_ensure() == 0
		    && ensure_fp8_buf(&g_fp8_A, &g_fp8_A_cap, nA) == 0
		    && ensure_fp8_buf(&g_fp8_B, &g_fp8_B_cap, nB) == 0
		    && fp8_quant_transpose_tensor(g_fp8_A, FP8_SCALE_CUR(wa),
		    FP8_SCALE_NEXT(wa), dA, T, M, 0) == 0
		    && fp8_quant_transpose_tensor(g_fp8_B, FP8_SCALE_CUR(wb),
		    FP8_SCALE_NEXT(wb), dB, T, N, 1) == 0
		    && cublas_lt_gemm_fp8_ex(CUBLAS_OP_T, CUBLAS_OP_N, N, M, T,
		    &alpha, g_fp8_B, T, FP8_SCALE_CUR(wb), CUDA_R_8F_E5M2,
		    g_fp8_A, T, FP8_SCALE_CUR(wa), CUDA_R_8F_E4M3, &beta, dC, N)
		    == 0) {
			wa->cur ^= 1;
			wb->cur ^= 1;
			pthread_mutex_unlock(&g_gemm_mu);
			return 0;
		}
	}
#endif
fp8_wgrad_skip:
	if (ensure_half_buf(&g_half_A, &g_half_A_cap, nA * 2) != 0
	    || ensure_half_buf(&g_half_B, &g_half_B_cap, nB * 2) != 0) {
		pthread_mutex_unlock(&g_gemm_mu);
		return -1;
	}
#if CUDART_VERSION >= 11000
	if (g_compute_prec == GPU_PREC_BF16) {
		const void *hA_use = act_bf16_get(dA, (int)nA);
		const void *hB_use = act_bf16_get(dB, (int)nB);

		if (!hA_use) {
			if (half_wt_convert(g_half_A, dA, (int)nA, 1) != 0) {
				pthread_mutex_unlock(&g_gemm_mu);
				return -1;
			}
			hA_use = g_half_A;
		}
		if (!hB_use) {
			if (half_wt_convert(g_half_B, dB, (int)nB, 1) != 0) {
				pthread_mutex_unlock(&g_gemm_mu);
				return -1;
			}
			hB_use = g_half_B;
		}
		rc = cublas_gemm_ex_bf16(CUBLAS_OP_N, CUBLAS_OP_T, N, M, T,
		    &alpha, hB_use, N, hA_use, M, &beta, dC, N);
		pthread_mutex_unlock(&g_gemm_mu);
		return rc;
	}
#endif
	f32_to_f16_kernel<<<(int)((nA + threads - 1) / threads), threads>>>(
	    (half *)g_half_A, dA, (int)nA);
	f32_to_f16_kernel<<<(int)((nB + threads - 1) / threads), threads>>>(
	    (half *)g_half_B, dB, (int)nB);
	rc = cublas_gemm_ex_f16(CUBLAS_OP_N, CUBLAS_OP_T, N, M, T, &alpha,
	    g_half_B, N, g_half_A, M, &beta, dC, N);
	pthread_mutex_unlock(&g_gemm_mu);
	return rc;
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

extern "C" int gpu_available(void)
{
	int n;

	if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0)
		return 0;
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

extern "C" void *gpu_malloc(size_t nbytes)
{
	void *p;

	if (nbytes == 0 || !gpu_available())
		return NULL;
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
#if CUDART_VERSION >= 12000 && defined(ANN_CUBLAS)
	fp8_wt_invalidate((const ann_real *)p);
	fp8_act_forget(p);
#endif
}

extern "C" void gpu_free(void *p)
{
	if (!p)
		return;
	twins_forget_dev(p);
	cudaFree(p);
}

extern "C" int gpu_memcpy_h2d(void *dst, const void *src, size_t nbytes)
{
	if (!dst || !src || nbytes == 0)
		return -1;
	/* Async on default stream; same-stream kernels/copies order themselves. */
	return cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, 0)
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
	return cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, 0)
	    == cudaSuccess ? 0 : -1;
}

extern "C" int gpu_stream_sync(void)
{
	return cudaStreamSynchronize(0) == cudaSuccess ? 0 : -1;
}

extern "C" int gpu_memcpy_d2d(void *dst, const void *src, size_t nbytes)
{
	if (!dst || !src || nbytes == 0)
		return -1;
	return cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, 0)
	    == cudaSuccess ? 0 : -1;
}

extern "C" int gpu_memset(void *dst, int value, size_t nbytes)
{
	if (!dst || nbytes == 0)
		return -1;
	return cudaMemset(dst, value, nbytes) == cudaSuccess ? 0 : -1;
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
	cudaStreamCaptureStatus st;

	if (cudaStreamIsCapturing(0, &st) != cudaSuccess)
		return 0;
	return st == cudaStreamCaptureStatusActive;
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
		(void)cudaMemset(a->dev, 0, nb);
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
	(void)cudaMemset(g_colsum_acc[free_slot].dev, 0, nb);
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
	if (cudaEventRecord(e->e0, 0) != cudaSuccess) {
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
	if (cudaEventRecord(e->e0, 0) != cudaSuccess) {
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
	if (cudaEventRecord(g_op_ev[i].e1, 0) != cudaSuccess)
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
	fprintf(stderr, "[opprof] %ld microbatches, %.1f ms/microbatch of GPU op time (convert counted inside gemm rows)\n",
	    mb, tot / (double)mb);
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
		gemm_kernel<<<grid, block>>>(dA, dB, dC, a->rows, b->cols,
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
	if (cublas_gemm_rm_bt(dC, dA, dB, a->rows, b->rows, a->cols) != 0) {
#endif
		block = dim3(16, 16);
		grid = dim3((c->cols + 15) / 16, (c->rows + 15) / 16);
		gemm_bt_kernel<<<grid, block>>>(dA, dB, dC, a->rows, b->rows,
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
		gemm_at_kernel<<<grid, block>>>(dA, dB, dC, a->rows, a->cols,
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
		outer_add_kernel<<<grid, block>>>(dW, dA, dB, a->rows, a->cols,
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
	** cudaPointerGetAttributes is illegal during graph capture and was
	** segfaulting --cuda-graphs. After warmup, large buffers are device
	** activations; small ones go through bias_cache.
	*/
	if (stream_is_capturing())
		return 1;
	if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
		(void)cudaGetLastError();
		return 0;
	}
	return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
}

/* Small host vectors (bias/LN) reused across launches until g_wt_epoch bumps. */
#define BIAS_CACHE_SLOTS	64
typedef struct {
	const void	*host;
	void		*dev;
	size_t		bytes;
	unsigned	epoch;
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
				    cudaMemcpyHostToDevice, 0) != cudaSuccess)
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
	if (cudaMemcpyAsync(e->dev, host, bytes, cudaMemcpyHostToDevice, 0)
	    != cudaSuccess) {
		cudaFree(e->dev);
		e->host = NULL;
		e->dev = NULL;
		return NULL;
	}
	e->epoch = g_wt_epoch;
	return e->dev;
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
			g_bias_cache[i].dev = NULL;
			g_bias_cache[i].host = NULL;
			g_bias_cache[i].bytes = 0;
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
	(void)cudaMemset(e->dev, 0, bytes);
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
	__nv_fp8_storage_t *y8, const float *scale8, float *amax8)
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
	ann_real *dgamma, ann_real *dbeta, int d)
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
		g = dy[off + j] * gamma[j];
		dx[off + j] = (g - (sum_dy + x_hat[off + j] * sum_dyx) / (ann_real)d)
		    * inv;
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
	const ann_real *x2, int rows, int cols)
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
	atomicAdd(&db[c], s);
}

__global__ static void bias_add_kernel(ann_real *x, const ann_real *bias,
	int n, int cols)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		x[i] += bias[i % cols];
}

__global__ static void relu_fwd_kernel(ann_real *x, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n && x[i] < 0.0f)
		x[i] = 0.0f;
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
	softmax_rows_kernel<<<n_rows, ew_block_for(n_cols)>>>(r.dev, n_cols);
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
	softmax_bwd_kernel<<<n_rows, ew_block_for(n_cols)>>>(rdx.dev, ry.dev,
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
** Same causal softmax as attn_fwd_kernel: S_ij = -inf when j > i or the
** position is outside seqlens. Writes L = m + log(sum exp(S-m)) only.
** The row lives in SRAM. No T×T score tensor in HBM.
*/
__global__ static void attn_lse_kernel(ann_real *Lstat, const ann_real *q,
	const ann_real *k, int T, int H, int dh, ann_real scale,
	const int *seqlens)
{
	extern __shared__ ann_real sm[];
	__shared__ ann_real red[ATT_BLOCK];
	ann_real *qi = sm;
	ann_real *sc = sm + dh;
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
		} else
			acc = -1e30f;
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
	if (threadIdx.x == 0)
		Lstat[blockIdx.x] = mx + __logf(sum);
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
/*
** One-shot wall clock of the tiled flash backward. -1 until measured.
** Not recorded inside CUDA graph capture (a sync there is illegal, and
** autotune stays outside the timed window).
*/
static float g_fa_bwd_ms = -1.f;
static int g_fa_time_done;
static cudaEvent_t g_fa_ev0;
static cudaEvent_t g_fa_ev1;
static int g_fa_ev_ok;

static int fa_time_begin(void)
{
	cudaStreamCaptureStatus cap;

	if (g_fa_time_done)
		return 0;
	cap = cudaStreamCaptureStatusNone;
	if (cudaStreamIsCapturing(0, &cap) != cudaSuccess
	    || cap != cudaStreamCaptureStatusNone)
		return 0;
	if (!g_fa_ev_ok) {
		if (cudaEventCreate(&g_fa_ev0) != cudaSuccess
		    || cudaEventCreate(&g_fa_ev1) != cudaSuccess) {
			g_fa_time_done = 1;
			return 0;
		}
		g_fa_ev_ok = 1;
	}
	if (cudaEventRecord(g_fa_ev0, 0) != cudaSuccess)
		return 0;
	return 1;
}

static void fa_time_end(int timing)
{
	if (!timing)
		return;
	g_fa_time_done = 1;
	if (cudaEventRecord(g_fa_ev1, 0) != cudaSuccess)
		return;
	if (cudaEventSynchronize(g_fa_ev1) != cudaSuccess)
		return;
	if (cudaEventElapsedTime(&g_fa_bwd_ms, g_fa_ev0, g_fa_ev1)
	    != cudaSuccess) {
		g_fa_bwd_ms = -1.f;
		return;
	}
	fprintf(stderr, "[attn] flash bwd %.3f ms\n", g_fa_bwd_ms);
}

extern "C" float gpu_attn_flash_bwd_ms(void)
{
	return g_fa_bwd_ms;
}

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
	attn_fwd_kernel<<<B * H * T, ATT_BLOCK, smem>>>(dc, dw, dq, dk, dv,
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
	attn_fwd_kernel<<<B * H * T, ATT_BLOCK, smem>>>(dc, NULL, dq, dk, dv,
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
** wrote. gpu_attn_flash_bwd_ms() records this kernel's wall clock once,
** outside graph capture; consult that before any SDPA port. Attention
** is about 1% of FLOPs at T=64, which only orders the work. T is not
** shortened and causal masking stays.
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
	int timing;
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
	timing = 0;
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
	if (cudaMemset(a_dk, 0, nbytes) != cudaSuccess
	    || cudaMemset(a_dv, 0, nbytes) != cudaSuccess)
		goto tiled_bwd_out;
	smem = (size_t)(4 * dh + 2 * FA_BC * dh) * sizeof(ann_real);
	timing = fa_time_begin();
	attn_tiled_bwd_kernel<<<B * H * T, ATT_BLOCK, smem>>>(a_dq, a_dk, a_dv,
	    a_dc, a_o, a_q, a_k, a_v, a_L, T, H, dh,
	    (ann_real)(1.0 / sqrt((double)dh)),
	    g_att_seqlens_host ? g_att_seqlens_dev : NULL);
	fa_time_end(timing);
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

extern "C" int gpu_attn_recompute_w(mat *lse, const mat *q, const mat *k,
	int B, int T, int H, int dh)
{
	ann_real *dq;
	ann_real *dk;
	ann_real *dL;
	int oq;
	int ok;
	int oL;
	int rc;
	size_t smem;

	if (!lse || !q || !k || !att_args_ok(B, T, H, dh) || !lse->d_data)
		return -1;
	if (q->rows != B * T || lse->rows != B * H * T || lse->cols != 1)
		return -1;
	pthread_mutex_lock(&g_att_mu);
	rc = -1;
	oq = ok = oL = 0;
	dq = dk = dL = NULL;
	if (ensure_dev(q, &dq, &oq) != 0)
		goto rec_out;
	if (ensure_dev(k, &dk, &ok) != 0)
		goto rec_out;
	if (ensure_dev_out(lse, &dL, &oL) != 0)
		goto rec_out;
	smem = (size_t)(dh + T) * sizeof(ann_real);
	attn_lse_kernel<<<B * H * T, ATT_BLOCK, smem>>>(dL, dq, dk, T, H, dh,
	    (ann_real)(1.0 / sqrt((double)dh)),
	    g_att_seqlens_host ? g_att_seqlens_dev : NULL);
	if (launch_done() == 0 && finish_out(lse, dL, oL) == 0)
		rc = 0;
rec_out:
	if (oq && dq)
		gpu_free(dq);
	if (ok && dk)
		gpu_free(dk);
	if (oL && dL)
		gpu_free(dL);
	pthread_mutex_unlock(&g_att_mu);
	return rc;
}

extern "C" int gpu_bias_relu_forward(ann_real *x, const ann_real *bias,
	int rows, int cols)
{
	OpScope _op(OP_ELEMWISE);
	ew_ref rx;
	ew_ref rb;
	size_t nb;
	int n;
	int rc;
	int dn;

	if (!x || !bias || rows <= 0 || cols <= 0 || !gpu_available())
		return -1;
	n = rows * cols;
	nb = (size_t)n * sizeof(ann_real);
	pthread_mutex_lock(&g_ew_mu);
	if (ew_acquire(&rx, 0, x, nb, 1, 1) != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	if (ew_acquire_param(&rb, 1, bias, (size_t)cols * sizeof(ann_real))
	    != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	bias_add_kernel<<<(n + EW_BLOCK - 1) / EW_BLOCK, EW_BLOCK>>>(rx.dev,
	    rb.dev, n, cols);
	if (ew_launch_ok() != 0) {
		pthread_mutex_unlock(&g_ew_mu);
		return -1;
	}
	relu_fwd_kernel<<<(n + EW_BLOCK - 1) / EW_BLOCK, EW_BLOCK>>>(rx.dev, n);
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
	if (rc == 0)
		act_bf16_publish(x, n);
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
	bias_relu_dst_kernel<<<(n + EW_BLOCK - 1) / EW_BLOCK, EW_BLOCK>>>(
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
		if (with_relu && c->d_data)
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
	attn_bwd_rows_kernel<<<B * H * T, ATT_BLOCK, smem>>>(a_dq, g_att_ds,
	    a_dc, w->d_data, a_k, a_v, T, H, dh,
	    (ann_real)(1.0 / sqrt((double)dh)));
	attn_bwd_cols_kernel<<<B * H * T, ATT_BLOCK>>>(a_dk, a_dv, a_dc,
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

#if CUDART_VERSION >= 11000
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
#endif

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
static __nv_fp8_storage_t *g_tc_fp8_w;
static size_t g_tc_fp8_w_cap;

/* E4M3 copy of attention probabilities. NULL when --fp8 is off. */
static __nv_fp8_storage_t *tc_fp8_softmax_buf(int batch, int T)
{
	size_t n;

	if (!g_fp8_fwd || batch <= 0 || T <= 0)
		return NULL;
	n = (size_t)batch * (size_t)T * (size_t)T;
	if (g_tc_fp8_w_cap < n) {
		if (g_tc_fp8_w)
			cudaFree(g_tc_fp8_w);
		g_tc_fp8_w = NULL;
		g_tc_fp8_w_cap = 0;
		if (cudaMalloc(&g_tc_fp8_w, n) != cudaSuccess)
			return NULL;
		g_tc_fp8_w_cap = n;
	}
	return g_tc_fp8_w;
}

__global__ static void attn_tc_softmax_kernel(ann_real *w,
#if CUDART_VERSION >= 11000
	__nv_bfloat16 *w_bf16,
#else
	void *w_bf16,
#endif
	int T, int H, ann_real scale, const int *seqlens,
	__nv_fp8_storage_t *w8)
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
	__nv_fp8_storage_t *w8r = w8 ? (w8 + base) : NULL;
#if CUDART_VERSION >= 11000
	__nv_bfloat16 *wbr = w_bf16 ? (w_bf16 + base) : NULL;
#endif

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
		/* E4M3 holds [0,1] probabilities; 1.0 is exact. PV stays FP32. */
		if (w8r)
			w8r[j] = __nv_cvt_float_to_fp8(p, __NV_SATFINITE, __NV_E4M3);
#if CUDART_VERSION >= 11000
		if (wbr)
			wbr[j] = __float2bfloat16(p);
#else
		(void)w_bf16;
#endif
	}
}

/* Softmax bwd: dP = P * (dP - sum_k P_k * dP_k) in place on dp; uses w. */
__global__ static void attn_tc_softmax_bwd_kernel(ann_real *dp,
#if CUDART_VERSION >= 11000
	__nv_bfloat16 *dp_bf16,
#else
	void *dp_bf16,
#endif
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
#if CUDART_VERSION >= 11000
	__nv_bfloat16 *dbr = dp_bf16 ? (dp_bf16 + base) : NULL;
#endif

	if (L < 1)
		L = 1;
	if (L > T)
		L = T;
	if (i >= T)
		return;
	if (i >= L) {
		for (j = (int)threadIdx.x; j < T; j += (int)blockDim.x) {
			drow[j] = 0.0f;
#if CUDART_VERSION >= 11000
			if (dbr)
				dbr[j] = __float2bfloat16(0.0f);
#endif
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
#if CUDART_VERSION >= 11000
		if (dbr)
			dbr[j] = __float2bfloat16(g);
#else
		(void)dp_bf16;
#endif
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
/* QK from already-packed BF16/FP16 twins in g_tc_half_{q,k}. */
static int tc_gemm_qk_half(ann_real *scores, int T, int dh, int batch, int bf16)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	long long stride = (long long)T * dh;
	long long stride_s = (long long)T * T;

	if (cublas_ensure() != 0)
		return -1;
#if CUDART_VERSION >= 11000
	if (bf16)
		return cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_T,
		    CUBLAS_OP_N, T, T, dh, &alpha, g_tc_half_k, CUDA_R_16BF,
		    dh, stride, g_tc_half_q, CUDA_R_16BF, dh, stride, &beta,
		    scores, CUDA_R_32F, T, stride_s, batch,
		    CUBLAS_COMPUTE_32F_FAST_16BF,
		    CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS
		    ? 0 : -1;
	return cublasGemmStridedBatchedEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
	    T, T, dh, &alpha, g_tc_half_k, CUDA_R_16F, dh, stride, g_tc_half_q,
	    CUDA_R_16F, dh, stride, &beta, scores, CUDA_R_32F, T, stride_s,
	    batch, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
	    == CUBLAS_STATUS_SUCCESS ? 0 : -1;
#else
	(void)scores;
	(void)T;
	(void)dh;
	(void)batch;
	(void)bf16;
	return -1;
#endif
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
#if CUDART_VERSION >= 11000
	if (g_compute_prec == GPU_PREC_BF16) {
		int blocks = (int)((n + (size_t)threads * 4 - 1)
		    / ((size_t)threads * 4));

		if (blocks < 1)
			blocks = 1;
		f32_to_bf16_kernel<<<blocks, threads>>>(
		    (__nv_bfloat16 *)g_tc_half_q, Q, (int)n);
		f32_to_bf16_kernel<<<blocks, threads>>>(
		    (__nv_bfloat16 *)g_tc_half_k, K, (int)n);
		return tc_gemm_qk_half(scores, T, dh, batch, 1);
	}
#endif
	f32_to_f16_kernel<<<(int)((n + threads - 1) / threads), threads>>>(
	    (half *)g_tc_half_q, Q, (int)n);
	f32_to_f16_kernel<<<(int)((n + threads - 1) / threads), threads>>>(
	    (half *)g_tc_half_k, K, (int)n);
#if CUDART_VERSION >= 11000
	return tc_gemm_qk_half(scores, T, dh, batch, 0);
#else
	return cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
	    T, T, dh, &alpha, K, dh, stride, Q, dh, stride, &beta, scores, T,
	    stride_s, batch) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
#endif
}

#if CUDART_VERSION >= 11000
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
#endif

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
#if CUDART_VERSION >= 11000
	if (g_compute_prec == GPU_PREC_BF16) {
		if (ensure_half_buf(&g_tc_half_q, &g_tc_half_q_cap,
		    (size_t)npack * 2) != 0
		    || ensure_half_buf(&g_tc_half_k, &g_tc_half_k_cap,
		    (size_t)npack * 2) != 0)
			goto tc_fwd_out;
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_bf16_kernel<<<(npack + threads - 1) / threads,
		    threads>>>((__nv_bfloat16 *)g_tc_half_q, dq, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_bf16_kernel<<<(npack + threads - 1) / threads,
		    threads>>>((__nv_bfloat16 *)g_tc_half_k, dk, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads>>>(g_tc_v, dv, B, T, H, dh); }
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_qk_half(dw, T, dh, batch, 1) != 0)
			goto tc_fwd_out;
	} else
#endif
	{
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads>>>(g_tc_q, dq, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads>>>(g_tc_k, dk, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads>>>(g_tc_v, dv, B, T, H, dh); }
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_qk_batched(dw, g_tc_q, g_tc_k, T, dh, batch) != 0)
			goto tc_fwd_out;
	}
	scale = (ann_real)(1.0 / sqrt((double)dh));
	sgrid = dim3(batch, T, 1);
	{
		int nth = 32;

		while (nth < T && nth < ATT_BLOCK)
			nth <<= 1;
		{ OpNest _n(OP_ATT_SMX_ID); attn_tc_softmax_kernel<<<sgrid, nth>>>(dw, NULL, T, H, scale,
		    g_att_seqlens_host ? g_att_seqlens_dev : NULL,
		    tc_fp8_softmax_buf(batch, T)); }
	}
	if (OpNest(OP_ATT_GEMM_ID), tc_gemm_pv_batched(g_tc_o, dw, g_tc_v, T, dh, batch) != 0)
		goto tc_fwd_out;
	{ OpNest _n(OP_ATT_PACK_ID); attn_unpack_heads_kernel<<<(npack + threads - 1) / threads, threads>>>(
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
	{
		int nth = 32;

		/* Pack Q/K/V and dO. */
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads>>>(g_tc_q, a_q, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads>>>(g_tc_k, a_k, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads>>>(g_tc_v, a_v, B, T, H, dh); }
		{ OpNest _n(OP_ATT_PACK_ID); attn_pack_heads_kernel<<<(npack + threads - 1) / threads,
		    threads>>>(g_tc_o, a_dc, B, T, H, dh); }
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_qk_batched(dw, g_tc_q, g_tc_k, T, dh, batch) != 0)
			goto tc_bwd_out;
		while (nth < T && nth < ATT_BLOCK)
			nth <<= 1;
		{ OpNest _n(OP_ATT_SMX_ID); attn_tc_softmax_kernel<<<sgrid, nth>>>(dw, NULL, T, H,
		    (ann_real)scale,
		    g_att_seqlens_host ? g_att_seqlens_dev : NULL,
		    tc_fp8_softmax_buf(batch, T)); }
		/* dP = dO V^T (needs V); then dV = W^T dO (overwrites V). */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_do_vt(g_tc_dp, g_tc_o, g_tc_v, T, dh, batch) != 0)
			goto tc_bwd_out;
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_wt_do(g_tc_v, dw, g_tc_o, T, dh, batch) != 0)
			goto tc_bwd_out;
		{ OpNest _n(OP_ATT_SMX_ID); attn_tc_softmax_bwd_kernel<<<sgrid, nth>>>(g_tc_dp, NULL, dw,
		    T, H, g_att_seqlens_host ? g_att_seqlens_dev : NULL); }
		/* dQ = dS K -> g_tc_o; dK = dS^T Q -> g_tc_k (Q still in g_tc_q). */
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_ds_k(g_tc_o, g_tc_dp, g_tc_k, T, dh, batch, scale)
		    != 0)
			goto tc_bwd_out;
		if (OpNest(OP_ATT_GEMM_ID), tc_gemm_dst_q(g_tc_k, g_tc_dp, g_tc_q, T, dh, batch, scale)
		    != 0)
			goto tc_bwd_out;
	}
	{ OpNest _n(OP_ATT_PACK_ID); attn_unpack_heads_kernel<<<(npack + threads - 1) / threads, threads>>>(
	    a_dq, g_tc_o, B, T, H, dh); }
	{ OpNest _n(OP_ATT_PACK_ID); attn_unpack_heads_kernel<<<(npack + threads - 1) / threads, threads>>>(
	    a_dk, g_tc_k, B, T, H, dh); }
	{ OpNest _n(OP_ATT_PACK_ID); attn_unpack_heads_kernel<<<(npack + threads - 1) / threads, threads>>>(
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
	** Three GemmEx sharing one BF16/FP16 convert of X (same for Wq/Wk/Wv).
	** One fat X@[Wq|Wk|Wv] was tried; pack/split overhead canceled the win
	** at T=64 / D=1024 on this card.
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
	if (ensure_dev_out(k, &dk, &ok) != 0)
		goto qkv_triple;
	if (ensure_dev_out(v, &dv, &ov) != 0)
		goto qkv_triple;
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
#if CUDART_VERSION < 11000
	use_bf16 = 0;
#endif
	{
		const void *cached = use_bf16 ? ln_bf16_consume(dx, (int)nA)
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
		** Overlap the three independent projections on dedicated
		** streams. Falls back to serial rm_ha if setup fails.
		*/
		{
			void *hwq;
			void *hwk;
			void *hwv;
			const float alpha = 1.0f;
			const float beta = 0.0f;
			int i;
			int par = 0;

#if CUDART_VERSION >= 11000
			if (use_bf16 && !stream_is_capturing()
			    && qkv_par_ensure() == 0) {
				hwq = half_wt_get(dwq, (size_t)D * (size_t)D,
				    1);
				hwk = half_wt_get(dwk, (size_t)D * (size_t)D,
				    1);
				hwv = half_wt_get(dwv, (size_t)D * (size_t)D,
				    1);
				if (hwq && hwk && hwv
				    && cudaEventRecord(g_qkv_ready, 0)
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
#endif
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
	pack_seq_rows_kernel<<<(n + threads - 1) / threads, threads>>>(dst, src,
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
	unpack_seq_rows_kernel<<<(n + threads - 1) / threads, threads>>>(dst,
	    src, B, T, D);
	return launch_done();
}

static int gpu_layernorm_forward_ex_locked(ann_real *y, const ann_real *x,
	const ann_real *gamma, const ann_real *beta, ann_real *x_hat,
	ann_real *mean, ann_real *inv_std, int T, int d, double eps)
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
		ln_fwd_kernel<<<T, bs, (size_t)d * sizeof(ann_real)>>>(ry.dev,
		    rx.dev, rg.dev, rb.dev, x_hat ? rh.dev : NULL,
		    mean ? rm.dev : NULL, inv_std ? ri.dev : NULL, d,
		    (ann_real)eps, y8, sc8, am8);
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

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_layernorm_forward_ex_locked(y, x, gamma, beta, x_hat, mean, inv_std, T, d, eps);
	pthread_mutex_unlock(&g_ew_mu);
	if (rc == 0 && y && T > 0 && d > 0
	    && g_compute_prec == GPU_PREC_BF16)
		ln_bf16_publish(y, T * d, 1);
	return rc;
}

extern "C" int gpu_layernorm_forward(ann_real *y, const ann_real *x,
	const ann_real *gamma, const ann_real *beta, ann_real *x_hat,
	ann_real *inv_std, int T, int d)
{
	return gpu_layernorm_forward_ex(y, x, gamma, beta, x_hat, NULL, inv_std,
	    T, d, 1e-5);
}

static int gpu_layernorm_backward_locked(mat *dx, const mat *dy, const ann_real *gamma,
	const ann_real *x_hat, const ann_real *inv_std, int T, int d,
	ann_real *dgamma, ann_real *dbeta)
{
	ew_ref rdx;
	ew_ref rdy;
	ew_ref rg;
	ew_ref rh;
	ew_ref ri;
	colsum_acc *acc_g;
	colsum_acc *acc_b;
	const ann_real *dyp;
	size_t mb;
	size_t vb;
	size_t gb;
	int did;
	int rc;

	if (!dx || !dy || !gamma || !x_hat || !inv_std || !dgamma || !dbeta)
		return -1;
	if (!gpu_available() || T <= 0 || d <= 0)
		return -1;
	mb = (size_t)T * (size_t)d * sizeof(ann_real);
	vb = (size_t)T * sizeof(ann_real);
	gb = (size_t)d * sizeof(ann_real);
	/* Prefer device twins when authoritative (R1 residency). */
	dyp = (dy->d_data && dy->on_device != MAT_DEV_HOST) ? dy->d_data : dy->data;
	if (dx->d_data) {
		if (ew_acquire(&rdx, 0, dx->d_data, mb, 0, 0) != 0)
			return -1;
	} else if (ew_acquire(&rdx, 0, dx->data, mb, 0, 1) != 0)
		return -1;
	if (ew_acquire(&rdy, 1, dyp, mb, 1, 0) != 0
	    || ew_acquire_param(&rg, 2, gamma, gb) != 0
	    || ew_acquire(&rh, 3, x_hat, mb, 1, 0) != 0)
		return -1;
	ri.dev = (ann_real *)(ew_is_dev(inv_std) ? (void *)inv_std
	    : devvec_get(inv_std, vb));
	if (!ri.dev)
		return -1;
	/*
	** dgamma / dbeta accumulate on the device (same accumulators as the
	** FFN bias grads) and reach the host at gpu_device_sync: staging them
	** through host copies cost two full-pipeline syncs per layer.
	*/
	acc_g = colsum_acc_get(dgamma, d);
	acc_b = colsum_acc_get(dbeta, d);
	if (!acc_g || !acc_b)
		return -1;
	ln_bwd_kernel<<<T, ew_block_for(d)>>>(rdx.dev, rdy.dev, rg.dev, rh.dev,
	    ri.dev, NULL, NULL, d);
	if (ew_launch_ok() != 0)
		return -1;
	{
		dim3 grid((d + EW_BLOCK - 1) / EW_BLOCK,
		    (T + COLSUM_SLAB - 1) / COLSUM_SLAB);

		col_sum_tiled_kernel<<<grid, EW_BLOCK>>>(acc_b->dev, rdy.dev,
		    NULL, T, d);
		col_sum_tiled_kernel<<<grid, EW_BLOCK>>>(acc_g->dev, rdy.dev,
		    rh.dev, T, d);
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
	rc = gpu_layernorm_backward_locked(dx, dy, gamma, x_hat, inv_std, T, d, dgamma, dbeta);
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
	bias_add_kernel<<<(int)((n + 255) / 256), 256>>>(rx.dev, rb.dev, (int)n,
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

static int gpu_relu_forward_locked(ann_real *x, int n)
{
	ew_ref rx;
	int rc;

	if (!x || n <= 0 || !gpu_available())
		return -1;
	if (ew_acquire(&rx, 0, x, (size_t)n * sizeof(ann_real), 1, 1) != 0)
		return -1;
	relu_fwd_kernel<<<(n + 255) / 256, 256>>>(rx.dev, n);
	if (ew_launch_ok() != 0)
		return -1;
	rc = ew_release(&rx);
	if (rc < 0)
		return -1;
	return ew_finish_sync(rc) == 0 ? 0 : -1;
}

extern "C" int gpu_relu_forward(ann_real *x, int n)
{
	int rc;

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_relu_forward_locked(x, n);
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
	relu_bwd_kernel<<<(n + 255) / 256, 256>>>(rdx.dev, rr.dev, n);
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

extern "C" int gpu_grad_sumsq(const ann_real *d_x, int n, double *out)
{
	OpScope _op(OP_ADAM);
	static double *s_acc;

	if (!d_x || n <= 0 || !out || !gpu_available())
		return -1;
	if (!s_acc && cudaMalloc((void **)&s_acc, sizeof(double)) != cudaSuccess) {
		s_acc = NULL;
		return -1;
	}
	if (cudaMemset(s_acc, 0, sizeof(double)) != cudaSuccess)
		return -1;
	sumsq_kernel<<<(n + 255) / 256, 256>>>(d_x, n, s_acc);
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

#if CUDART_VERSION >= 11000
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

/* Group-128 E4M3 moments and an E4M3 master of the updated weight. */
#define ADAM_FP8_GROUP	128

__device__ static float adam_grp_max(float v)
{
	__shared__ float s[ADAM_FP8_GROUP];
	int m;

	s[threadIdx.x] = v;
	__syncthreads();
	for (m = ADAM_FP8_GROUP >> 1; m > 0; m >>= 1) {
		if ((int)threadIdx.x < m && s[threadIdx.x + m] > s[threadIdx.x])
			s[threadIdx.x] = s[threadIdx.x + m];
		__syncthreads();
	}
	v = s[0];
	__syncthreads();
	return v;
}

__device__ static float adam_fp8_load(__nv_fp8_storage_t q, float scale)
{
	__half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);

	return __half2float(h) * scale;
}

__global__ static void adam_kernel_fp8(ann_real *param,
	__nv_bfloat16 *param_bf16, ann_real *grad, ann_real *m, ann_real *v,
	__nv_fp8_storage_t *p8, __nv_fp8_storage_t *m8, __nv_fp8_storage_t *v8,
	float *ps, float *ms, float *vs, int n, int live, ann_real lr,
	ann_real beta1, ann_real beta2, ann_real eps, ann_real bc1, ann_real bc2,
	ann_real gscale)
{
	int i = (int)blockIdx.x * ADAM_FP8_GROUP + (int)threadIdx.x;
	int on = i < n;
	ann_real mold = 0.0f;
	ann_real vold = 0.0f;
	ann_real newm = 0.0f;
	ann_real newv = 0.0f;
	ann_real p = 0.0f;
	ann_real am;
	ann_real av;
	ann_real ap;
	__shared__ float sm;
	__shared__ float sv;
	__shared__ float sp;

	if (on) {
		ann_real g;
		ann_real mhat;
		ann_real vhat;

		if (live) {
			mold = adam_fp8_load(m8[i], ms[blockIdx.x]);
			vold = adam_fp8_load(v8[i], vs[blockIdx.x]);
		} else {
			mold = m[i];
			vold = v[i];
		}
		g = grad[i] * gscale;
		newm = beta1 * mold + (1.0f - beta1) * g;
		newv = beta2 * vold + (1.0f - beta2) * g * g;
		mhat = newm / bc1;
		vhat = newv / bc2;
		p = param[i] - lr * mhat / (sqrtf(vhat) + eps);
		param[i] = p;
		if (param_bf16)
			param_bf16[i] = __float2bfloat16(p);
		m[i] = newm;
		v[i] = newv;
		grad[i] = 0.0f;
	}
	am = adam_grp_max(on ? fabsf(newm) : 0.0f);
	av = adam_grp_max(on ? fabsf(newv) : 0.0f);
	ap = adam_grp_max(on ? fabsf(p) : 0.0f);
	if (threadIdx.x == 0) {
		sm = am / 448.f;
		sv = av / 448.f;
		sp = ap / 448.f;
		if (sm < 1e-12f)
			sm = 1e-12f;
		if (sv < 1e-12f)
			sv = 1e-12f;
		if (sp < 1e-12f)
			sp = 1e-12f;
		ms[blockIdx.x] = sm;
		vs[blockIdx.x] = sv;
		ps[blockIdx.x] = sp;
	}
	__syncthreads();
	if (on) {
		m8[i] = __nv_cvt_float_to_fp8(newm / sm, __NV_SATFINITE, __NV_E4M3);
		v8[i] = __nv_cvt_float_to_fp8(newv / sv, __NV_SATFINITE, __NV_E4M3);
		p8[i] = __nv_cvt_float_to_fp8(p / sp, __NV_SATFINITE, __NV_E4M3);
	}
}

static int fp8_adam_ensure(int n)
{
	int ng;

	if (n <= 0)
		return -1;
	if (g_fp8_adam_n == n && g_fp8_master && g_fp8_adam_m && g_fp8_adam_v)
		return 0;
	if (g_fp8_master)
		cudaFree(g_fp8_master);
	if (g_fp8_adam_m)
		cudaFree(g_fp8_adam_m);
	if (g_fp8_adam_v)
		cudaFree(g_fp8_adam_v);
	if (g_fp8_master_scale)
		cudaFree(g_fp8_master_scale);
	if (g_fp8_adam_m_scale)
		cudaFree(g_fp8_adam_m_scale);
	if (g_fp8_adam_v_scale)
		cudaFree(g_fp8_adam_v_scale);
	g_fp8_master = g_fp8_adam_m = g_fp8_adam_v = NULL;
	g_fp8_master_scale = g_fp8_adam_m_scale = g_fp8_adam_v_scale = NULL;
	g_fp8_adam_n = 0;
	g_fp8_mv_live = 0;
	ng = (n + ADAM_FP8_GROUP - 1) / ADAM_FP8_GROUP;
	if (cudaMalloc(&g_fp8_master, (size_t)n) != cudaSuccess
	    || cudaMalloc(&g_fp8_adam_m, (size_t)n) != cudaSuccess
	    || cudaMalloc(&g_fp8_adam_v, (size_t)n) != cudaSuccess
	    || cudaMalloc(&g_fp8_master_scale, (size_t)ng * sizeof(float))
	    != cudaSuccess
	    || cudaMalloc(&g_fp8_adam_m_scale, (size_t)ng * sizeof(float))
	    != cudaSuccess
	    || cudaMalloc(&g_fp8_adam_v_scale, (size_t)ng * sizeof(float))
	    != cudaSuccess)
		return -1;
	g_fp8_adam_n = n;
	return 0;
}
#endif

extern "C" int gpu_adam_step_dev(ann_real *d_p, ann_real *d_g, ann_real *d_m,
	ann_real *d_v, int n, double lr, double beta1, double beta2, double eps,
	int t, double grad_scale)
{
	return gpu_adam_step_dev_bf16(d_p, NULL, d_g, d_m, d_v, n, lr, beta1,
	    beta2, eps, t, grad_scale);
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
#if CUDART_VERSION >= 12000 && defined(ANN_CUBLAS)
	/*
	** Group-128 E4M3 moments underflow (one large v sets the scale, the
	** rest quantize to 0) and the step explodes. Default off. ANN_FP8_ADAM=1
	** opts into that kernel; GEMM weights still come from the FP32 param.
	*/
	{
		static int s_adam = -1;

		if (s_adam < 0) {
			const char *e = getenv("ANN_FP8_ADAM");

			s_adam = (e && e[0] == '1') ? 1 : 0;
		}
		if (s_adam && g_fp8_fwd && d_p_bf16 && fp8_adam_ensure(n) == 0) {
			int live = g_fp8_mv_live;

			adam_kernel_fp8<<<(n + ADAM_FP8_GROUP - 1) / ADAM_FP8_GROUP,
			    ADAM_FP8_GROUP>>>(d_p, (__nv_bfloat16 *)d_p_bf16, d_g,
			    d_m, d_v, g_fp8_master, g_fp8_adam_m, g_fp8_adam_v,
			    g_fp8_master_scale, g_fp8_adam_m_scale,
			    g_fp8_adam_v_scale, n, live, lr, beta1, beta2, eps,
			    bc1, bc2, (ann_real)grad_scale);
			g_fp8_mv_live = 1;
			return cudaGetLastError() == cudaSuccess ? 0 : -1;
		}
	}
#endif
#if CUDART_VERSION >= 11000
	if (d_p_bf16 && g_compute_prec == GPU_PREC_BF16) {
		adam_kernel_bf16<<<(n + 255) / 256, 256>>>(d_p,
		    (__nv_bfloat16 *)d_p_bf16, d_g, d_m, d_v, n, lr, beta1,
		    beta2, eps, bc1, bc2, (ann_real)grad_scale, g_adam_wd);
	} else
#endif
		adam_kernel<<<(n + 255) / 256, 256>>>(d_p, d_g, d_m, d_v, n, lr,
		    beta1, beta2, eps, bc1, bc2, (ann_real)grad_scale, g_adam_wd);
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int gpu_adam_step(ann_real *param, ann_real *grad, ann_real *m, ann_real *v,
	int n, double lr, double beta1, double beta2, double eps, int t,
	double grad_scale)
{
	ann_real *d_p;
	ann_real *d_g;
	ann_real *d_m;
	ann_real *d_v;
	size_t nbytes;

	if (!param || !grad || !m || !v || n <= 0 || lr <= 0.0 || t < 1)
		return -1;
	if (!gpu_available())
		return -1;
	nbytes = (size_t)n * sizeof(ann_real);
	d_p = d_g = d_m = d_v = NULL;
	if (!(d_p = (ann_real *)gpu_malloc(nbytes))
	    || !(d_g = (ann_real *)gpu_malloc(nbytes))
	    || !(d_m = (ann_real *)gpu_malloc(nbytes))
	    || !(d_v = (ann_real *)gpu_malloc(nbytes)))
		goto fail;
	if (gpu_memcpy_h2d(d_p, param, nbytes) != 0
	    || gpu_memcpy_h2d(d_g, grad, nbytes) != 0
	    || gpu_memcpy_h2d(d_m, m, nbytes) != 0
	    || gpu_memcpy_h2d(d_v, v, nbytes) != 0)
		goto fail;
	if (gpu_adam_step_dev(d_p, d_g, d_m, d_v, n, lr, beta1, beta2, eps, t,
	    grad_scale) != 0)
		goto fail;
	if (gpu_memcpy_d2h(param, d_p, nbytes) != 0
	    || gpu_memcpy_d2h(grad, d_g, nbytes) != 0
	    || gpu_memcpy_d2h(m, d_m, nbytes) != 0
	    || gpu_memcpy_d2h(v, d_v, nbytes) != 0)
		goto fail;
	gpu_free(d_p);
	gpu_free(d_g);
	gpu_free(d_m);
	gpu_free(d_v);
	return 0;
fail:
	gpu_free(d_p);
	gpu_free(d_g);
	gpu_free(d_m);
	gpu_free(d_v);
	return -1;
}

extern "C" int gpu_mat_mul_backward(mat *da, mat *db, const mat *dc,
	const mat *a, const mat *b)
{
	/*
	** dA = dC * B^T ; dB = A^T * dC. H2: OP_T GEMMs read B / A in place,
	** so no host transpose copies (and no D2H of device-resident a/b).
	*/
	if (!dc || !a || !b || !gpu_available())
		return -1;
	if (da && gpu_mat_mul_bt(da, dc, b) != 0)
		return -1;
	if (db && gpu_mat_mul_at(db, a, dc) != 0)
		return -1;
	return 0;
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
	add_kernel<<<(n + bs - 1) / bs, bs>>>(dst->d_data, src->d_data, n);
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
	add2_kernel<<<(n + bs - 1) / bs, bs>>>(dst->d_data, a->d_data, b->d_data,
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
	axpy_kernel<<<(n + bs - 1) / bs, bs>>>(y->d_data, x->d_data,
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

extern "C" int gpu_embed_lookup(mat *out, const mat *table, const int *ids, int n)
{
	OpScope _op(OP_EMBED);
	int *d_ids;
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
	ib = (size_t)n * sizeof(int);
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
	embed_lookup_kernel<<<(nd + 255) / 256, 256>>>(out->d_data, table->d_data,
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
	int *d_ids;
	int d;
	int nd;
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
	ib = (size_t)n * sizeof(int);
	d_ids = (int *)gpu_malloc(ib);
	if (!d_ids)
		return -1;
	if (gpu_memcpy_h2d(d_ids, ids, ib) != 0) {
		gpu_free(d_ids);
		return -1;
	}
	nd = n * d;
	embed_scatter_kernel<<<(nd + 255) / 256, 256>>>(dtable->d_data,
	    dout->d_data, d_ids, n, d);
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

extern "C" void gpu_stream_sync_copy(void)
{
	if (g_copy_stream)
		(void)cudaStreamSynchronize(g_copy_stream);
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

__global__ static void col_sum_kernel(ann_real *db, const ann_real *x,
	int rows, int cols)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r;
	double s;

	if (c >= cols)
		return;
	s = 0.0;
	for (r = 0; r < rows; r++)
		s += (double)x[(size_t)r * (size_t)cols + (size_t)c];
	atomicAdd(&db[c], (ann_real)s);
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

/* Device CE sum + pinned host mirror (captured graph D2H updates host). */
static float *g_ce_dev;
static float *g_ce_host_pinned;
static int g_ce_last_rows;

extern "C" int gpu_ce_delta_from_softmax(ann_real *probs, const int *tgt,
	int rows, int V, ann_real *db_host, double *mean_ce)
{
	OpScope _op(OP_CE);
	ew_ref rp;
	ew_ref rt;
	static ann_real *s_db_dev;
	static ann_real *s_db_host;
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
	if (ew_acquire(&rt, 1, tgt, nbt, 1, 0) != 0) {
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
	cudaMemset(g_ce_dev, 0, sizeof(float));
	cudaMemset(s_db_dev, 0, need_db);
	ce_onehot_ce_kernel<<<(rows + EW_BLOCK - 1) / EW_BLOCK, EW_BLOCK>>>(
	    rp.dev, (const int *)rt.dev, rows, V, g_ce_dev);
	if (ew_launch_ok() != 0)
		goto ce_fail;
	{
		dim3 grid((V + EW_BLOCK - 1) / EW_BLOCK,
		    (rows + COLSUM_SLAB - 1) / COLSUM_SLAB);

		col_sum_tiled_kernel<<<grid, EW_BLOCK>>>(s_db_dev, rp.dev, NULL,
		    rows, V);
	}
	if (ew_launch_ok() != 0)
		goto ce_fail;
	/* Always mirror CE to pinned host so graph replay can report NLL. */
	if (cudaMemcpyAsync(g_ce_host_pinned, g_ce_dev, sizeof(float),
	    cudaMemcpyDeviceToHost, 0) != cudaSuccess)
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

extern "C" int gpu_ce_fetch_mean(double *mean_ce, int rows)
{
	if (!mean_ce || rows <= 0 || !g_ce_host_pinned || g_ce_last_rows <= 0)
		return -1;
	if (cudaStreamSynchronize(0) != cudaSuccess)
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
	posenc_packed_kernel<<<(n + EW_BLOCK - 1) / EW_BLOCK, EW_BLOCK>>>(rx.dev,
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
	{
		dim3 grid((cols + EW_BLOCK - 1) / EW_BLOCK,
		    (rows + COLSUM_SLAB - 1) / COLSUM_SLAB);

		col_sum_tiled_kernel<<<grid, EW_BLOCK>>>(acc->dev, rx.dev, NULL,
		    rows, cols);
	}
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
		(void)cudaMemset(acc->dev, 0, (size_t)cols * sizeof(ann_real));
		acc->dirty = 0;
	}
	pthread_mutex_unlock(&g_ew_mu);
	return 0;
}

extern "C" int gpu_graph_begin(void)
{
	if (!gpu_available())
		return -1;
	gpu_graph_destroy();
	/* Drain copy stream so ThreadLocal capture on default stream is clean. */
	if (g_copy_stream)
		(void)cudaStreamSynchronize(g_copy_stream);
	(void)cudaDeviceSynchronize();
	if (cudaStreamBeginCapture(0, cudaStreamCaptureModeThreadLocal)
	    != cudaSuccess)
		return -1;
	return 0;
}

extern "C" int gpu_graph_end(void)
{
	if (!gpu_available())
		return -1;
	if (cudaStreamEndCapture(0, &g_graph) != cudaSuccess)
		return -1;
	if (cudaGraphInstantiate(&g_graph_exec, g_graph, NULL, NULL, 0)
	    != cudaSuccess) {
		cudaGraphDestroy(g_graph);
		g_graph = NULL;
		return -1;
	}
	g_graph_ok = 1;
	return 0;
}

extern "C" int gpu_graph_replay(void)
{
	if (!g_graph_ok || !g_graph_exec)
		return -1;
	if (cudaGraphLaunch(g_graph_exec, 0) != cudaSuccess)
		return -1;
	return launch_done();
}

extern "C" void gpu_graph_destroy(void)
{
	if (g_graph_exec) {
		cudaGraphExecDestroy(g_graph_exec);
		g_graph_exec = NULL;
	}
	if (g_graph) {
		cudaGraphDestroy(g_graph);
		g_graph = NULL;
	}
	g_graph_ok = 0;
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
		    NULL);
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
		    NULL);
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
