/*
**	Optional CUDA backend (build with: make cuda)
*/
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#ifdef ANN_CUBLAS
#include <cublas_v2.h>
#endif
extern "C" {
#include "gpu.h"
#include "mat.h"
}

static int g_sync_mode;
static cudaStream_t g_copy_stream;
static cudaGraph_t g_graph;
static cudaGraphExec_t g_graph_exec;
static int g_graph_ok;

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
static int g_cublas_ok;

static int cublas_ensure(void)
{
	if (g_cublas_ok)
		return 0;
	if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS)
		return -1;
	/* R10: TF32 tensor cores on Ampere+ when the cuBLAS header knows it. */
#ifdef CUBLAS_TF32_TENSOR_OP_MATH
	(void)cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
#elif defined(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
	(void)cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
#endif
	g_cublas_ok = 1;
	return 0;
}

/* Row-major C = A B via column-major cuBLAS identity. */
static int cublas_gemm_rm(ann_real *dC, const ann_real *dA, const ann_real *dB,
	int M, int N, int K)
{
	const ann_real alpha = 1.0f;
	const ann_real beta = 0.0f;

	if (cublas_ensure() != 0)
		return -1;
	if (cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
		dB, N, dA, K, &beta, dC, N) != CUBLAS_STATUS_SUCCESS)
		return -1;
	return 0;
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

	if (cublas_ensure() != 0)
		return -1;
	if (cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
		dB, K, dA, K, &beta, dC, N) != CUBLAS_STATUS_SUCCESS)
		return -1;
	return 0;
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

	if (cublas_ensure() != 0)
		return -1;
	if (cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, N, M, T, &alpha,
		dB, N, dA, M, &beta, dC, N) != CUBLAS_STATUS_SUCCESS)
		return -1;
	return 0;
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

extern "C" void gpu_free(void *p)
{
	if (p)
		cudaFree(p);
}

extern "C" int gpu_memcpy_h2d(void *dst, const void *src, size_t nbytes)
{
	if (!dst || !src || nbytes == 0)
		return -1;
	return cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice) == cudaSuccess
	    ? 0 : -1;
}

extern "C" int gpu_memcpy_d2h(void *dst, const void *src, size_t nbytes)
{
	if (!dst || !src || nbytes == 0)
		return -1;
	return cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost) == cudaSuccess
	    ? 0 : -1;
}

extern "C" int gpu_memcpy_d2d(void *dst, const void *src, size_t nbytes)
{
	if (!dst || !src || nbytes == 0)
		return -1;
	return cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice) == cudaSuccess
	    ? 0 : -1;
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
		    && m->on_device != MAT_DEV_SYNCED) {
			if (!m->data
			    || gpu_memcpy_h2d(m->d_data, m->data, mat_bytes(m)) != 0)
				return -1;
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

/* Output buffer for a pure-write result: reuse the twin, else temp malloc. */
static int ensure_dev_out(const mat *m, ann_real **out, int *owned)
{
	*owned = 0;
	*out = m->d_data;
	if (*out)
		return 0;
	*out = (ann_real *)gpu_malloc(mat_bytes(m));
	if (!*out)
		return -1;
	*owned = 1;
	return 0;
}

/*
** After a launch: always surface launch errors; only block on the device in
** eager mode. BATCH defers to gpu_device_sync() once per microbatch.
*/
static int launch_done(void)
{
	if (cudaGetLastError() != cudaSuccess)
		return -1;
	if (g_sync_mode == GPU_SYNC_BATCH)
		return 0;
	return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}

extern "C" void gpu_device_sync(void)
{
	cudaError_t err;

	if (!gpu_available())
		return;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		fprintf(stderr, "gpu_device_sync: %s\n", cudaGetErrorString(err));
}

/*
** Finish an output mat: BATCH + pinned leaves the result on device
** (on_device = 1, no PCIe D2H); otherwise D2H into host and mark host-auth.
*/
static int finish_out(mat *m, ann_real *dptr, int owned)
{
	if (g_sync_mode == GPU_SYNC_BATCH && m->d_data && !owned) {
		m->on_device = MAT_DEV_DEVICE;
		return 0;
	}
	if (m->data && gpu_memcpy_d2h(m->data, dptr, mat_bytes(m)) != 0)
		return -1;
	if (m->d_data && !owned)
		m->on_device = MAT_DEV_HOST;
	return 0;
}

extern "C" int gpu_mat_mul(mat *c, const mat *a, const mat *b)
{
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

	if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
		(void)cudaGetLastError();
		return 0;
	}
	return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
}

/* Bind host (or device) memory to a device pointer; slot = scratch id. */
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

__global__ static void ln_fwd_kernel(ann_real *y, const ann_real *x,
	const ann_real *gamma, const ann_real *beta, ann_real *x_hat,
	ann_real *mean_out, ann_real *inv_out, int d, ann_real eps)
{
	__shared__ ann_real sh[EW_BLOCK];
	size_t off = (size_t)blockIdx.x * (size_t)d;
	ann_real mean = 0.0f;
	ann_real var = 0.0f;
	ann_real inv;
	ann_real v;
	ann_real xh;
	int j;

	for (j = threadIdx.x; j < d; j += blockDim.x)
		mean += x[off + j];
	mean = ew_block_sum(mean, sh) / (ann_real)d;
	for (j = threadIdx.x; j < d; j += blockDim.x) {
		v = x[off + j] - mean;
		var += v * v;
	}
	var = ew_block_sum(var, sh) / (ann_real)d;
	inv = 1.0f / sqrtf(var + eps);
	for (j = threadIdx.x; j < d; j += blockDim.x) {
		xh = (x[off + j] - mean) * inv;
		if (x_hat)
			x_hat[off + j] = xh;
		y[off + j] = xh * gamma[j] + beta[j];	/* y may alias x */
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

	for (j = threadIdx.x; j < d; j += blockDim.x) {
		g = dy[off + j];
		xh = x_hat[off + j];
		atomicAdd(&dbeta[j], g);
		atomicAdd(&dgamma[j], g * xh);
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

/* grid = B*H*T blocks (bh, i); dyn smem = (dh + T) floats. */
__global__ static void attn_fwd_kernel(ann_real *concat, ann_real *w,
	const ann_real *q, const ann_real *k, const ann_real *v,
	int T, int H, int dh, ann_real scale, const int *seqlens)
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
	ann_real *wr = w + (size_t)blockIdx.x * T;

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
	for (j = threadIdx.x; j < T; j += blockDim.x) {
		sc[j] /= sum;
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
	    g_att_seqlens_host ? g_att_seqlens_dev : NULL);
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
	if (ew_acquire(&rg, 2, gamma, gb, 1, 0) != 0
	    || ew_acquire(&rb, 3, beta, gb, 1, 0) != 0)
		return -1;
	memset(&rh, 0, sizeof(rh));
	memset(&rm, 0, sizeof(rm));
	memset(&ri, 0, sizeof(ri));
	if (x_hat && ew_acquire(&rh, 4, x_hat, mb, 0, 1) != 0)
		return -1;
	if (mean && ew_acquire(&rm, 5, mean, vb, 0, 1) != 0)
		return -1;
	if (inv_std && ew_acquire(&ri, 6, inv_std, vb, 0, 1) != 0)
		return -1;
	ln_fwd_kernel<<<T, ew_block_for(d)>>>(ry.dev, rx.dev, rg.dev, rb.dev,
	    x_hat ? rh.dev : NULL, mean ? rm.dev : NULL,
	    inv_std ? ri.dev : NULL, d, (ann_real)eps);
	if (ew_launch_ok() != 0)
		return -1;
	/* Download y last: with y == x a late failure must not lose the input. */
	did = 0;
	if (x_hat) {
		if ((rc = ew_release(&rh)) < 0)
			return -1;
		did |= rc;
	}
	if (mean) {
		if ((rc = ew_release(&rm)) < 0)
			return -1;
		did |= rc;
	}
	if (inv_std) {
		if ((rc = ew_release(&ri)) < 0)
			return -1;
		did |= rc;
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
	int rc;

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_layernorm_forward_ex_locked(y, x, gamma, beta, x_hat, mean, inv_std, T, d, eps);
	pthread_mutex_unlock(&g_ew_mu);
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
	ew_ref rdg;
	ew_ref rdb;
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
	    || ew_acquire(&rg, 2, gamma, gb, 1, 0) != 0
	    || ew_acquire(&rh, 3, x_hat, mb, 1, 0) != 0
	    || ew_acquire(&ri, 4, inv_std, vb, 1, 0) != 0
	    || ew_acquire(&rdg, 5, dgamma, gb, 1, 1) != 0
	    || ew_acquire(&rdb, 6, dbeta, gb, 1, 1) != 0)
		return -1;
	ln_bwd_kernel<<<T, ew_block_for(d)>>>(rdx.dev, rdy.dev, rg.dev, rh.dev,
	    ri.dev, rdg.dev, rdb.dev, d);
	if (ew_launch_ok() != 0)
		return -1;
	/* Accumulators first: dx last so a late failure can still fall back. */
	did = 0;
	if ((rc = ew_release(&rdg)) < 0)
		return -1;
	did |= rc;
	if ((rc = ew_release(&rdb)) < 0)
		return -1;
	did |= rc;
	if ((rc = ew_release(&rdx)) < 0)
		return -1;
	did |= rc;
	if (dx->d_data)
		dx->on_device = MAT_DEV_DEVICE;
	return ew_finish_sync(did) == 0 ? 0 : -1;
}

extern "C" int gpu_layernorm_backward(mat *dx, const mat *dy, const ann_real *gamma,
	const ann_real *x_hat, const ann_real *inv_std, int T, int d,
	ann_real *dgamma, ann_real *dbeta)
{
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
	    || ew_acquire(&rb, 1, bias, (size_t)cols * sizeof(ann_real), 1, 0) != 0)
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
	int rc;

	pthread_mutex_lock(&g_ew_mu);
	rc = gpu_relu_backward_locked(dx, x_pre_relu_or_mask, n);
	pthread_mutex_unlock(&g_ew_mu);
	return rc;
}

__global__ static void adam_kernel(ann_real *param, ann_real *grad, ann_real *m,
	ann_real *v, int n, ann_real lr, ann_real beta1, ann_real beta2, ann_real eps,
	ann_real bc1, ann_real bc2, ann_real gscale)
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
	grad[i] = 0.0f;
}

extern "C" int gpu_adam_step_dev(ann_real *d_p, ann_real *d_g, ann_real *d_m,
	ann_real *d_v, int n, double lr, double beta1, double beta2, double eps,
	int t, double grad_scale)
{
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
	adam_kernel<<<(n + 255) / 256, 256>>>(d_p, d_g, d_m, d_v, n, lr, beta1,
	    beta2, eps, bc1, bc2, (ann_real)grad_scale);
	if (cudaDeviceSynchronize() != cudaSuccess)
		return -1;
	return 0;
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

__global__ static void gemv_kernel(const ann_real *x, const ann_real *W, ann_real *y,
	int D)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	ann_real sum;

	if (j >= D)
		return;
	sum = 0.0f;
	for (i = 0; i < D; i++)
		sum += x[i] * W[i * D + j];
	y[j] = sum;
}

__global__ static void kv_append_kernel(ann_real *k_cache, ann_real *v_cache,
	const ann_real *k, const ann_real *v, int row, int D)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j >= D)
		return;
	k_cache[row * D + j] = k[j];
	v_cache[row * D + j] = v[j];
}

__global__ static void kv_scores_hd(const ann_real *q, const ann_real *k_cache,
	ann_real *scores, int len, int d_model, int off, int d, ann_real scale)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int t;
	ann_real sum;

	if (j >= len)
		return;
	sum = 0.0f;
	for (t = 0; t < d; t++)
		sum += q[off + t] * k_cache[j * d_model + off + t];
	scores[j] = sum * scale;
}

__global__ static void kv_av_hd(const ann_real *scores, const ann_real *v_cache,
	ann_real *otmp, int len, int d_model, int off, int d)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	ann_real sum;

	if (t >= d)
		return;
	sum = 0.0f;
	for (j = 0; j < len; j++)
		sum += scores[j] * v_cache[j * d_model + off + t];
	otmp[off + t] = sum;
}

extern "C" int gpu_kv_attn_decode(ann_real *out, const ann_real *x,
	const ann_real *wq, const ann_real *wk, const ann_real *wv,
	const ann_real *wo, ann_real *k_cache, ann_real *v_cache,
	int len, int max_T, int d_model, int n_heads, int d_head)
{
	ann_real *d_x;
	ann_real *d_wq;
	ann_real *d_wk;
	ann_real *d_wv;
	ann_real *d_wo;
	ann_real *d_q;
	ann_real *d_k;
	ann_real *d_v;
	ann_real *d_kc;
	ann_real *d_vc;
	ann_real *d_otmp;
	ann_real *d_scores;
	ann_real *h_scores;
	ann_real *h_tmp;
	size_t db;
	size_t wb;
	size_t cb;
	double scale;
	double mx;
	double sum;
	int h;
	int j;
	int off;
	int threads;

	if (!out || !x || !wq || !wk || !wv || !wo || !k_cache || !v_cache)
		return -1;
	if (!gpu_available() || len <= 0 || len > max_T || d_model <= 0
	    || n_heads <= 0 || d_head <= 0 || n_heads * d_head != d_model)
		return -1;
	db = (size_t)d_model * sizeof(ann_real);
	wb = (size_t)d_model * (size_t)d_model * sizeof(ann_real);
	cb = (size_t)max_T * (size_t)d_model * sizeof(ann_real);
	scale = 1.0 / sqrt((double)d_head);
	threads = 256;
	d_x = d_wq = d_wk = d_wv = d_wo = NULL;
	d_q = d_k = d_v = d_kc = d_vc = d_otmp = d_scores = NULL;
	h_scores = h_tmp = NULL;
	if (!(d_x = (ann_real *)gpu_malloc(db))
	    || !(d_wq = (ann_real *)gpu_malloc(wb))
	    || !(d_wk = (ann_real *)gpu_malloc(wb))
	    || !(d_wv = (ann_real *)gpu_malloc(wb))
	    || !(d_wo = (ann_real *)gpu_malloc(wb))
	    || !(d_q = (ann_real *)gpu_malloc(db))
	    || !(d_k = (ann_real *)gpu_malloc(db))
	    || !(d_v = (ann_real *)gpu_malloc(db))
	    || !(d_kc = (ann_real *)gpu_malloc(cb))
	    || !(d_vc = (ann_real *)gpu_malloc(cb))
	    || !(d_otmp = (ann_real *)gpu_malloc(db))
	    || !(d_scores = (ann_real *)gpu_malloc((size_t)len * sizeof(ann_real)))
	    || !(h_scores = (ann_real *)malloc((size_t)len * sizeof(ann_real)))
	    || !(h_tmp = (ann_real *)malloc((size_t)len * sizeof(ann_real))))
		goto fail;
	if (gpu_memcpy_h2d(d_x, x, db) != 0
	    || gpu_memcpy_h2d(d_wq, wq, wb) != 0
	    || gpu_memcpy_h2d(d_wk, wk, wb) != 0
	    || gpu_memcpy_h2d(d_wv, wv, wb) != 0
	    || gpu_memcpy_h2d(d_wo, wo, wb) != 0
	    || gpu_memcpy_h2d(d_kc, k_cache, cb) != 0
	    || gpu_memcpy_h2d(d_vc, v_cache, cb) != 0)
		goto fail;
	gemv_kernel<<<(d_model + threads - 1) / threads, threads>>>(d_x, d_wq,
	    d_q, d_model);
	gemv_kernel<<<(d_model + threads - 1) / threads, threads>>>(d_x, d_wk,
	    d_k, d_model);
	gemv_kernel<<<(d_model + threads - 1) / threads, threads>>>(d_x, d_wv,
	    d_v, d_model);
	kv_append_kernel<<<(d_model + threads - 1) / threads, threads>>>(d_kc,
	    d_vc, d_k, d_v, len - 1, d_model);
	if (cudaDeviceSynchronize() != cudaSuccess)
		goto fail;
	if (gpu_memset(d_otmp, 0, db) != 0)
		goto fail;
	for (h = 0; h < n_heads; h++) {
		off = h * d_head;
		kv_scores_hd<<<(len + threads - 1) / threads, threads>>>(d_q,
		    d_kc, d_scores, len, d_model, off, d_head, scale);
		if (cudaDeviceSynchronize() != cudaSuccess)
			goto fail;
		if (gpu_memcpy_d2h(h_scores, d_scores,
			(size_t)len * sizeof(ann_real)) != 0)
			goto fail;
		mx = h_scores[0];
		for (j = 1; j < len; j++)
			if (h_scores[j] > mx)
				mx = h_scores[j];
		sum = 0.0f;
		for (j = 0; j < len; j++) {
			h_tmp[j] = (ann_real)exp((double)h_scores[j] - mx);
			sum += h_tmp[j];
		}
		for (j = 0; j < len; j++)
			h_scores[j] = h_tmp[j] / sum;
		if (gpu_memcpy_h2d(d_scores, h_scores,
			(size_t)len * sizeof(ann_real)) != 0)
			goto fail;
		kv_av_hd<<<(d_head + threads - 1) / threads, threads>>>(d_scores,
		    d_vc, d_otmp, len, d_model, off, d_head);
	}
	gemv_kernel<<<(d_model + threads - 1) / threads, threads>>>(d_otmp, d_wo,
	    d_q, d_model);
	if (cudaDeviceSynchronize() != cudaSuccess)
		goto fail;
	if (gpu_memcpy_d2h(out, d_q, db) != 0
	    || gpu_memcpy_d2h(k_cache, d_kc, cb) != 0
	    || gpu_memcpy_d2h(v_cache, d_vc, cb) != 0)
		goto fail;
	gpu_free(d_x);
	gpu_free(d_wq);
	gpu_free(d_wk);
	gpu_free(d_wv);
	gpu_free(d_wo);
	gpu_free(d_q);
	gpu_free(d_k);
	gpu_free(d_v);
	gpu_free(d_kc);
	gpu_free(d_vc);
	gpu_free(d_otmp);
	gpu_free(d_scores);
	free(h_scores);
	free(h_tmp);
	return 0;
fail:
	gpu_free(d_x);
	gpu_free(d_wq);
	gpu_free(d_wk);
	gpu_free(d_wv);
	gpu_free(d_wo);
	gpu_free(d_q);
	gpu_free(d_k);
	gpu_free(d_v);
	gpu_free(d_kc);
	gpu_free(d_vc);
	gpu_free(d_otmp);
	gpu_free(d_scores);
	free(h_scores);
	free(h_tmp);
	return -1;
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

static size_t mat_nbytes(const mat *m)
{
	return (size_t)m->rows * (size_t)m->cols * sizeof(ann_real);
}

static int mat_ensure_dev_auth(mat *m)
{
	if (!m || !m->data)
		return -1;
	if (mat_device_alloc(m) != 0)
		return -1;
	if (m->on_device == MAT_DEV_HOST) {
		if (gpu_memcpy_h2d(m->d_data, m->data, mat_nbytes(m)) != 0)
			return -1;
		m->on_device = MAT_DEV_SYNCED;
	}
	return 0;
}

extern "C" int gpu_mat_copy(mat *dst, const mat *src)
{
	size_t n;

	if (!dst || !src || !dst->data || !src->data)
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
	int n;
	int bs;

	if (!dst || !src || !dst->data || !src->data)
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

extern "C" int gpu_mat_axpy(mat *y, double a, const mat *x)
{
	int n;
	int bs;

	if (!y || !x || !y->data || !x->data)
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

extern "C" int gpu_graph_begin(void)
{
	if (!gpu_available())
		return -1;
	gpu_graph_destroy();
	if (cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal) != cudaSuccess)
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
