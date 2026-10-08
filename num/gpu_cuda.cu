/*
**	Optional CUDA backend (build with: make cuda)
*/
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
#include "gpu.h"
#include "mat.h"
}

__global__ static void gemm_kernel(const double *A, const double *B, double *C,
	int M, int N, int K)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	double sum;

	if (row >= M || col >= N)
		return;
	sum = 0.0;
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

extern "C" const char *gpu_backend_name(void)
{
	return "cuda";
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
	return (size_t)m->rows * (size_t)m->cols * sizeof(double);
}

extern "C" int gpu_mat_mul(mat *c, const mat *a, const mat *b)
{
	double *dA;
	double *dB;
	double *dC;
	int ownA;
	int ownB;
	int ownC;
	dim3 block;
	dim3 grid;
	cudaError_t err;

	if (!c || !a || !b)
		return -1;
	if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols)
		return -1;
	if (!gpu_available())
		return -1;

	ownA = ownB = ownC = 0;
	dA = a->d_data;
	dB = b->d_data;
	dC = c->d_data;
	if (!dA) {
		if (!a->data)
			return -1;
		dA = (double *)gpu_malloc(mat_bytes(a));
		if (!dA || gpu_memcpy_h2d(dA, a->data, mat_bytes(a)) != 0) {
			gpu_free(dA);
			return -1;
		}
		ownA = 1;
	}
	if (!dB) {
		if (!b->data)
			return -1;
		dB = (double *)gpu_malloc(mat_bytes(b));
		if (!dB || gpu_memcpy_h2d(dB, b->data, mat_bytes(b)) != 0) {
			if (ownA)
				gpu_free(dA);
			gpu_free(dB);
			return -1;
		}
		ownB = 1;
	}
	if (!dC) {
		dC = (double *)gpu_malloc(mat_bytes(c));
		if (!dC) {
			if (ownA)
				gpu_free(dA);
			if (ownB)
				gpu_free(dB);
			return -1;
		}
		ownC = 1;
	}
	block = dim3(16, 16);
	grid = dim3((c->cols + 15) / 16, (c->rows + 15) / 16);
	gemm_kernel<<<grid, block>>>(dA, dB, dC, a->rows, b->cols, a->cols);
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		goto fail;
	if (c->data && gpu_memcpy_d2h(c->data, dC, mat_bytes(c)) != 0)
		goto fail;
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownC)
		gpu_free(dC);
	else if (c->d_data)
		c->on_device = 1;
	return 0;
fail:
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownC)
		gpu_free(dC);
	return -1;
}

__global__ static void gemm_bt_kernel(const double *A, const double *B, double *C,
	int M, int N, int K)
{
	/* C[m,n] = sum_k A[m,k] * B[n,k] */
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	double sum;

	if (row >= M || col >= N)
		return;
	sum = 0.0;
	for (k = 0; k < K; k++)
		sum += A[row * K + k] * B[col * K + k];
	C[row * N + col] = sum;
}

__global__ static void outer_add_kernel(double *DW, const double *A, const double *B,
	int T, int M, int N)
{
	/* DW[m,n] += sum_t A[t,m] * B[t,n] */
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int t;
	double sum;

	if (m >= M || n >= N)
		return;
	sum = 0.0;
	for (t = 0; t < T; t++)
		sum += A[t * M + m] * B[t * N + n];
	DW[m * N + n] += sum;
}

static int ensure_dev(const mat *m, double **out, int *owned)
{
	*owned = 0;
	*out = m->d_data;
	if (*out)
		return 0;
	if (!m->data)
		return -1;
	*out = (double *)gpu_malloc(mat_bytes(m));
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

extern "C" int gpu_mat_mul_bt(mat *c, const mat *a, const mat *b)
{
	double *dA;
	double *dB;
	double *dC;
	int ownA;
	int ownB;
	int ownC;
	dim3 block;
	dim3 grid;

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
	ownC = 0;
	dC = c->d_data;
	if (!dC) {
		dC = (double *)gpu_malloc(mat_bytes(c));
		if (!dC) {
			if (ownA)
				gpu_free(dA);
			if (ownB)
				gpu_free(dB);
			return -1;
		}
		ownC = 1;
	}
	block = dim3(16, 16);
	grid = dim3((c->cols + 15) / 16, (c->rows + 15) / 16);
	gemm_bt_kernel<<<grid, block>>>(dA, dB, dC, a->rows, b->rows, a->cols);
	if (cudaDeviceSynchronize() != cudaSuccess)
		goto fail;
	if (c->data && gpu_memcpy_d2h(c->data, dC, mat_bytes(c)) != 0)
		goto fail;
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownC)
		gpu_free(dC);
	return 0;
fail:
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownC)
		gpu_free(dC);
	return -1;
}

extern "C" int gpu_mat_outer_add(mat *dw, const mat *a, const mat *b)
{
	double *dW;
	double *dA;
	double *dB;
	int ownW;
	int ownA;
	int ownB;
	dim3 block;
	dim3 grid;

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
	ownW = 0;
	dW = dw->d_data;
	if (!dW) {
		dW = (double *)gpu_malloc(mat_bytes(dw));
		if (!dW || (dw->data && gpu_memcpy_h2d(dW, dw->data, mat_bytes(dw)) != 0)) {
			gpu_free(dW);
			if (ownA)
				gpu_free(dA);
			if (ownB)
				gpu_free(dB);
			return -1;
		}
		ownW = 1;
	}
	block = dim3(16, 16);
	grid = dim3((dw->cols + 15) / 16, (dw->rows + 15) / 16);
	outer_add_kernel<<<grid, block>>>(dW, dA, dB, a->rows, a->cols, b->cols);
	if (cudaDeviceSynchronize() != cudaSuccess)
		goto fail;
	if (dw->data && gpu_memcpy_d2h(dw->data, dW, mat_bytes(dw)) != 0)
		goto fail;
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownW)
		gpu_free(dW);
	return 0;
fail:
	if (ownA)
		gpu_free(dA);
	if (ownB)
		gpu_free(dB);
	if (ownW)
		gpu_free(dW);
	return -1;
}

__global__ static void ln_bwd_kernel(double *dx, const double *dy,
	const double *gamma, const double *x_hat, const double *inv_std,
	double *dgamma, double *dbeta, int T, int d)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	double inv;
	double sum_dy;
	double sum_dy_xhat;
	double dxhat;
	double term;

	if (t >= T)
		return;
	inv = inv_std[t];
	sum_dy = 0.0;
	sum_dy_xhat = 0.0;
	for (j = 0; j < d; j++) {
		atomicAdd(&dbeta[j], dy[t * d + j]);
		atomicAdd(&dgamma[j], dy[t * d + j] * x_hat[t * d + j]);
		dxhat = dy[t * d + j] * gamma[j];
		sum_dy += dxhat;
		sum_dy_xhat += dxhat * x_hat[t * d + j];
	}
	for (j = 0; j < d; j++) {
		dxhat = dy[t * d + j] * gamma[j];
		term = (dxhat - (sum_dy + x_hat[t * d + j] * sum_dy_xhat) / (double)d)
		    * inv;
		dx[t * d + j] = term;
	}
}

extern "C" int gpu_layernorm_backward(mat *dx, const mat *dy, const double *gamma,
	const double *x_hat, const double *inv_std, int T, int d,
	double *dgamma, double *dbeta)
{
	double *d_dx;
	double *d_dy;
	double *d_gamma;
	double *d_xhat;
	double *d_inv;
	double *d_dg;
	double *d_db;
	size_t matb;
	size_t vb;
	size_t gb;

	if (!dx || !dy || !gamma || !x_hat || !inv_std || !dgamma || !dbeta)
		return -1;
	if (!gpu_available() || T <= 0 || d <= 0)
		return -1;
	matb = (size_t)T * (size_t)d * sizeof(double);
	vb = (size_t)T * sizeof(double);
	gb = (size_t)d * sizeof(double);
	d_dx = d_dy = d_gamma = d_xhat = d_inv = d_dg = d_db = NULL;
	if (!(d_dx = (double *)gpu_malloc(matb))
	    || !(d_dy = (double *)gpu_malloc(matb))
	    || !(d_gamma = (double *)gpu_malloc(gb))
	    || !(d_xhat = (double *)gpu_malloc(matb))
	    || !(d_inv = (double *)gpu_malloc(vb))
	    || !(d_dg = (double *)gpu_malloc(gb))
	    || !(d_db = (double *)gpu_malloc(gb)))
		goto fail;
	if (gpu_memcpy_h2d(d_dy, dy->data, matb) != 0
	    || gpu_memcpy_h2d(d_gamma, gamma, gb) != 0
	    || gpu_memcpy_h2d(d_xhat, x_hat, matb) != 0
	    || gpu_memcpy_h2d(d_inv, inv_std, vb) != 0
	    || gpu_memcpy_h2d(d_dg, dgamma, gb) != 0
	    || gpu_memcpy_h2d(d_db, dbeta, gb) != 0)
		goto fail;
	ln_bwd_kernel<<<(T + 255) / 256, 256>>>(d_dx, d_dy, d_gamma, d_xhat,
	    d_inv, d_dg, d_db, T, d);
	if (cudaDeviceSynchronize() != cudaSuccess)
		goto fail;
	if (gpu_memcpy_d2h(dx->data, d_dx, matb) != 0
	    || gpu_memcpy_d2h(dgamma, d_dg, gb) != 0
	    || gpu_memcpy_d2h(dbeta, d_db, gb) != 0)
		goto fail;
	gpu_free(d_dx);
	gpu_free(d_dy);
	gpu_free(d_gamma);
	gpu_free(d_xhat);
	gpu_free(d_inv);
	gpu_free(d_dg);
	gpu_free(d_db);
	return 0;
fail:
	gpu_free(d_dx);
	gpu_free(d_dy);
	gpu_free(d_gamma);
	gpu_free(d_xhat);
	gpu_free(d_inv);
	gpu_free(d_dg);
	gpu_free(d_db);
	return -1;
}

__global__ static void adam_kernel(double *param, double *grad, double *m,
	double *v, int n, double lr, double beta1, double beta2, double eps,
	double bc1, double bc2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double g;
	double mhat;
	double vhat;

	if (i >= n)
		return;
	g = grad[i];
	m[i] = beta1 * m[i] + (1.0 - beta1) * g;
	v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
	mhat = m[i] / bc1;
	vhat = v[i] / bc2;
	param[i] -= lr * mhat / (sqrt(vhat) + eps);
	grad[i] = 0.0;
}

extern "C" int gpu_adam_step(double *param, double *grad, double *m, double *v,
	int n, double lr, double beta1, double beta2, double eps, int t)
{
	double *d_p;
	double *d_g;
	double *d_m;
	double *d_v;
	size_t nbytes;
	double bc1;
	double bc2;

	if (!param || !grad || !m || !v || n <= 0 || lr <= 0.0 || t < 1)
		return -1;
	if (!gpu_available())
		return -1;
	nbytes = (size_t)n * sizeof(double);
	d_p = d_g = d_m = d_v = NULL;
	bc1 = 1.0 - pow(beta1, (double)t);
	bc2 = 1.0 - pow(beta2, (double)t);
	if (bc1 <= 0.0 || bc2 <= 0.0)
		return -1;
	if (!(d_p = (double *)gpu_malloc(nbytes))
	    || !(d_g = (double *)gpu_malloc(nbytes))
	    || !(d_m = (double *)gpu_malloc(nbytes))
	    || !(d_v = (double *)gpu_malloc(nbytes)))
		goto fail;
	if (gpu_memcpy_h2d(d_p, param, nbytes) != 0
	    || gpu_memcpy_h2d(d_g, grad, nbytes) != 0
	    || gpu_memcpy_h2d(d_m, m, nbytes) != 0
	    || gpu_memcpy_h2d(d_v, v, nbytes) != 0)
		goto fail;
	adam_kernel<<<(n + 255) / 256, 256>>>(d_p, d_g, d_m, d_v, n, lr, beta1,
	    beta2, eps, bc1, bc2);
	if (cudaDeviceSynchronize() != cudaSuccess)
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
	/* dA = dC * B^T ; dB = A^T * dC */
	mat *bt;
	mat *at;
	int i;
	int j;
	int rc;

	if (!dc || !a || !b || !gpu_available())
		return -1;
	rc = 0;
	if (da) {
		/* Build B^T on host then gemm */
		bt = mat_create(b->cols, b->rows);
		if (!bt)
			return -1;
		for (i = 0; i < b->rows; i++)
			for (j = 0; j < b->cols; j++)
				bt->data[j * bt->cols + i] = b->data[i * b->cols + j];
		if (gpu_mat_mul(da, dc, bt) != 0)
			rc = -1;
		mat_free(&bt);
		if (rc != 0)
			return -1;
	}
	if (db) {
		at = mat_create(a->cols, a->rows);
		if (!at)
			return -1;
		for (i = 0; i < a->rows; i++)
			for (j = 0; j < a->cols; j++)
				at->data[j * at->cols + i] = a->data[i * a->cols + j];
		if (gpu_mat_mul(db, at, dc) != 0)
			rc = -1;
		mat_free(&at);
	}
	return rc;
}
