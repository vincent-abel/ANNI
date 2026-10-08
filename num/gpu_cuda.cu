/*
**	Optional CUDA backend (build with: make cuda)
*/
#include <cuda_runtime.h>
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
