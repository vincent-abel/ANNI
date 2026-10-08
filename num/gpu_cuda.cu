/*
**	Optional CUDA mat_mul (build with: make cuda)
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
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

extern "C" int gpu_mat_mul(mat *c, const mat *a, const mat *b)
{
	double *dA;
	double *dB;
	double *dC;
	size_t sa;
	size_t sb;
	size_t sc;
	dim3 block;
	dim3 grid;
	cudaError_t err;

	if (!c || !a || !b || !c->data || !a->data || !b->data)
		return -1;
	if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols)
		return -1;
	if (!gpu_available())
		return -1;
	sa = (size_t)a->rows * (size_t)a->cols * sizeof(double);
	sb = (size_t)b->rows * (size_t)b->cols * sizeof(double);
	sc = (size_t)c->rows * (size_t)c->cols * sizeof(double);
	dA = dB = dC = NULL;
	if (cudaMalloc((void **)&dA, sa) != cudaSuccess
	    || cudaMalloc((void **)&dB, sb) != cudaSuccess
	    || cudaMalloc((void **)&dC, sc) != cudaSuccess)
		goto fail;
	if (cudaMemcpy(dA, a->data, sa, cudaMemcpyHostToDevice) != cudaSuccess
	    || cudaMemcpy(dB, b->data, sb, cudaMemcpyHostToDevice) != cudaSuccess)
		goto fail;
	block = dim3(16, 16);
	grid = dim3((c->cols + 15) / 16, (c->rows + 15) / 16);
	gemm_kernel<<<grid, block>>>(dA, dB, dC, a->rows, b->cols, a->cols);
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		goto fail;
	if (cudaMemcpy(c->data, dC, sc, cudaMemcpyDeviceToHost) != cudaSuccess)
		goto fail;
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	return 0;
fail:
	if (dA)
		cudaFree(dA);
	if (dB)
		cudaFree(dB);
	if (dC)
		cudaFree(dC);
	return -1;
}
