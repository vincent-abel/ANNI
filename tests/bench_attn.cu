/*
** Stand-alone harness for the fused attention kernels: random Q, K, V, dO at
** the talk-170m shape (B=16, H=16, T=256, dh=64), timings of the forward
** and backward kernels, and a comparison of a candidate kernel against the
** WMMA reference. Includes num/gpu_cuda.cu to reach its static kernels.
**
**   nvcc -O2 -I. -Inum -Ilang -Icompat -arch=sm_120 --use_fast_math -DANN_CUBLAS \
**        tests/bench_attn.cu -o tests/bench_attn -lcublas -lcublasLt
*/
#include "../num/gpu_cuda.cu"
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
	fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static float frand(std::mt19937 &g) { return std::uniform_real_distribution<float>(-1.f, 1.f)(g); }

static double max_abs_diff(const float *a, const float *b, size_t n, double *max_abs)
{
	double m = 0, ma = 0;
	for (size_t i = 0; i < n; i++) {
		double d = fabs((double)a[i] - (double)b[i]);
		if (d > m) m = d;
		if (fabs((double)a[i]) > ma) ma = fabs((double)a[i]);
	}
	if (max_abs) *max_abs = ma;
	return m;
}

int main(int argc, char **argv)
{
	const int B = 16, H = 16, T = 256, dh = 64, D = H * dh;
	const int iters = argc > 1 ? atoi(argv[1]) : 20;
	const size_t npack = (size_t)B * H * T * dh;	/* head-major [b][h][t][dh] */
	const size_t nrows = (size_t)B * H * T;
	const float scale = 1.0f / sqrtf((float)dh);
	std::mt19937 gen(1);
	std::vector<__nv_bfloat16> hq(npack), hk(npack), hv(npack), hdo(npack);
	std::vector<float> hO(npack), hdOf(npack);

	for (size_t i = 0; i < npack; i++) {
		hq[i] = __float2bfloat16(frand(gen));
		hk[i] = __float2bfloat16(frand(gen));
		hv[i] = __float2bfloat16(frand(gen));
		float g = 0.1f * frand(gen);
		hdo[i] = __float2bfloat16(g);
	}
	__nv_bfloat16 *q, *k, *v, *dO;
	float *O, *O2, *lse, *lse2, *dOf, *dvec, *dq, *dk, *dv, *dq2, *dk2, *dv2;
	CK(cudaMalloc(&q, npack * 2)); CK(cudaMalloc(&k, npack * 2));
	CK(cudaMalloc(&v, npack * 2)); CK(cudaMalloc(&dO, npack * 2));
	CK(cudaMalloc(&O, npack * 4)); CK(cudaMalloc(&O2, npack * 4));
	CK(cudaMalloc(&dOf, npack * 4));
	CK(cudaMalloc(&lse, nrows * 4)); CK(cudaMalloc(&lse2, nrows * 4));
	CK(cudaMalloc(&dvec, nrows * 4));
	CK(cudaMalloc(&dq, npack * 4)); CK(cudaMalloc(&dk, npack * 4)); CK(cudaMalloc(&dv, npack * 4));
	CK(cudaMalloc(&dq2, npack * 4)); CK(cudaMalloc(&dk2, npack * 4)); CK(cudaMalloc(&dv2, npack * 4));
	CK(cudaMemcpy(q, hq.data(), npack * 2, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(k, hk.data(), npack * 2, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(v, hv.data(), npack * 2, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(dO, hdo.data(), npack * 2, cudaMemcpyHostToDevice));
	/*
	** The kernels write O / dq / dk / dv in the model layout [b][t][h*dh]
	** and read dO (FP32, for dvec) in that layout too; build dOf = dO in
	** model layout from the head-major BF16 copy.
	*/
	{
		std::vector<float> tmp(npack);
		for (int b = 0; b < B; b++) for (int h = 0; h < H; h++)
			for (int t = 0; t < T; t++) for (int c = 0; c < dh; c++)
				tmp[((size_t)b * T + t) * D + h * dh + c] =
				    __bfloat162float(hdo[(((size_t)b * H + h) * T + t) * dh + c]);
		CK(cudaMemcpy(dOf, tmp.data(), npack * 4, cudaMemcpyHostToDevice));
	}
	cudaEvent_t e0, e1;
	CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
	float ms;

	/* ---- reference forward (WMMA) ---- */
	size_t smem_f = fa_wmma_smem(dh);
	dim3 gridf(B * H, T / FA_BQ);
	attn_fa_wmma_fwd_kernel<<<gridf, 32 * FA_WARPS, smem_f>>>(O, lse, q, k, v, T, H, dh, scale);
	CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
	CK(cudaEventRecord(e0));
	for (int i = 0; i < iters; i++)
		attn_fa_wmma_fwd_kernel<<<gridf, 32 * FA_WARPS, smem_f>>>(O, lse, q, k, v, T, H, dh, scale);
	CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
	CK(cudaEventElapsedTime(&ms, e0, e1));
	double fwd_flops = 4.0 * B * H * (double)T * T * dh / 2.0;	/* causal */
	printf("wmma fwd : %.3f ms  (%.1f TFLOP/s causal-counted)\n", ms / iters, fwd_flops / (ms / iters * 1e-3) / 1e12);

	/* ---- reference backward (WMMA) ---- */
	size_t smem_b = fa_wmma_bwd_smem(dh);
	dim3 gridb(B * H, T / FA_BK);
	attn_fa_dvec_kernel<<<(unsigned)((nrows + 255) / 256), 256>>>(dvec, dOf, O, (int)nrows, T, H, dh);
	CK(cudaMemset(dq, 0, npack * 4));
	attn_fa_wmma_bwd_kernel<<<gridb, 32 * FA_WARPS, smem_b>>>(dq, dk, dv, q, k, v, dO, lse, dvec, T, H, dh, scale);
	CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
	CK(cudaEventRecord(e0));
	for (int i = 0; i < iters; i++) {
		cudaMemsetAsync(dq, 0, npack * 4, 0);
		attn_fa_wmma_bwd_kernel<<<gridb, 32 * FA_WARPS, smem_b>>>(dq, dk, dv, q, k, v, dO, lse, dvec, T, H, dh, scale);
	}
	CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
	CK(cudaEventElapsedTime(&ms, e0, e1));
	printf("wmma bwd : %.3f ms  (%.1f TFLOP/s causal-counted, 2.5x fwd flops)\n", ms / iters, 2.5 * fwd_flops / (ms / iters * 1e-3) / 1e12);

#ifdef ATTN_FA2
	/* ---- candidate forward / backward: same signatures ---- */
	{
		CK(cudaMemset(O2, 0, npack * 4));
		fa2_fwd_launch(O2, lse2, q, k, v, B, T, H, dh, scale);
		CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
		std::vector<float> a(npack), b(npack), la(nrows), lb(nrows);
		CK(cudaMemcpy(a.data(), O, npack * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(b.data(), O2, npack * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(la.data(), lse, nrows * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(lb.data(), lse2, nrows * 4, cudaMemcpyDeviceToHost));
		double ma, mb;
		double d = max_abs_diff(a.data(), b.data(), npack, &ma);
		max_abs_diff(b.data(), a.data(), npack, &mb);
		printf("fa2 fwd  : max|O-O2| %.3e (max|O| %.3e, max|O2| %.3e), max|dlse| %.3e\n",
		    d, ma, mb, max_abs_diff(la.data(), lb.data(), nrows, NULL));
		printf("           O[0..3] %.5f %.5f %.5f %.5f | O2 %.5f %.5f %.5f %.5f | O[%zu] %.5f O2 %.5f\n",
		    a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], npack - 1, a[npack - 1], b[npack - 1]);
		CK(cudaEventRecord(e0));
		for (int i = 0; i < iters; i++)
			fa2_fwd_launch(O2, lse2, q, k, v, B, T, H, dh, scale);
		CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
		CK(cudaEventElapsedTime(&ms, e0, e1));
		printf("fa2 fwd  : %.3f ms  (%.1f TFLOP/s)\n", ms / iters, fwd_flops / (ms / iters * 1e-3) / 1e12);
	}
#ifdef ATTN_FA2_BWD
	{
		CK(cudaMemset(dq2, 0, npack * 4));
		fa2_bwd_launch(dq2, dk2, dv2, q, k, v, dO, lse, dvec, B, T, H, dh, scale);
		CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
		std::vector<float> a(npack), b(npack);
		double ma, d;
		CK(cudaMemcpy(a.data(), dq, npack * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(b.data(), dq2, npack * 4, cudaMemcpyDeviceToHost));
		d = max_abs_diff(a.data(), b.data(), npack, &ma); printf("fa2 bwd  : dq max diff %.3e (max %.3e)\n", d, ma);
		CK(cudaMemcpy(a.data(), dk, npack * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(b.data(), dk2, npack * 4, cudaMemcpyDeviceToHost));
		d = max_abs_diff(a.data(), b.data(), npack, &ma); printf("fa2 bwd  : dk max diff %.3e (max %.3e)\n", d, ma);
		CK(cudaMemcpy(a.data(), dv, npack * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(b.data(), dv2, npack * 4, cudaMemcpyDeviceToHost));
		d = max_abs_diff(a.data(), b.data(), npack, &ma); printf("fa2 bwd  : dv max diff %.3e (max %.3e)\n", d, ma);
		CK(cudaEventRecord(e0));
		for (int i = 0; i < iters; i++) {
			cudaMemsetAsync(dq2, 0, npack * 4, 0);
			fa2_bwd_launch(dq2, dk2, dv2, q, k, v, dO, lse, dvec, B, T, H, dh, scale);
		}
		CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
		CK(cudaEventElapsedTime(&ms, e0, e1));
		printf("fa2 bwd  : %.3f ms  (%.1f TFLOP/s)\n", ms / iters, 2.5 * fwd_flops / (ms / iters * 1e-3) / 1e12);
	}
#endif
#endif
	return 0;
}
