/*
** Stand-alone harness for the LayerNorm kernels: old (one row per block +
** col_sum_tiled) against the warp-per-row ANN_LN_FAST kernels at the
** talk-170m shape (4096 rows, D = 1024). Prints timings and max errors.
**
**   nvcc -O2 -I. -Inum -Ilang -Icompat -arch=sm_120 --use_fast_math -DANN_CUBLAS \
**        tests/bench_ln.cu -o tests/bench_ln -lcublas -lcublasLt
*/
#include "../num/gpu_cuda.cu"
#include <vector>
#include <random>

/* The harness never calls the mat paths; satisfy the linker. */
extern "C" int mat_device_alloc(mat *) { return -1; }
extern "C" void mat_try_pin(mat *) {}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
	fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static double maxdiff(const std::vector<float> &a, const std::vector<float> &b, double *mx)
{
	double m = 0, ma = 0;
	for (size_t i = 0; i < a.size(); i++) {
		m = fmax(m, fabs((double)a[i] - b[i]));
		ma = fmax(ma, fabs((double)a[i]));
	}
	*mx = ma;
	return m;
}

/* Minimum over 10 batches: the GPU may be shared with a training run. */
template <class F> static float timeit(F f, int iters)
{
	cudaEvent_t e0, e1;
	float ms, best = 1e30f;
	CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
	f(); CK(cudaDeviceSynchronize());
	for (int k = 0; k < 10; k++) {
		CK(cudaEventRecord(e0));
		for (int i = 0; i < iters / 10; i++) f();
		CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
		CK(cudaEventElapsedTime(&ms, e0, e1));
		if (ms * 10 / iters < best) best = ms * 10 / iters;
	}
	return best;
}

/*
** Per-call timing, minimum over many trials (the GPU may be time-sliced with
** a training run, which inflates averages). cold: 128 MB flush before each
** trial, outside the timed window, so the operands come from DRAM.
*/
static char *g_flush;
template <class F> static float timeit_min(F f, int trials, int cold)
{
	cudaEvent_t e0, e1;
	float ms, best = 1e30f;
	if (!g_flush) CK(cudaMalloc(&g_flush, 128u << 20));
	CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
	f(); CK(cudaDeviceSynchronize());
	for (int k = 0; k < trials; k++) {
		if (cold) CK(cudaMemsetAsync(g_flush, k, 128u << 20));
		CK(cudaEventRecord(e0));
		f();
		CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
		CK(cudaEventElapsedTime(&ms, e0, e1));
		if (ms < best) best = ms;
	}
	return best;
}

int main(int argc, char **argv)
{
	const int T = 4096, d = argc > 2 ? atoi(argv[2]) : 1024;
	const int iters = argc > 1 ? atoi(argv[1]) : 50;
	const size_t n = (size_t)T * d;
	std::mt19937 gen(1);
	std::uniform_real_distribution<float> u(-1.f, 1.f);
	std::vector<float> hx(n), hdy(n), hg(d), hb(d), hdx0(n);
	for (auto &v : hx) v = 2.f * u(gen) + 0.3f;
	for (auto &v : hdy) v = 0.1f * u(gen);
	for (auto &v : hdx0) v = 0.05f * u(gen);
	for (auto &v : hg) v = 1.f + 0.2f * u(gen);
	for (auto &v : hb) v = 0.2f * u(gen);
	float *x, *dy, *g, *b, *y1, *y2, *xh1, *xh2, *m1, *m2, *i1, *i2;
	float *dx1, *dx2, *dg1, *db1, *dg2, *db2;
	__nv_bfloat16 *y16a, *y16b, *dx16a, *dx16b;
	CK(cudaMalloc(&x, n * 4)); CK(cudaMalloc(&dy, n * 4));
	CK(cudaMalloc(&g, d * 4)); CK(cudaMalloc(&b, d * 4));
	CK(cudaMalloc(&y1, n * 4)); CK(cudaMalloc(&y2, n * 4));
	CK(cudaMalloc(&xh1, n * 4)); CK(cudaMalloc(&xh2, n * 4));
	CK(cudaMalloc(&m1, T * 4)); CK(cudaMalloc(&m2, T * 4));
	CK(cudaMalloc(&i1, T * 4)); CK(cudaMalloc(&i2, T * 4));
	CK(cudaMalloc(&dx1, n * 4)); CK(cudaMalloc(&dx2, n * 4));
	CK(cudaMalloc(&dg1, d * 4)); CK(cudaMalloc(&db1, d * 4));
	CK(cudaMalloc(&dg2, d * 4)); CK(cudaMalloc(&db2, d * 4));
	CK(cudaMalloc(&y16a, n * 2)); CK(cudaMalloc(&y16b, n * 2));
	CK(cudaMalloc(&dx16a, n * 2)); CK(cudaMalloc(&dx16b, n * 2));
	CK(cudaMemcpy(x, hx.data(), n * 4, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(dy, hdy.data(), n * 4, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(g, hg.data(), d * 4, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(b, hb.data(), d * 4, cudaMemcpyHostToDevice));

	auto fwd_old = [&]() { ln_fwd_kernel<<<T, ew_block_for(d), d * 4>>>(y1, x, g, b, xh1, m1, i1,
	    d, 1e-5f, NULL, NULL, NULL, y16a); };
	auto fwd_new = [&]() { ln_fwd_fast_launch(y2, x, g, b, xh2, m2, i2, T, d, 1e-5f, y16b); };
	fwd_old(); fwd_new(); CK(cudaDeviceSynchronize());
	auto cp = [&](float *p, size_t k) { std::vector<float> h(k); CK(cudaMemcpy(h.data(), p, k * 4, cudaMemcpyDeviceToHost)); return h; };
	auto cp16 = [&](__nv_bfloat16 *p, size_t k) { std::vector<__nv_bfloat16> h(k); CK(cudaMemcpy(h.data(), p, k * 2, cudaMemcpyDeviceToHost));
		std::vector<float> f(k); for (size_t i = 0; i < k; i++) f[i] = __bfloat162float(h[i]); return f; };
	double mx;
	printf("fwd  xhat maxdiff %.3g  inv %.3g  mean %.3g  y16 %.3g (xhat max %.3g)\n",
	    maxdiff(cp(xh1, n), cp(xh2, n), &mx), maxdiff(cp(i1, T), cp(i2, T), &mx),
	    maxdiff(cp(m1, T), cp(m2, T), &mx), maxdiff(cp16(y16a, n), cp16(y16b, n), &mx), 0.0);
	/* FP32 y path too */
	{
		ln_fwd_kernel<<<T, ew_block_for(d), d * 4>>>(y1, x, g, b, NULL, NULL, NULL, d, 1e-5f, NULL, NULL, NULL, NULL);
		ln_fwd_fast_launch(y2, x, g, b, NULL, NULL, NULL, T, d, 1e-5f, NULL);
		CK(cudaDeviceSynchronize());
		printf("fwd  y fp32 maxdiff %.3g\n", maxdiff(cp(y1, n), cp(y2, n), &mx));
	}
	printf("fwd  old %.4f ms   new %.4f ms\n", timeit(fwd_old, iters), timeit(fwd_new, iters));
	printf("fwd  cold old %.4f ms   new %.4f ms\n", timeit_min(fwd_old, 300, 1), timeit_min(fwd_new, 300, 1));

	/* backward (accumulate mode, as in training) */
	dim3 grid((d + EW_BLOCK - 1) / EW_BLOCK, (T + COLSUM_SLAB - 1) / COLSUM_SLAB);
	auto bwd_old = [&]() {
		ln_bwd_kernel<<<T, ew_block_for(d)>>>(dx1, dy, g, xh1, i1, NULL, NULL, d, 1, dx16a);
		col_sum_tiled_kernel<<<grid, EW_BLOCK>>>(db1, dy, NULL, T, d);
		col_sum_tiled_kernel<<<grid, EW_BLOCK>>>(dg1, dy, xh1, T, d);
	};
	auto bwd_new = [&]() { ln_bwd_fast_launch(dx2, dy, g, xh1, i1, dg2, db2, T, d, 1, dx16b); };
	CK(cudaMemset(dg1, 0, d * 4)); CK(cudaMemset(db1, 0, d * 4));
	CK(cudaMemset(dg2, 0, d * 4)); CK(cudaMemset(db2, 0, d * 4));
	CK(cudaMemcpy(dx1, hdx0.data(), n * 4, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(dx2, hdx0.data(), n * 4, cudaMemcpyHostToDevice));
	bwd_old(); bwd_new(); CK(cudaDeviceSynchronize());
	printf("bwd  dx %.3g  dx16 %.3g  dgamma %.3g  dbeta %.3g\n",
	    maxdiff(cp(dx1, n), cp(dx2, n), &mx), maxdiff(cp16(dx16a, n), cp16(dx16b, n), &mx),
	    maxdiff(cp(dg1, d), cp(dg2, d), &mx), maxdiff(cp(db1, d), cp(db2, d), &mx));
	printf("     (dgamma max %.3g)\n", ({ auto v = cp(dg1, d); double m = 0; for (float f : v) m = fmax(m, fabs(f)); m; }));
	printf("bwd  old %.4f ms   new %.4f ms\n", timeit(bwd_old, iters), timeit(bwd_new, iters));
	printf("bwd  cold old %.4f ms   new %.4f ms\n", timeit_min(bwd_old, 300, 1), timeit_min(bwd_new, 300, 1));
	return 0;
}
