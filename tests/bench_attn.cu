/*
** Stand-alone harness for the fused FA2 attention kernels: random Q, K, V,
** dO at the talk-170m shape (B=16, H=16, T=256, dh=64), timings of the
** forward and backward kernels, and a comparison against the TC FP32 path
** (gpu_attn_tc_fwd / gpu_attn_tc_bwd with FP32 GEMMs and an explicit score
** buffer, the reference of make test-gpu). Inputs are BF16-rounded so both
** sides see the same numbers. Includes num/gpu_cuda.cu to reach its static
** kernels. argv: [iters [B]]; B=4 keeps the VRAM under 0.6 GB.
**
**   nvcc -O2 -I. -Inum -Ilang -Icompat -arch=sm_120 --use_fast_math -DANN_CUBLAS \
**        tests/bench_attn.cu <objects/*.o without gpu_cuda/gpu_stub/main/tower/chat_*> \
**        -o tests/bench_attn -lcublas -lcublasLt -lgomp
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

/* Host mat of rows x cols filled from v, pushed to the device. */
static mat *mat_from(const std::vector<float> &v, int rows, int cols)
{
	mat *m = mat_create(rows, cols);

	if (!m) { fprintf(stderr, "mat_create\n"); exit(1); }
	memcpy(m->data, v.data(), (size_t)rows * cols * sizeof(float));
	mat_try_pin(m);
	if (mat_sync_device(m) != 0) { fprintf(stderr, "mat_sync_device\n"); exit(1); }
	return m;
}

static std::vector<float> mat_dl(mat *m)
{
	size_t n = (size_t)m->rows * m->cols;
	std::vector<float> v(n);

	CK(cudaMemcpy(v.data(), m->d_data, n * 4, cudaMemcpyDeviceToHost));
	return v;
}

int main(int argc, char **argv)
{
	const int H = 16, T = 256, dh = 64, D = H * dh;
	const int iters = argc > 1 ? atoi(argv[1]) : 20;
	const int B = argc > 2 ? atoi(argv[2]) : 16;
	const size_t nel = (size_t)B * T * D;		/* model layout [b][t][h*dh] */
	const size_t nrows = (size_t)B * H * T;
	const float scale = 1.0f / sqrtf((float)dh);
	std::mt19937 gen(1);
	std::vector<float> fq(nel), fk(nel), fv(nel), fdo(nel);
	std::vector<__nv_bfloat16> qkv((size_t)B * T * 3 * D), bdo(nel);

	/* BF16-rounded values in FP32 (TC reference) and as BF16 [B*T][3D] rows (FA2). */
	for (size_t i = 0; i < nel; i++) {
		size_t r = i / D, c = i % D;
		__nv_bfloat16 q = __float2bfloat16(frand(gen)), k = __float2bfloat16(frand(gen));
		__nv_bfloat16 v = __float2bfloat16(frand(gen)), g = __float2bfloat16(0.1f * frand(gen));

		fq[i] = __bfloat162float(q); fk[i] = __bfloat162float(k);
		fv[i] = __bfloat162float(v); fdo[i] = __bfloat162float(g);
		qkv[r * 3 * D + c] = q; qkv[r * 3 * D + D + c] = k; qkv[r * 3 * D + 2 * D + c] = v;
		bdo[i] = g;
	}
	gpu_set_compute_prec(GPU_PREC_FP32);
	if (!gpu_available()) { fprintf(stderr, "no GPU\n"); return 1; }

	cudaEvent_t e0, e1;
	CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
	float ms;
	double fwd_flops = 4.0 * B * H * (double)T * T * dh / 2.0;	/* causal */

	/* ---- reference: TC strided GemmEx path, FP32 ---- */
	mat *mq = mat_from(fq, B * T, D), *mk = mat_from(fk, B * T, D), *mv = mat_from(fv, B * T, D);
	mat *mdo = mat_from(fdo, B * T, D);
	mat *mw = mat_create_dev(B * H * T, T), *mc = mat_create_dev(B * T, D);
	mat *mdq = mat_create_dev(B * T, D), *mdk = mat_create_dev(B * T, D), *mdv = mat_create_dev(B * T, D);
	if (!mw || !mw->d_data || !mc || !mdq || !mdk || !mdv) { fprintf(stderr, "alloc\n"); return 1; }
	if (gpu_attn_tc_fwd(mc, mw, mq, mk, mv, B, T, H, dh) != 0) { fprintf(stderr, "tc fwd failed\n"); return 1; }
	CK(cudaDeviceSynchronize());
	CK(cudaEventRecord(e0));
	for (int i = 0; i < iters; i++)
		gpu_attn_tc_fwd(mc, mw, mq, mk, mv, B, T, H, dh);
	CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
	CK(cudaEventElapsedTime(&ms, e0, e1));
	printf("tc fwd   : %.3f ms  (%.1f TFLOP/s causal-counted)\n", ms / iters, fwd_flops / (ms / iters * 1e-3) / 1e12);
	if (gpu_attn_tc_bwd(mdq, mdk, mdv, mdo, mw, mq, mk, mv, B, T, H, dh) != 0) { fprintf(stderr, "tc bwd failed\n"); return 1; }
	CK(cudaDeviceSynchronize());
	CK(cudaEventRecord(e0));
	for (int i = 0; i < iters; i++)
		gpu_attn_tc_bwd(mdq, mdk, mdv, mdo, mw, mq, mk, mv, B, T, H, dh);
	CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
	CK(cudaEventElapsedTime(&ms, e0, e1));
	printf("tc bwd   : %.3f ms  (%.1f TFLOP/s causal-counted, 2.5x fwd flops)\n", ms / iters, 2.5 * fwd_flops / (ms / iters * 1e-3) / 1e12);
	std::vector<float> rO = mat_dl(mc), rdq = mat_dl(mdq), rdk = mat_dl(mdk), rdv = mat_dl(mdv);

	/* ---- FA2 on the BF16 QKV rows (the production layout) ---- */
	__nv_bfloat16 *dqkv, *dO;
	float *O2, *lse2, *dOf, *dvec, *dq2, *dk2, *dv2;
	CK(cudaMalloc(&dqkv, qkv.size() * 2)); CK(cudaMalloc(&dO, nel * 2));
	CK(cudaMalloc(&O2, nel * 4)); CK(cudaMalloc(&lse2, nrows * 4)); CK(cudaMalloc(&dOf, nel * 4));
	CK(cudaMalloc(&dvec, nrows * 4));
	CK(cudaMalloc(&dq2, nel * 4)); CK(cudaMalloc(&dk2, nel * 4)); CK(cudaMalloc(&dv2, nel * 4));
	CK(cudaMemcpy(dqkv, qkv.data(), qkv.size() * 2, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(dO, bdo.data(), nel * 2, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(dOf, fdo.data(), nel * 4, cudaMemcpyHostToDevice));
	const __nv_bfloat16 *q = dqkv, *k = dqkv + D, *v = dqkv + 2 * D;
	const fa2_lay lq = fa2_lay_rows(T, 3 * D, dh), lo = fa2_lay_rows(T, D, dh);

	CK(cudaMemset(O2, 0, nel * 4));
	if (fa2_fwd_launch_lay(O2, lse2, q, k, v, B, T, H, dh, scale, lq, NULL) != 0) { fprintf(stderr, "fa2 fwd refused\n"); return 1; }
	CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
	{
		std::vector<float> b(nel);
		CK(cudaMemcpy(b.data(), O2, nel * 4, cudaMemcpyDeviceToHost));
		double ma, d = max_abs_diff(rO.data(), b.data(), nel, &ma);
		printf("fa2 fwd  : max|O_tc-O_fa2| %.3e (max|O_tc| %.3e)\n", d, ma);
	}
	CK(cudaEventRecord(e0));
	for (int i = 0; i < iters; i++)
		fa2_fwd_launch_lay(O2, lse2, q, k, v, B, T, H, dh, scale, lq, NULL);
	CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
	CK(cudaEventElapsedTime(&ms, e0, e1));
	printf("fa2 fwd  : %.3f ms  (%.1f TFLOP/s)\n", ms / iters, fwd_flops / (ms / iters * 1e-3) / 1e12);

	attn_fa_dvec_kernel<<<(unsigned)((nrows + 255) / 256), 256>>>(dvec, dOf, O2, (int)nrows, T, H, dh);
	CK(cudaMemset(dq2, 0, nel * 4));
	if (fa2_bwd_launch_lay(dq2, dk2, dv2, q, k, v, dO, lse2, dvec, B, T, H, dh, scale, lq, lo, NULL, NULL, NULL) != 0) {
		fprintf(stderr, "fa2 bwd refused\n"); return 1;
	}
	CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
	{
		std::vector<float> b(nel);
		double ma, d;
		CK(cudaMemcpy(b.data(), dq2, nel * 4, cudaMemcpyDeviceToHost));
		d = max_abs_diff(rdq.data(), b.data(), nel, &ma); printf("fa2 bwd  : dq max diff %.3e (max %.3e)\n", d, ma);
		CK(cudaMemcpy(b.data(), dk2, nel * 4, cudaMemcpyDeviceToHost));
		d = max_abs_diff(rdk.data(), b.data(), nel, &ma); printf("fa2 bwd  : dk max diff %.3e (max %.3e)\n", d, ma);
		CK(cudaMemcpy(b.data(), dv2, nel * 4, cudaMemcpyDeviceToHost));
		d = max_abs_diff(rdv.data(), b.data(), nel, &ma); printf("fa2 bwd  : dv max diff %.3e (max %.3e)\n", d, ma);
	}
	CK(cudaEventRecord(e0));
	for (int i = 0; i < iters; i++) {
		cudaMemsetAsync(dq2, 0, nel * 4, 0);
		fa2_bwd_launch_lay(dq2, dk2, dv2, q, k, v, dO, lse2, dvec, B, T, H, dh, scale, lq, lo, NULL, NULL, NULL);
	}
	CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
	CK(cudaEventElapsedTime(&ms, e0, e1));
	printf("fa2 bwd  : %.3f ms  (%.1f TFLOP/s)\n", ms / iters, 2.5 * fwd_flops / (ms / iters * 1e-3) / 1e12);
	return 0;
}
