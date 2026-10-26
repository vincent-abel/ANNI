/*
** attn_fa2.cuh: fused causal attention on BF16 tensor cores with mma.sync
** (FlashAttention-2 structure), included by gpu_cuda.cu.
**
** Forward: one block per (b, h, 64-query tile), four warps of 16 rows. Q
** stays in A fragments; 64-key tiles of K and V stream through shared
** memory (ldmatrix); S = Q K^T, online softmax in registers (row max and
** sum through quad shuffles), P is repacked from the C fragments into the
** A fragments of O += P V without touching memory; O is rescaled in
** registers. Writes O [b][t][h*dh] (FP32) and lse[b,h,i] = m_i + log l_i.
** dh in {16, 32, 48, 64} (template), T % 64 == 0, no seqlens.
*/
#ifndef ATTN_FA2_CUH
#define ATTN_FA2_CUH

#define FA2_BQ		64
#define FA2_BK		64
/* DH (head size, 16..64 by 16) is a template parameter; FA2_LDS(DH) is the
** smem row stride in bf16 (DH + 8: 144 B at 64, conflict-free ldmatrix). */
#define FA2_LDS(DH)	((DH) + 8)
#define FA2_WARPS	4

__device__ __forceinline__ unsigned fa2_smem_addr(const void *p)
{
	return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void fa2_ldmatrix_x4(unsigned *r, const void *p)
{
	unsigned a = fa2_smem_addr(p);

	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
	    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void fa2_ldmatrix_x4_trans(unsigned *r, const void *p)
{
	unsigned a = fa2_smem_addr(p);

	asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
	    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

/* D = A (16x16 bf16, row) x B (16x8 bf16, col) + D, FP32. */
__device__ __forceinline__ void fa2_mma(float *d, const unsigned *a, const unsigned *b)
{
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
	    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
	    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
	    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned fa2_pack_bf16(float lo, float hi)
{
	__nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);

	return *(unsigned *)&v;
}

/* 1 when the FA2 kernels take this shape. */
static int fa2_ok(int T, int dh)
{
	return (T % FA2_BQ) == 0 && (dh == 16 || dh == 32 || dh == 48 || dh == 64);
}

/* Copy a 64 x DH bf16 tile (row stride src_ld) into smem (stride FA2_LDS(DH)), 16 B per thread per step. */
template <int DH>
__device__ __forceinline__ void fa2_load_tile(__nv_bfloat16 *dst, const __nv_bfloat16 *src, int src_ld)
{
	int i;

	for (i = threadIdx.x; i < FA2_BK * (DH / 8); i += FA2_WARPS * 32) {
		int row = i / (DH / 8);
		int c8 = (i % (DH / 8)) * 8;

		*(int4 *)(dst + row * FA2_LDS(DH) + c8) = *(const int4 *)(src + (size_t)row * src_ld + c8);
	}
}

template <int DH>
__global__ static void __launch_bounds__(128)
attn_fa2_fwd_kernel(float *out, float *lse, const __nv_bfloat16 *q,
	const __nv_bfloat16 *k, const __nv_bfloat16 *v, int T, int H, float scale)
{
	const int NT = DH / 8;		/* n-tiles over dh */
	const int KS = DH / 16;		/* k-steps over dh */
	const int LDS = FA2_LDS(DH);
	__shared__ __align__(16) __nv_bfloat16 Qs[FA2_BQ * FA2_LDS(DH)];
	__shared__ __align__(16) __nv_bfloat16 Ks[FA2_BK * FA2_LDS(DH)];
	__shared__ __align__(16) __nv_bfloat16 Vs[FA2_BK * FA2_LDS(DH)];
	const int bh = blockIdx.x;
	const int b = bh / H;
	const int h = bh % H;
	const int qt = blockIdx.y;
	const int q0 = qt * FA2_BQ;
	const int warp = threadIdx.x >> 5;
	const int lane = threadIdx.x & 31;
	const int D = H * DH;
	const __nv_bfloat16 *qsrc = q + ((size_t)bh * T + q0) * DH;
	const __nv_bfloat16 *ksrc = k + (size_t)bh * T * DH;
	const __nv_bfloat16 *vsrc = v + (size_t)bh * T * DH;
	unsigned qa[KS][4];		/* Q A-fragments, KS k-steps of 16 */
	float oacc[NT][4];		/* O: NT n-tiles of 8 columns */
	float m0 = -1e30f, m1 = -1e30f;	/* running max of rows lane/4 and lane/4 + 8 */
	float l0 = 0.0f, l1 = 0.0f;	/* running sum */
	const int r0 = warp * 16 + (lane >> 2);	/* this lane's rows in the tile: r0, r0 + 8 */
	const int qi0 = q0 + r0;
	const int qi1 = qi0 + 8;
	int kt, kk, n, i;

	for (n = 0; n < NT; n++)
		for (i = 0; i < 4; i++)
			oacc[n][i] = 0.0f;
	fa2_load_tile<DH>(Qs, qsrc, DH);
	__syncthreads();
	/* Q fragments: ldmatrix.x4 of {rows 0-7 | 8-15} x {k 0-7 | 8-15}. */
	for (kk = 0; kk < KS; kk++) {
		const __nv_bfloat16 *p = Qs + (warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS
		    + kk * 16 + (lane >> 4) * 8;

		fa2_ldmatrix_x4(qa[kk], p);
	}
	for (kt = 0; kt <= qt; kt++) {
		const int k0 = kt * FA2_BK;
		float s[8][4];
		float mx0, mx1, mn0, mn1, a0, a1, rs0, rs1;
		unsigned pa[4][4];

		__syncthreads();	/* previous tile fully consumed */
		fa2_load_tile<DH>(Ks, ksrc + (size_t)k0 * DH, DH);
		fa2_load_tile<DH>(Vs, vsrc + (size_t)k0 * DH, DH);
		__syncthreads();
		/* S = Q K^T: 8 n-tiles of 8 keys; B fragments by ldmatrix.x4 (two n-tiles per load). */
		for (n = 0; n < 8; n++)
			for (i = 0; i < 4; i++)
				s[n][i] = 0.0f;
		for (kk = 0; kk < KS; kk++) {
			for (n = 0; n < 8; n += 2) {
				unsigned kb[4];
				const __nv_bfloat16 *p = Ks + (n * 8 + (lane & 7) + (lane >> 4) * 8) * LDS
				    + kk * 16 + ((lane >> 3) & 1) * 8;

				fa2_ldmatrix_x4(kb, p);
				fa2_mma(s[n], qa[kk], kb);
				fa2_mma(s[n + 1], qa[kk], kb + 2);
			}
		}
		/* Scale, causal mask on the diagonal tile, row max. */
		mx0 = -1e30f;
		mx1 = -1e30f;
		for (n = 0; n < 8; n++) {
			const int kj = k0 + n * 8 + (lane & 3) * 2;

			s[n][0] *= scale;
			s[n][1] *= scale;
			s[n][2] *= scale;
			s[n][3] *= scale;
			if (kt == qt) {
				if (kj > qi0) s[n][0] = -1e30f;
				if (kj + 1 > qi0) s[n][1] = -1e30f;
				if (kj > qi1) s[n][2] = -1e30f;
				if (kj + 1 > qi1) s[n][3] = -1e30f;
			}
			mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
			mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
		}
		mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
		mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
		mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
		mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
		mn0 = fmaxf(m0, mx0);
		mn1 = fmaxf(m1, mx1);
		a0 = __expf(m0 - mn0);
		a1 = __expf(m1 - mn1);
		m0 = mn0;
		m1 = mn1;
		/* P = exp(S - m), row sums (FP32, before the BF16 rounding of P). */
		rs0 = 0.0f;
		rs1 = 0.0f;
		for (n = 0; n < 8; n++) {
			s[n][0] = __expf(s[n][0] - mn0);
			s[n][1] = __expf(s[n][1] - mn0);
			s[n][2] = __expf(s[n][2] - mn1);
			s[n][3] = __expf(s[n][3] - mn1);
			rs0 += s[n][0] + s[n][1];
			rs1 += s[n][2] + s[n][3];
		}
		rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
		rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
		rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
		rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
		l0 = l0 * a0 + rs0;
		l1 = l1 * a1 + rs1;
		/* O *= alpha per row. */
		for (n = 0; n < NT; n++) {
			oacc[n][0] *= a0;
			oacc[n][1] *= a0;
			oacc[n][2] *= a1;
			oacc[n][3] *= a1;
		}
		/* P as A fragments: k-step kk takes n-tiles 2kk and 2kk+1. */
		for (kk = 0; kk < 4; kk++) {
			pa[kk][0] = fa2_pack_bf16(s[2 * kk][0], s[2 * kk][1]);
			pa[kk][1] = fa2_pack_bf16(s[2 * kk][2], s[2 * kk][3]);
			pa[kk][2] = fa2_pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
			pa[kk][3] = fa2_pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
		}
		/* O += P V: V as B fragments via ldmatrix.trans (two n-tiles per load). */
		for (kk = 0; kk < 4; kk++) {
			for (n = 0; n < NT; n += 2) {
				unsigned vb[4];
				const __nv_bfloat16 *p = Vs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS
				    + n * 8 + (lane >> 4) * 8;

				fa2_ldmatrix_x4_trans(vb, p);
				fa2_mma(oacc[n], pa[kk], vb);
				fa2_mma(oacc[n + 1], pa[kk], vb + 2);
			}
		}
	}
	/* O / l, written as float2 pairs; lse by the first lane of each quad. */
	{
		const float i0 = 1.0f / l0;
		const float i1 = 1.0f / l1;
		float *o0 = out + ((size_t)b * T + qi0) * D + h * DH + (lane & 3) * 2;
		float *o1 = o0 + (size_t)8 * D;

		for (n = 0; n < NT; n++) {
			*(float2 *)(o0 + n * 8) = make_float2(oacc[n][0] * i0, oacc[n][1] * i0);
			*(float2 *)(o1 + n * 8) = make_float2(oacc[n][2] * i1, oacc[n][3] * i1);
		}
		if ((lane & 3) == 0) {
			lse[(size_t)bh * T + qi0] = m0 + __logf(l0);
			lse[(size_t)bh * T + qi1] = m1 + __logf(l1);
		}
	}
}

/*
** Backward (FA-2 order): one block per (b, h, 64-key tile), four warps of
** 16 keys. K and V rows stay in A fragments; 64-query tiles of Q and dO
** stream through shared memory from the diagonal on. Per tile: S^T = K Q^T,
** P^T = exp(S^T - lse) (causal), dV += P^T dO, dP^T = V dO^T,
** dS^T = P^T (dP^T - D), dK += dS^T Q, and dQ += dS K through a BF16 dS^T
** tile in shared memory (transposed by ldmatrix) and FP32 atomics. dq must
** be zeroed by the caller; dk, dv are written. dq, dk, dv in the model
** layout [b][t][h*dh]; dvec[b,h,i] = <dO_i, O_i>.
*/
template <int DH>
__global__ static void __launch_bounds__(128)
attn_fa2_bwd_kernel(float *dq, float *dk, float *dv, const __nv_bfloat16 *q,
	const __nv_bfloat16 *k, const __nv_bfloat16 *v, const __nv_bfloat16 *dO,
	const float *lse, const float *dvec, int T, int H, float scale)
{
	const int NT = DH / 8;
	const int KS = DH / 16;
	const int LDS = FA2_LDS(DH);
	/* dS^T [key][query] needs 64 columns: its stride is FA2_LDS(64) whatever DH is. */
	const int LDT = FA2_LDS(64);
	__shared__ __align__(16) __nv_bfloat16 Ks[FA2_BK * FA2_LDS(DH)];
	__shared__ __align__(16) __nv_bfloat16 Qs[FA2_BQ * FA2_LDS(DH)];
	__shared__ __align__(16) __nv_bfloat16 dOs[FA2_BQ * FA2_LDS(DH)];
	__shared__ __align__(16) __nv_bfloat16 dSt[FA2_BK * FA2_LDS(64)];	/* dS^T [key][query] */
	__shared__ __align__(16) __nv_bfloat16 Vs0[FA2_BK * FA2_LDS(DH)];	/* V staged once */
	__shared__ float ls[FA2_BQ];
	__shared__ float dvs[FA2_BQ];
	const int bh = blockIdx.x;
	const int b = bh / H;
	const int h = bh % H;
	const int kt = blockIdx.y;
	const int k0 = kt * FA2_BK;
	const int warp = threadIdx.x >> 5;
	const int lane = threadIdx.x & 31;
	const int D = H * DH;
	const __nv_bfloat16 *ksrc = k + ((size_t)bh * T + k0) * DH;
	const __nv_bfloat16 *vsrc = v + ((size_t)bh * T + k0) * DH;
	const __nv_bfloat16 *qsrc = q + (size_t)bh * T * DH;
	const __nv_bfloat16 *dsrc = dO + (size_t)bh * T * DH;
	unsigned ka[KS][4];		/* K rows of this warp as A fragments (k = dh) */
	unsigned va[KS][4];		/* V rows */
	float dkacc[NT][4];
	float dvacc[NT][4];
	const int r0 = warp * 16 + (lane >> 2);	/* this lane's key rows in the tile: r0, r0 + 8 */
	const int kj0 = k0 + r0;
	const int kj1 = kj0 + 8;
	int qt, kk, n, i;

	for (n = 0; n < NT; n++)
		for (i = 0; i < 4; i++) {
			dkacc[n][i] = 0.0f;
			dvacc[n][i] = 0.0f;
		}
	fa2_load_tile<DH>(Ks, ksrc, DH);
	fa2_load_tile<DH>(Vs0, vsrc, DH);
	__syncthreads();
	for (kk = 0; kk < KS; kk++) {
		const int off = (warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS
		    + kk * 16 + (lane >> 4) * 8;

		fa2_ldmatrix_x4(ka[kk], Ks + off);
		fa2_ldmatrix_x4(va[kk], Vs0 + off);
	}
	for (qt = kt; qt < T / FA2_BQ; qt++) {
		const int q0 = qt * FA2_BQ;
		float s[8][4];		/* S^T then P^T: 16 keys x 64 queries */
		float dp[8][4];		/* dP^T then dS^T */
		unsigned pa[4][4];

		__syncthreads();	/* previous tile's Qs / dOs / dSt consumed */
		fa2_load_tile<DH>(Qs, qsrc + (size_t)q0 * DH, DH);
		fa2_load_tile<DH>(dOs, dsrc + (size_t)q0 * DH, DH);
		if (threadIdx.x < FA2_BQ) {
			ls[threadIdx.x] = lse[(size_t)bh * T + q0 + threadIdx.x];
			dvs[threadIdx.x] = dvec[(size_t)bh * T + q0 + threadIdx.x];
		}
		__syncthreads();
		for (n = 0; n < 8; n++)
			for (i = 0; i < 4; i++) {
				s[n][i] = 0.0f;
				dp[n][i] = 0.0f;
			}
		/* S^T = K Q^T and dP^T = V dO^T: B fragments from Q / dO rows (non-trans). */
		for (kk = 0; kk < KS; kk++) {
			for (n = 0; n < 8; n += 2) {
				unsigned qb[4], db[4];
				const int off = (n * 8 + (lane & 7) + (lane >> 4) * 8) * LDS
				    + kk * 16 + ((lane >> 3) & 1) * 8;

				fa2_ldmatrix_x4(qb, Qs + off);
				fa2_ldmatrix_x4(db, dOs + off);
				fa2_mma(s[n], ka[kk], qb);
				fa2_mma(s[n + 1], ka[kk], qb + 2);
				fa2_mma(dp[n], va[kk], db);
				fa2_mma(dp[n + 1], va[kk], db + 2);
			}
		}
		/* P^T = exp(scale S^T - lse_q), causal; dS^T = P^T (dP^T - D_q). */
		for (n = 0; n < 8; n++) {
			const int qc = n * 8 + (lane & 3) * 2;	/* query columns qc, qc + 1 */
			const int qi = q0 + qc;
			const float l0 = ls[qc], l1 = ls[qc + 1];
			const float d0 = dvs[qc], d1 = dvs[qc + 1];

			s[n][0] = kj0 <= qi ? __expf(s[n][0] * scale - l0) : 0.0f;
			s[n][1] = kj0 <= qi + 1 ? __expf(s[n][1] * scale - l1) : 0.0f;
			s[n][2] = kj1 <= qi ? __expf(s[n][2] * scale - l0) : 0.0f;
			s[n][3] = kj1 <= qi + 1 ? __expf(s[n][3] * scale - l1) : 0.0f;
			/* The WMMA kernel used the BF16-rounded P^T here; so do we. */
			{
				float p0 = __bfloat162float(__float2bfloat16(s[n][0]));
				float p1 = __bfloat162float(__float2bfloat16(s[n][1]));
				float p2 = __bfloat162float(__float2bfloat16(s[n][2]));
				float p3 = __bfloat162float(__float2bfloat16(s[n][3]));

				dp[n][0] = p0 * (dp[n][0] - d0);
				dp[n][1] = p1 * (dp[n][1] - d1);
				dp[n][2] = p2 * (dp[n][2] - d0);
				dp[n][3] = p3 * (dp[n][3] - d1);
			}
		}
		/* dV += P^T dO: A = P^T (k = queries), B = dO rows via ldmatrix.trans. */
		for (kk = 0; kk < 4; kk++) {
			pa[kk][0] = fa2_pack_bf16(s[2 * kk][0], s[2 * kk][1]);
			pa[kk][1] = fa2_pack_bf16(s[2 * kk][2], s[2 * kk][3]);
			pa[kk][2] = fa2_pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
			pa[kk][3] = fa2_pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
		}
		for (kk = 0; kk < 4; kk++) {
			for (n = 0; n < NT; n += 2) {
				unsigned ob[4];
				const int off = (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS
				    + n * 8 + (lane >> 4) * 8;

				fa2_ldmatrix_x4_trans(ob, dOs + off);
				fa2_mma(dvacc[n], pa[kk], ob);
				fa2_mma(dvacc[n + 1], pa[kk], ob + 2);
			}
		}
		/* dK += dS^T Q: A = dS^T, B = Q rows via ldmatrix.trans; stage dS^T (BF16) for dQ. */
		for (kk = 0; kk < 4; kk++) {
			pa[kk][0] = fa2_pack_bf16(dp[2 * kk][0], dp[2 * kk][1]);
			pa[kk][1] = fa2_pack_bf16(dp[2 * kk][2], dp[2 * kk][3]);
			pa[kk][2] = fa2_pack_bf16(dp[2 * kk + 1][0], dp[2 * kk + 1][1]);
			pa[kk][3] = fa2_pack_bf16(dp[2 * kk + 1][2], dp[2 * kk + 1][3]);
		}
		for (n = 0; n < 8; n++) {
			const int qc = n * 8 + (lane & 3) * 2;

			*(unsigned *)(dSt + r0 * LDT + qc) = pa[n >> 1][(n & 1) * 2];
			*(unsigned *)(dSt + (r0 + 8) * LDT + qc) = pa[n >> 1][(n & 1) * 2 + 1];
		}
		for (kk = 0; kk < 4; kk++) {
			for (n = 0; n < NT; n += 2) {
				unsigned qb[4];
				const int off = (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS
				    + n * 8 + (lane >> 4) * 8;

				fa2_ldmatrix_x4_trans(qb, Qs + off);
				fa2_mma(dkacc[n], pa[kk], qb);
				fa2_mma(dkacc[n + 1], pa[kk], qb + 2);
			}
		}
		__syncthreads();	/* dS^T complete from all warps */
		/*
		** dQ (this warp: 16 queries x dh) = scale dS K over the 64 keys:
		** A = dS from dS^T via ldmatrix.trans, B = K rows via ldmatrix.trans.
		*/
		{
			float dqa[NT][4];
			float *dst;

			for (n = 0; n < NT; n++)
				for (i = 0; i < 4; i++)
					dqa[n][i] = 0.0f;
			for (kk = 0; kk < 4; kk++) {
				unsigned da[4];
				const int aoff = (kk * 16 + (lane & 7) + (lane >> 4) * 8) * LDT
				    + warp * 16 + ((lane >> 3) & 1) * 8;

				fa2_ldmatrix_x4_trans(da, dSt + aoff);
				for (n = 0; n < NT; n += 2) {
					unsigned kb[4];
					const int boff = (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS
					    + n * 8 + (lane >> 4) * 8;

					fa2_ldmatrix_x4_trans(kb, Ks + boff);
					fa2_mma(dqa[n], da, kb);
					fa2_mma(dqa[n + 1], da, kb + 2);
				}
			}
			dst = dq + ((size_t)b * T + q0 + warp * 16 + (lane >> 2)) * D + h * DH + (lane & 3) * 2;
			for (n = 0; n < NT; n++) {
				atomicAdd(dst + n * 8, dqa[n][0] * scale);
				atomicAdd(dst + n * 8 + 1, dqa[n][1] * scale);
				atomicAdd(dst + (size_t)8 * D + n * 8, dqa[n][2] * scale);
				atomicAdd(dst + (size_t)8 * D + n * 8 + 1, dqa[n][3] * scale);
			}
		}
	}
	/* dK (scaled) and dV out, float2 pairs. */
	{
		float *k0p = dk + ((size_t)b * T + kj0) * D + h * DH + (lane & 3) * 2;
		float *k1p = k0p + (size_t)8 * D;
		float *v0p = dv + ((size_t)b * T + kj0) * D + h * DH + (lane & 3) * 2;
		float *v1p = v0p + (size_t)8 * D;

		for (n = 0; n < NT; n++) {
			*(float2 *)(k0p + n * 8) = make_float2(dkacc[n][0] * scale, dkacc[n][1] * scale);
			*(float2 *)(k1p + n * 8) = make_float2(dkacc[n][2] * scale, dkacc[n][3] * scale);
			*(float2 *)(v0p + n * 8) = make_float2(dvacc[n][0], dvacc[n][1]);
			*(float2 *)(v1p + n * 8) = make_float2(dvacc[n][2], dvacc[n][3]);
		}
	}
}

static int fa2_bwd_launch(float *dq, float *dk, float *dv, const __nv_bfloat16 *q,
	const __nv_bfloat16 *k, const __nv_bfloat16 *v, const __nv_bfloat16 *dO,
	const float *lse, const float *dvec, int B, int T, int H, int dh, float scale)
{
	dim3 grid(B * H, T / FA2_BK);

	if (!fa2_ok(T, dh))
		return -1;
	switch (dh) {
	case 16: attn_fa2_bwd_kernel<16><<<grid, 32 * FA2_WARPS>>>(dq, dk, dv, q, k, v, dO, lse, dvec, T, H, scale); break;
	case 32: attn_fa2_bwd_kernel<32><<<grid, 32 * FA2_WARPS>>>(dq, dk, dv, q, k, v, dO, lse, dvec, T, H, scale); break;
	case 48: attn_fa2_bwd_kernel<48><<<grid, 32 * FA2_WARPS>>>(dq, dk, dv, q, k, v, dO, lse, dvec, T, H, scale); break;
	default: attn_fa2_bwd_kernel<64><<<grid, 32 * FA2_WARPS>>>(dq, dk, dv, q, k, v, dO, lse, dvec, T, H, scale); break;
	}
	return 0;
}

/* Host launcher; q, k, v head-major [b][h][t][dh] BF16. */
/* Host launcher; q, k, v head-major [b][h][t][dh] BF16; -1 = shape not covered. */
static int fa2_fwd_launch(float *out, float *lse, const __nv_bfloat16 *q,
	const __nv_bfloat16 *k, const __nv_bfloat16 *v, int B, int T, int H, int dh,
	float scale)
{
	dim3 grid(B * H, T / FA2_BQ);

	if (!fa2_ok(T, dh))
		return -1;
	switch (dh) {
	case 16: attn_fa2_fwd_kernel<16><<<grid, 32 * FA2_WARPS>>>(out, lse, q, k, v, T, H, scale); break;
	case 32: attn_fa2_fwd_kernel<32><<<grid, 32 * FA2_WARPS>>>(out, lse, q, k, v, T, H, scale); break;
	case 48: attn_fa2_fwd_kernel<48><<<grid, 32 * FA2_WARPS>>>(out, lse, q, k, v, T, H, scale); break;
	default: attn_fa2_fwd_kernel<64><<<grid, 32 * FA2_WARPS>>>(out, lse, q, k, v, T, H, scale); break;
	}
	return 0;
}

#endif /* ATTN_FA2_CUH */
