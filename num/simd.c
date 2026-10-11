/*
**	SIMD-friendly kernels + optional AVX2 / AVX-512 intrinsics (FP32)
*/
#include "simd.h"

#if defined(__AVX512F__)
#include <immintrin.h>

static ann_real ann_dot_avx512(const ann_real *a, const ann_real *b, int n)
{
	int i;
	__m512 vac;
	__m512 va;
	__m512 vb;
	ann_real tmp[16];
	double sum;
	int n16;

	vac = _mm512_setzero_ps();
	n16 = n & ~15;
	for (i = 0; i < n16; i += 16) {
		va = _mm512_loadu_ps(a + i);
		vb = _mm512_loadu_ps(b + i);
		vac = _mm512_add_ps(vac, _mm512_mul_ps(va, vb));
	}
	_mm512_storeu_ps(tmp, vac);
	sum = 0.0;
	for (i = 0; i < 16; i++)
		sum += (double)tmp[i];
	for (i = n16; i < n; i++)
		sum += (double)a[i] * (double)b[i];
	return (ann_real)sum;
}
#elif defined(__AVX2__)
#include <immintrin.h>

static ann_real ann_dot_avx2(const ann_real *a, const ann_real *b, int n)
{
	int i;
	__m256 vac;
	__m256 va;
	__m256 vb;
	ann_real lane[8];
	double sum;
	int n8;

	vac = _mm256_setzero_ps();
	n8 = n & ~7;
	for (i = 0; i < n8; i += 8) {
		va = _mm256_loadu_ps(a + i);
		vb = _mm256_loadu_ps(b + i);
		vac = _mm256_add_ps(vac, _mm256_mul_ps(va, vb));
	}
	_mm256_storeu_ps(lane, vac);
	sum = 0.0;
	for (i = 0; i < 8; i++)
		sum += (double)lane[i];
	for (i = n8; i < n; i++)
		sum += (double)a[i] * (double)b[i];
	return (ann_real)sum;
}
#endif

static ann_real ann_dot_scalar(const ann_real *ANN_RESTRICT a,
	const ann_real *ANN_RESTRICT b, int n)
{
	int i;
	double sum;

	sum = 0.0;
	for (i = 0; i < n; i++)
		sum += (double)a[i] * (double)b[i];
	return (ann_real)sum;
}

ann_real ann_dot(const ann_real *ANN_RESTRICT a,
	const ann_real *ANN_RESTRICT b, int n)
{
	if (!a || !b || n <= 0)
		return 0.0f;
#if defined(__AVX512F__)
	return ann_dot_avx512(a, b, n);
#elif defined(__AVX2__)
	return ann_dot_avx2(a, b, n);
#else
	return ann_dot_scalar(a, b, n);
#endif
}

void ann_axpy(ann_real *ANN_RESTRICT y, ann_real alpha,
	const ann_real *ANN_RESTRICT x, int n)
{
	int i;

	if (!y || !x || n <= 0)
		return;
	for (i = 0; i < n; i++)
		y[i] += alpha * x[i];
}

void ann_scale(ann_real *ANN_RESTRICT v, ann_real s, int n)
{
	int i;

	if (!v || n <= 0)
		return;
	for (i = 0; i < n; i++)
		v[i] *= s;
}

const char *ann_simd_backend(void)
{
#if defined(__AVX512F__)
	return "avx512";
#elif defined(__AVX2__)
	return "avx2";
#else
	return "scalar";
#endif
}
