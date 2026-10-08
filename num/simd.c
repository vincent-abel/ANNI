/*
**	SIMD-friendly kernels + optional AVX2 / AVX-512 intrinsics
*/
#include "simd.h"

#if defined(__AVX512F__)
#include <immintrin.h>

static double ann_dot_avx512(const double *a, const double *b, int n)
{
	int i;
	__m512d vac;
	__m512d va;
	__m512d vb;
	double tmp[8];
	double sum;
	int n8;

	vac = _mm512_setzero_pd();
	n8 = n & ~7;
	for (i = 0; i < n8; i += 8) {
		va = _mm512_loadu_pd(a + i);
		vb = _mm512_loadu_pd(b + i);
		vac = _mm512_add_pd(vac, _mm512_mul_pd(va, vb));
	}
	_mm512_storeu_pd(tmp, vac);
	sum = tmp[0] + tmp[1] + tmp[2] + tmp[3]
	    + tmp[4] + tmp[5] + tmp[6] + tmp[7];
	for (; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}
#elif defined(__AVX2__)
#include <immintrin.h>

static double ann_dot_avx2(const double *a, const double *b, int n)
{
	int i;
	__m256d vac;
	__m256d va;
	__m256d vb;
	double lane[4];
	double sum;
	int n4;

	vac = _mm256_setzero_pd();
	n4 = n & ~3;
	for (i = 0; i < n4; i += 4) {
		va = _mm256_loadu_pd(a + i);
		vb = _mm256_loadu_pd(b + i);
		vac = _mm256_add_pd(vac, _mm256_mul_pd(va, vb));
	}
	_mm256_storeu_pd(lane, vac);
	sum = lane[0] + lane[1] + lane[2] + lane[3];
	for (; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}
#endif

static double ann_dot_scalar(const double *ANN_RESTRICT a,
	const double *ANN_RESTRICT b, int n)
{
	int i;
	double sum;

	sum = 0.0;
	for (i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

double ann_dot(const double *ANN_RESTRICT a, const double *ANN_RESTRICT b, int n)
{
	if (!a || !b || n <= 0)
		return 0.0;
#if defined(__AVX512F__)
	return ann_dot_avx512(a, b, n);
#elif defined(__AVX2__)
	return ann_dot_avx2(a, b, n);
#else
	return ann_dot_scalar(a, b, n);
#endif
}

void ann_axpy(double *ANN_RESTRICT y, double alpha,
	const double *ANN_RESTRICT x, int n)
{
	int i;

	if (!y || !x || n <= 0)
		return;
	for (i = 0; i < n; i++)
		y[i] += alpha * x[i];
}

void ann_scale(double *ANN_RESTRICT v, double s, int n)
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
