/*
**	Projections of the latent space viewer: PCA, kNN graph, UMAP layout
*/
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "project.h"

#define PCA_ROUNDS	5
#define UMAP_A		1.577f	/* the fit of min_dist 0.1, spread 1 */
#define UMAP_B		0.8951f
#define UMAP_NEGATIVE	5
#define UMAP_CLIP	4.0f
#define SIGMA_STEPS	64
#define KNN_BLOCK	32

static uint64_t next_random(uint64_t *state)
{
	uint64_t x = *state;

	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	*state = x;
	return x;
}

/* Gram-Schmidt on the k columns of q, stored as k rows of d. */
static void orthonormalize(float *q, int d, int k)
{
	int a;
	int b;
	int i;

	for (a = 0; a < k; a++) {
		float *qa = q + (size_t)a * (size_t)d;
		double norm = 0.0;

		for (b = 0; b < a; b++) {
			const float *qb = q + (size_t)b * (size_t)d;
			double dot = 0.0;

			for (i = 0; i < d; i++)
				dot += (double)qa[i] * qb[i];
			for (i = 0; i < d; i++)
				qa[i] -= (float)dot * qb[i];
		}
		for (i = 0; i < d; i++)
			norm += (double)qa[i] * qa[i];
		norm = norm > 0.0 ? 1.0 / sqrt(norm) : 0.0;
		for (i = 0; i < d; i++)
			qa[i] *= (float)norm;
	}
}

/* out = x q^T: n rows of k. */
static void scores(const float *x, long n, int d, int k, const float *q, float *out)
{
	long r;

	#pragma omp parallel for schedule(static)
	for (r = 0; r < n; r++) {
		const float *row = x + (size_t)r * (size_t)d;
		int a;

		for (a = 0; a < k; a++) {
			const float *qa = q + (size_t)a * (size_t)d;
			float dot = 0.0f;
			int i;

			for (i = 0; i < d; i++)
				dot += row[i] * qa[i];
			out[(size_t)r * (size_t)k + (size_t)a] = dot;
		}
	}
}

/* q = y^T x: k rows of d. */
static void back_project(const float *x, long n, int d, int k, const float *y, float *q)
{
	int a;

	memset(q, 0, sizeof(float) * (size_t)k * (size_t)d);
	#pragma omp parallel for schedule(dynamic)
	for (a = 0; a < k; a++) {
		float *qa = q + (size_t)a * (size_t)d;
		long r;

		for (r = 0; r < n; r++) {
			const float *row = x + (size_t)r * (size_t)d;
			float w = y[(size_t)r * (size_t)k + (size_t)a];
			int i;

			for (i = 0; i < d; i++)
				qa[i] += w * row[i];
		}
	}
}

int pca_fit(float *x, long n, int d, int k, float *out)
{
	double *mean = calloc((size_t)d, sizeof(double));
	float *q = malloc(sizeof(float) * (size_t)k * (size_t)d);
	uint64_t seed = 0x2545f4914f6cdd1dULL;
	long r;
	int i;

	if (!mean || !q) {
		free(mean);
		free(q);
		return -1;
	}
	for (r = 0; r < n; r++) {
		for (i = 0; i < d; i++)
			mean[i] += x[(size_t)r * (size_t)d + (size_t)i];
	}
	for (i = 0; i < d; i++)
		mean[i] /= (double)n;
	#pragma omp parallel for schedule(static) private(i)
	for (r = 0; r < n; r++) {
		for (i = 0; i < d; i++)
			x[(size_t)r * (size_t)d + (size_t)i] -= (float)mean[i];
	}
	for (i = 0; i < k * d; i++)
		q[i] = (float)(next_random(&seed) >> 40) / (float)(1 << 24) - 0.5f;
	orthonormalize(q, d, k);
	for (i = 0; i < PCA_ROUNDS; i++) {
		scores(x, n, d, k, q, out);
		back_project(x, n, d, k, out, q);
		orthonormalize(q, d, k);
	}
	scores(x, n, d, k, q, out);
	free(mean);
	free(q);
	return 0;
}

int knn_graph(const float *x, long n, int d, int *idx, float *dist)
{
	float *unit = malloc(sizeof(float) * (size_t)n * (size_t)d);
	long r;

	if (!unit)
		return -1;
	#pragma omp parallel for schedule(static)
	for (r = 0; r < n; r++) {
		const float *row = x + (size_t)r * (size_t)d;
		float *u = unit + (size_t)r * (size_t)d;
		double norm = 1e-12;
		int i;

		for (i = 0; i < d; i++)
			norm += (double)row[i] * row[i];
		norm = 1.0 / sqrt(norm);
		for (i = 0; i < d; i++)
			u[i] = row[i] * (float)norm;
	}
	/* A block of rows stays in cache while every other row goes past it. */
	#pragma omp parallel for schedule(dynamic)
	for (r = 0; r < n; r += KNN_BLOCK) {
		long end = r + KNN_BLOCK < n ? r + KNN_BLOCK : n;
		int found[KNN_BLOCK];
		long c;
		long q;

		memset(found, 0, sizeof(found));
		for (c = 0; c < n; c++) {
			const float *b = unit + (size_t)c * (size_t)d;

			for (q = r; q < end; q++) {
				const float *a = unit + (size_t)q * (size_t)d;
				int *best = idx + (size_t)q * KNN_K;
				float *near = dist + (size_t)q * KNN_K;
				int *have = &found[q - r];
				float dot = 0.0f;
				float dd;
				int i;

				if (c == q)
					continue;
				for (i = 0; i < d; i++)
					dot += a[i] * b[i];
				dd = 1.0f - dot;
				if (*have == KNN_K && dd >= near[KNN_K - 1])
					continue;
				i = *have < KNN_K ? (*have)++ : KNN_K - 1;
				while (i > 0 && near[i - 1] > dd) {
					near[i] = near[i - 1];
					best[i] = best[i - 1];
					i--;
				}
				near[i] = dd;
				best[i] = (int)c;
			}
		}
	}
	free(unit);
	return 0;
}

typedef struct	edge {
	int	from;
	int	to;
	float	weight;
} edge;

static int edge_order(const void *pa, const void *pb)
{
	const edge *a = pa;
	const edge *b = pb;

	if (a->from != b->from)
		return a->from < b->from ? -1 : 1;
	if (a->to != b->to)
		return a->to < b->to ? -1 : 1;
	return 0;
}

/* The membership of each neighbour: exp(-(d - rho) / sigma), summing to log2 k. */
static void memberships(const float *dist, float *w)
{
	float target = log2f((float)KNN_K);
	float rho = dist[0];
	float lo = 0.0f;
	float hi = 1e9f;
	float sigma = 1.0f;
	int s;
	int i;

	for (s = 0; s < SIGMA_STEPS; s++) {
		float sum = 0.0f;

		for (i = 0; i < KNN_K; i++)
			sum += expf(-fmaxf(dist[i] - rho, 0.0f) / sigma);
		if (fabsf(sum - target) < 1e-4f)
			break;
		if (sum > target) {
			hi = sigma;
			sigma = (lo + hi) / 2.0f;
		} else {
			lo = sigma;
			sigma = hi >= 1e9f ? sigma * 2.0f : (lo + hi) / 2.0f;
		}
	}
	for (i = 0; i < KNN_K; i++)
		w[i] = expf(-fmaxf(dist[i] - rho, 0.0f) / sigma);
}

/* Both directions of each edge, joined as a + b - ab; n_out = edges kept. */
static edge *fuzzy_graph(const int *idx, const float *dist, long n, long *n_out)
{
	long total = n * KNN_K * 2;
	edge *e = malloc(sizeof(edge) * (size_t)total);
	long kept = 0;
	long r;
	long i;

	if (!e)
		return NULL;
	for (r = 0; r < n; r++) {
		float w[KNN_K];
		int c;

		memberships(dist + (size_t)r * KNN_K, w);
		for (c = 0; c < KNN_K; c++) {
			int to = idx[(size_t)r * KNN_K + (size_t)c];

			e[kept].from = (int)r;
			e[kept].to = to;
			e[kept++].weight = w[c];
			e[kept].from = to;
			e[kept].to = (int)r;
			e[kept++].weight = w[c];
		}
	}
	qsort(e, (size_t)kept, sizeof(edge), edge_order);
	total = kept;
	kept = 0;
	for (i = 0; i < total; i++) {
		if (kept && e[kept - 1].from == e[i].from && e[kept - 1].to == e[i].to) {
			float a = e[kept - 1].weight;

			e[kept - 1].weight = a + e[i].weight - a * e[i].weight;
		} else {
			e[kept++] = e[i];
		}
	}
	*n_out = kept;
	return e;
}

static float clip(float v)
{
	return v > UMAP_CLIP ? UMAP_CLIP : v < -UMAP_CLIP ? -UMAP_CLIP : v;
}

int umap_layout(const int *idx, const float *dist, long n, float *pos, int epochs)
{
	long n_edges;
	edge *e = fuzzy_graph(idx, dist, n, &n_edges);
	float top = 0.0f;
	long i;
	int epoch;

	if (!e)
		return -1;
	for (i = 0; i < n_edges; i++)
		top = fmaxf(top, e[i].weight);
	for (epoch = 0; epoch < epochs; epoch++) {
		float alpha = 1.0f - (float)epoch / (float)epochs;

		#pragma omp parallel
		{
			uint64_t seed = 0x9e3779b97f4a7c15ULL + (uint64_t)epoch * 7919u
			    + (uint64_t)(uintptr_t)&seed;
			long k;

			#pragma omp for schedule(static)
			for (k = 0; k < n_edges; k++) {
				float *a = pos + (size_t)e[k].from * 3;
				float *b = pos + (size_t)e[k].to * 3;
				float dx[3];
				float d2;
				float g;
				int c;
				int s;

				/* An edge of weight w is drawn in a share w of the epochs. */
				if ((float)(next_random(&seed) >> 40) / (float)(1 << 24)
				    > e[k].weight / top)
					continue;
				for (c = 0; c < 3; c++)
					dx[c] = a[c] - b[c];
				d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
				g = 0.0f;
				if (d2 > 0.0f)
					g = -2.0f * UMAP_A * UMAP_B * powf(d2, UMAP_B - 1.0f)
					    / (UMAP_A * powf(d2, UMAP_B) + 1.0f);
				for (c = 0; c < 3; c++) {
					float step = clip(g * dx[c]) * alpha;

					a[c] += step;
					b[c] -= step;
				}
				for (s = 0; s < UMAP_NEGATIVE; s++) {
					long other = (long)(next_random(&seed) % (uint64_t)n);
					float *o = pos + (size_t)other * 3;

					if (other == e[k].from)
						continue;
					for (c = 0; c < 3; c++)
						dx[c] = a[c] - o[c];
					d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
					g = 2.0f * UMAP_B / ((0.001f + d2)
					    * (UMAP_A * powf(d2, UMAP_B) + 1.0f));
					for (c = 0; c < 3; c++)
						a[c] += clip(g * dx[c]) * alpha;
				}
			}
		}
	}
	free(e);
	return 0;
}

static int compare_float(const void *a, const void *b)
{
	float x = *(const float *)a;
	float y = *(const float *)b;

	return (x > y) - (x < y);
}

/*
** The median, not the mean: a few strays far out would shrink the whole cloud
** to a dot in the middle of the window.
*/
void layout_normalize(float *pos, long n, float radius)
{
	float *v = malloc(sizeof(float) * (size_t)n);
	float centre[3];
	float spread;
	long i;
	int c;

	if (!v || n < 1) {
		free(v);
		return;
	}
	for (c = 0; c < 3; c++) {
		for (i = 0; i < n; i++)
			v[i] = pos[i * 3 + c];
		qsort(v, (size_t)n, sizeof(float), compare_float);
		centre[c] = v[n / 2];
	}
	for (i = 0; i < n; i++) {
		float r = 0.0f;

		for (c = 0; c < 3; c++) {
			pos[i * 3 + c] -= centre[c];
			r += pos[i * 3 + c] * pos[i * 3 + c];
		}
		v[i] = r;
	}
	qsort(v, (size_t)n, sizeof(float), compare_float);
	spread = sqrtf(v[n / 2]);
	free(v);
	if (spread <= 0.0f)
		return;
	for (i = 0; i < n * 3; i++)
		pos[i] *= radius / spread;
}
