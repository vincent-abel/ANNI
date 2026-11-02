/*
**	Direction-ablation edits of an LM's FFN output weights, without
**	training (GPU).
**
**	  scripts/lm_edit MODEL SPEC...
**
**	Build after make cuda (the objects without main.o):
**	  gcc -O2 -fopenmp -march=native -std=gnu99 -I. -Inet -Inum -Ilang \
**	      -Icli -Icompat -c scripts/lm_edit.c -o objects/lm_edit.x
**	  g++ -o scripts/lm_edit objects/lm_edit.x $(ls objects/*.o | grep -v \
**	      /main.o) -fopenmp -lm -lncurses -L$CUDA_HOME/lib64 -lcudart \
**	      -lcublas -lcublasLt
**
**	The FFN of each block writes o = FFN(ln2(x_mid)) into the residual. On
**	training windows of code and of web text, per block: the mean of o
**	per class and its second moment. Three unit directions per block:
**	  mean  m_code - m_web
**	  web   top eigenvector of C_web - C_code (written more on web)
**	  code  top eigenvector of C_code - C_web (written more on code)
**	A SPEC is one round: edits joined by '+', each L:DIR:K:BETA. It takes
**	the K rows w_i of block L's W2 (D_FF x D) with the largest |w_i . u|
**	(K = 0: all rows, and b2 too) and sets w_i -= BETA (w_i . u) u. BETA 1
**	projects u out; BETA < 0 writes more of it. Each round is scored by the
**	mean NLL of held-out code and web windows, then undone.
**
**	With LM_EDIT_PAIRS (scripts/edit_pairs.js), the two classes are reply
**	behaviours instead: after each taskbench prompt, an "edit" reply in the
**	format the bench applies and the model's own "talk" reply; o is taken
**	at the positions that predict the first TGT reply tokens. The
**	directions are mean (m_edit - m_talk), talk and edit, from the even
**	tasks; a round is scored by the NLL of both replies on the odd tasks
**	(talk - edit is the margin for the edit format), and by the code and
**	web windows, which show the damage.
**
**	Environment: LM_EDIT_T (window, default 2048), LM_EDIT_B (windows per
**	forward, default 4; 2 at 2048), LM_EDIT_FP32 (1: FP32 GEMMs instead of
**	BF16), LM_EDIT_WIN (training windows per class, default 24),
**	LM_EDIT_EVAL (held-out windows per class, default 24), LM_EDIT_DIRS
**	(direction cache file: read if it exists, else written), LM_EDIT_SAVE
**	(save the model with the last SPEC applied to this path).
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "block.h"
#include "ffn.h"
#include "gpu.h"
#include "lm.h"
#include "lm_serialize.h"

static int B;	/* windows per packed forward (LM_EDIT_B, default 4) */
#define STRIDE	4	/* every STRIDE-th position feeds the second moments */
#define TRAIN	"data/mix_train_ids.bin"
#define VAL_CODE "data/mix_val_code_ids.bin"
#define VAL_WEB	"data/mix_val_web_ids.bin"
#define CODE_IDS 603362012L	/* code ids at the end of TRAIN (build_mix.log) */
#define WEB_OFF	2000000000L
#define D	LM_D_MODEL
#define NDIR	3

#define TGT	64	/* reply tokens scored per pair */
#define MAX_PAIRS 64

static const char *text_dirs[NDIR] = { "mean", "web", "code" };
static const char *pair_dirs[NDIR] = { "mean", "talk", "edit" };
static const char **dir_names = text_dirs;
static int g_T;

static int env_int(const char *name, int def)
{
	const char *v = getenv(name);

	return v && atoi(v) > 0 ? atoi(v) : def;
}

/* n windows of g_T ids, spread evenly over [off, off + span) of an int32 file. */
static int *read_windows(const char *path, long off, long span, int n)
{
	FILE *f;
	int *ids;
	long step;
	int i;

	ids = malloc((size_t)n * g_T * sizeof(int));
	f = fopen(path, "rb");
	if (!ids || !f) {
		fprintf(stderr, "lm_edit: cannot read %s\n", path);
		exit(1);
	}
	if (span <= 0) {
		fseek(f, 0, SEEK_END);
		span = ftell(f) / 4 - off;
	}
	step = (span - g_T) / n;
	for (i = 0; i < n; i++)
		if (fseek(f, (off + i * step) * 4, SEEK_SET) != 0
		    || fread(ids + (size_t)i * g_T, sizeof(int), g_T, f)
		    != (size_t)g_T) {
			fprintf(stderr, "lm_edit: short read in %s\n", path);
			exit(1);
		}
	fclose(f);
	return ids;
}

static const ann_real *host_rows(mat *m)
{
	if (mat_pull_host(m) != 0 || !m->data) {
		fprintf(stderr, "lm_edit: cannot read activations\n");
		exit(1);
	}
	return m->data;
}

/*
** Per block l: mean[l][D] and the upper triangle of mom[l][D][D] of the FFN
** output over n windows (n a multiple of B).
*/
static void ffn_stats(lm *m, lm_cache *c, const int *ids, int n,
	double *mean, double *mom)
{
	int nb = m->tr->n_blocks;
	double cnt = 0.0;
	double rows_seen = 0.0;
	float *o;
	int nr;
	int w;
	int l;
	int r;
	int i;

	nr = B * g_T / STRIDE;
	o = malloc((size_t)nr * D * sizeof(float));
	if (!o)
		exit(1);
	memset(mean, 0, (size_t)nb * D * sizeof(double));
	memset(mom, 0, (size_t)nb * D * D * sizeof(double));
	for (w = 0; w < n; w += B) {
		if (lm_forward_cache_packed(m, ids + (size_t)w * g_T, B, g_T, c)
		    != 0) {
			fprintf(stderr, "lm_edit: forward failed: %s\n", gpu_debug_last_error());
			exit(1);
		}
		for (l = 0; l < nb; l++) {
			mat *nx = l + 1 < nb ? c->trc->bc[l + 1].x_in : c->hidden;
			const ann_real *y = host_rows(nx);
			const ann_real *x = host_rows(c->trc->bc[l].x_mid);
			double *ml = mean + (size_t)l * D;
			double *sl = mom + (size_t)l * D * D;

			for (r = 0; r < B * g_T; r++)
				for (i = 0; i < D; i++)
					ml[i] += y[(size_t)r * D + i]
					    - x[(size_t)r * D + i];
			for (r = 0; r < nr; r++)
				for (i = 0; i < D; i++)
					o[(size_t)r * D + i] =
					    y[(size_t)r * STRIDE * D + i]
					    - x[(size_t)r * STRIDE * D + i];
#pragma omp parallel for schedule(dynamic, 8)
			for (i = 0; i < D; i++) {
				double *si = sl + (size_t)i * D;
				int rr;
				int j;

				for (rr = 0; rr < nr; rr++) {
					const float *orow = o + (size_t)rr * D;
					double a = orow[i];

					for (j = i; j < D; j++)
						si[j] += a * orow[j];
				}
			}
		}
		cnt += (double)B * g_T;
		rows_seen += nr;
	}
	for (i = 0; i < nb * D; i++)
		mean[i] /= cnt;
	for (l = 0; l < nb; l++)
		for (i = 0; i < D; i++) {
			double *si = mom + ((size_t)l * D + i) * D;
			int j;

			for (j = i; j < D; j++)
				si[j] /= rows_seen;
		}
	free(o);
}

static void normalize(double *v)
{
	double n2 = 0.0;
	int i;

	for (i = 0; i < D; i++)
		n2 += v[i] * v[i];
	n2 = sqrt(n2);
	for (i = 0; i < D; i++)
		v[i] /= n2 > 0.0 ? n2 : 1.0;
}

/*
** Top eigenvector of the symmetric sgn * (P - Q), only the upper triangles
** stored; power iteration on the shifted matrix, eigenvalue in *lambda.
*/
static void top_eig(const double *p, const double *q, double sgn, double *v,
	double *lambda)
{
	double *a;
	double *t;
	double fro = 0.0;
	double l = 0.0;
	int it;
	int i;
	int j;

	a = malloc((size_t)D * D * sizeof(double));
	t = malloc((size_t)D * sizeof(double));
	if (!a || !t)
		exit(1);
	for (i = 0; i < D; i++)
		for (j = i; j < D; j++) {
			double x = sgn * (p[(size_t)i * D + j] - q[(size_t)i * D + j]);

			a[(size_t)i * D + j] = a[(size_t)j * D + i] = x;
			fro += (i == j ? 1.0 : 2.0) * x * x;
		}
	fro = sqrt(fro);
	for (i = 0; i < D; i++)
		v[i] = 1.0 + 0.001 * i;
	normalize(v);
	for (it = 0; it < 1000; it++) {
#pragma omp parallel for
		for (i = 0; i < D; i++) {
			double s = fro * v[i];
			int k;

			for (k = 0; k < D; k++)
				s += a[(size_t)i * D + k] * v[k];
			t[i] = s;
		}
		memcpy(v, t, (size_t)D * sizeof(double));
		normalize(v);
	}
	for (i = 0; i < D; i++) {
		double s = 0.0;

		for (j = 0; j < D; j++)
			s += a[(size_t)i * D + j] * v[j];
		l += v[i] * s;
	}
	*lambda = l;
	free(a);
	free(t);
}

static double mean_nll(lm *m, lm_cache *c, const int *ids, int n)
{
	double wn[64];
	double s = 0.0;
	int w;
	int j;

	for (w = 0; w < n; w += B) {
		if (lm_nll_packed(m, c, ids + (size_t)w * g_T, B, g_T, wn) != 0) {
			fprintf(stderr, "lm_edit: packed NLL failed: %s\n", gpu_debug_last_error());
			exit(1);
		}
		for (j = 0; j < B; j++)
			s += wn[j];
	}
	return s / n;
}

static int dir_index(const char *s)
{
	int i;

	for (i = 0; i < NDIR; i++)
		if (strcmp(s, dir_names[i]) == 0)
			return i;
	return -1;
}

typedef struct	row_dot {
	double	dot;
	int	row;
} row_dot;

static int by_abs_dot(const void *a, const void *b)
{
	double x = fabs(((const row_dot *)a)->dot);
	double y = fabs(((const row_dot *)b)->dot);

	return x < y ? 1 : x > y ? -1 : 0;
}

/* w_i -= beta (w_i . u) u on the k rows of block l's W2 most aligned with u. */
static void edit_block(lm *m, int l, const double *u, int k, double beta)
{
	ffn *f = m->tr->blocks[l].ffn;
	row_dot *rd;
	int i;
	int j;

	rd = malloc((size_t)LM_D_FF * sizeof(row_dot));
	if (!rd)
		exit(1);
	for (i = 0; i < LM_D_FF; i++) {
		double s = 0.0;

		for (j = 0; j < D; j++)
			s += f->w2->data[(size_t)i * D + j] * u[j];
		rd[i].dot = s;
		rd[i].row = i;
	}
	if (k <= 0 || k > LM_D_FF)
		k = LM_D_FF;
	else
		qsort(rd, LM_D_FF, sizeof(row_dot), by_abs_dot);
	for (i = 0; i < k; i++) {
		ann_real *w = f->w2->data + (size_t)rd[i].row * D;

		for (j = 0; j < D; j++)
			w[j] -= (ann_real)(beta * rd[i].dot * u[j]);
	}
	if (k == LM_D_FF) {
		double s = 0.0;

		for (j = 0; j < D; j++)
			s += f->b2->data[j] * u[j];
		for (j = 0; j < D; j++)
			f->b2->data[j] -= (ann_real)(beta * s * u[j]);
	}
	free(rd);
}

/* Push every block's W2 and b2 to the device and drop stale half copies. */
static void sync_ffn_out(lm *m)
{
	int l;

	for (l = 0; l < m->tr->n_blocks; l++)
		if (mat_sync_device(m->tr->blocks[l].ffn->w2) != 0) {
			fprintf(stderr, "lm_edit: weight upload failed\n");
			exit(1);
		}
	gpu_half_wt_invalidate(NULL);
}

/* Apply a SPEC; 0 on success, -1 if it does not parse. */
static int apply_spec(lm *m, const double *dirs, const char *spec)
{
	char buf[1024];
	char *save = NULL;
	char *tok;
	int nb = m->tr->n_blocks;

	snprintf(buf, sizeof(buf), "%s", spec);
	for (tok = strtok_r(buf, "+", &save); tok;
	    tok = strtok_r(NULL, "+", &save)) {
		char dn[16];
		int l;
		int k;
		int di;
		double beta;

		if (sscanf(tok, "%d:%15[a-z]:%d:%lf", &l, dn, &k, &beta) != 4
		    || l < 0 || l >= nb || (di = dir_index(dn)) < 0)
			return -1;
		edit_block(m, l, dirs + ((size_t)l * NDIR + di) * D, k, beta);
	}
	return 0;
}


/* One prompt + reply: g_T ids padded with 0, reply tokens [lo, hi). */
typedef struct	pair_seq {
	int	*ids;
	int	lo;
	int	hi;
} pair_seq;

static int pair_seq_make(const lm *m, const char *prompt, const char *reply,
	pair_seq *s)
{
	size_t lp = strlen(prompt);
	size_t lr = strlen(reply);
	char *text;
	int *pids;
	int np;
	int n;
	int i;

	text = malloc(lp + lr + 1);
	pids = malloc((size_t)LM_MAX_CTX * sizeof(int));
	s->ids = calloc((size_t)LM_MAX_CTX, sizeof(int));
	if (!text || !pids || !s->ids)
		exit(1);
	memcpy(text, prompt, lp);
	memcpy(text + lp, reply, lr + 1);
	np = lm_encode(m, prompt, pids, LM_MAX_CTX);
	n = lm_encode(m, text, s->ids, LM_MAX_CTX);
	free(text);
	if (np <= 0 || n <= np) {
		free(pids);
		return -1;
	}
	/* The reply starts at the first token the prompt alone does not have. */
	for (i = 0; i < np && s->ids[i] == pids[i]; i++)
		;
	free(pids);
	s->lo = i;
	s->hi = n < i + TGT ? n : i + TGT;
	if (s->hi > g_T)
		s->hi = g_T;
	for (i = s->hi; i < LM_MAX_CTX; i++)
		s->ids[i] = 0;
	return s->lo >= 1 && s->hi > s->lo ? 0 : -1;
}

/* LM_EDIT_PAIRS records: name \x1f prompt \x1f edit \x1f talk \x1e. */
static int load_pairs(const lm *m, const char *path, pair_seq *edit,
	pair_seq *talk, int max)
{
	FILE *f = fopen(path, "rb");
	char *buf;
	char *rec;
	long len;
	int n = 0;

	if (!f) {
		fprintf(stderr, "lm_edit: cannot read %s\n", path);
		exit(1);
	}
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	buf = malloc((size_t)len + 1);
	if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len)
		exit(1);
	buf[len] = '\0';
	fclose(f);
	rec = buf;
	while (*rec && n < max) {
		char *end = strchr(rec, '\x1e');
		char *fld[4];
		int k;

		if (end)
			*end = '\0';
		fld[0] = rec;
		for (k = 1; k < 4; k++) {
			fld[k] = fld[k - 1] ? strchr(fld[k - 1], '\x1f') : NULL;
			if (fld[k])
				*fld[k]++ = '\0';
		}
		if (fld[3] && pair_seq_make(m, fld[1], fld[2], &edit[n]) == 0
		    && pair_seq_make(m, fld[1], fld[3], &talk[n]) == 0)
			n++;
		else
			fprintf(stderr, "lm_edit: skipped pair %s\n", fld[0]);
		if (!end)
			break;
		rec = end + 1;
	}
	free(buf);
	return n;
}

/*
** Packed forward of seqs[0..n) (every step-th from first), B at a time; the
** last batch repeats its final sequence. fn sees each real sequence.
*/
typedef void	(*seq_fn)(lm *m, lm_cache *c, int b, const pair_seq *s,
		    void *arg);

static void seq_each(lm *m, lm_cache *c, const pair_seq *seqs, int n,
	int first, int step, seq_fn fn, void *arg)
{
	int *ids;
	int idx[64];
	int k;
	int b;

	ids = malloc((size_t)B * g_T * sizeof(int));
	if (!ids)
		exit(1);
	for (k = first; k < n; ) {
		int nb = 0;

		while (nb < B && k < n) {
			idx[nb++] = k;
			k += step;
		}
		for (b = 0; b < B; b++)
			memcpy(ids + (size_t)b * g_T, seqs[idx[b < nb ? b : nb - 1]].ids,
			    (size_t)g_T * sizeof(int));
		if (lm_forward_cache_packed(m, ids, B, g_T, c) != 0) {
			fprintf(stderr, "lm_edit: forward failed: %s\n",
			    gpu_debug_last_error());
			exit(1);
		}
		for (b = 0; b < nb; b++)
			fn(m, c, b, &seqs[idx[b]], arg);
	}
	free(ids);
}

typedef struct	stats_acc {
	double	*mean;
	double	*mom;
	double	cnt;
} stats_acc;

/* Adds o at the positions that predict the reply (t = lo-1 .. hi-2). */
static void stats_fn(lm *m, lm_cache *c, int b, const pair_seq *s, void *arg)
{
	stats_acc *st = arg;
	int nb = m->tr->n_blocks;
	int nr = s->hi - s->lo;
	float *o;
	int l;

	o = malloc((size_t)nr * D * sizeof(float));
	if (!o)
		exit(1);
	for (l = 0; l < nb; l++) {
		mat *nx = l + 1 < nb ? c->trc->bc[l + 1].x_in : c->hidden;
		const ann_real *y = host_rows(nx);
		const ann_real *x = host_rows(c->trc->bc[l].x_mid);
		double *ml = st->mean + (size_t)l * D;
		double *sl = st->mom + (size_t)l * D * D;
		size_t r0 = (size_t)b * g_T + s->lo - 1;
		int r;
		int i;

		for (r = 0; r < nr; r++)
			for (i = 0; i < D; i++) {
				o[(size_t)r * D + i] = y[(r0 + r) * D + i]
				    - x[(r0 + r) * D + i];
				ml[i] += o[(size_t)r * D + i];
			}
#pragma omp parallel for schedule(dynamic, 8)
		for (i = 0; i < D; i++) {
			double *si = sl + (size_t)i * D;
			int rr;
			int j;

			for (rr = 0; rr < nr; rr++) {
				const float *orow = o + (size_t)rr * D;
				double a = orow[i];

				for (j = i; j < D; j++)
					si[j] += a * orow[j];
			}
		}
	}
	st->cnt += nr;
	free(o);
}

static void pair_stats(lm *m, lm_cache *c, const pair_seq *seqs, int n,
	double *mean, double *mom)
{
	int nb = m->tr->n_blocks;
	stats_acc st;
	int i;

	st.mean = mean;
	st.mom = mom;
	st.cnt = 0.0;
	memset(mean, 0, (size_t)nb * D * sizeof(double));
	memset(mom, 0, (size_t)nb * D * D * sizeof(double));
	seq_each(m, c, seqs, n, 0, 2, stats_fn, &st);	/* even tasks */
	for (i = 0; i < nb * D; i++)
		mean[i] /= st.cnt;
	for (i = 0; i < nb * D; i++) {
		double *si = mom + (size_t)i * D;
		int j;

		for (j = i % D; j < D; j++)
			si[j] /= st.cnt;
	}
}

typedef struct	nll_acc {
	double	sum;
	long	cnt;
} nll_acc;

/* NLL of the reply tokens, with the head on the host (rows are few). */
static void nll_fn(lm *m, lm_cache *c, int b, const pair_seq *s, void *arg)
{
	nll_acc *acc = arg;
	const ann_real *h = host_rows(c->hidden);
	const ann_real *w = m->w_head->data;
	const ann_real *bh = m->b_head->data;
	int V = m->vocab_size;
	double sum = 0.0;
	int t;

#pragma omp parallel for reduction(+:sum) schedule(dynamic, 1)
	for (t = s->lo - 1; t < s->hi - 1; t++) {
		const ann_real *ht = h + ((size_t)b * g_T + t) * D;
		float *lg = malloc((size_t)V * sizeof(float));
		double mx;
		double z = 0.0;
		int d;
		int v;

		if (!lg)
			exit(1);
		for (v = 0; v < V; v++)
			lg[v] = bh[v];
		for (d = 0; d < D; d++) {
			const ann_real *wd = w + (size_t)d * V;
			float hd = ht[d];

			for (v = 0; v < V; v++)
				lg[v] += hd * wd[v];
		}
		mx = lg[0];
		for (v = 1; v < V; v++)
			if (lg[v] > mx)
				mx = lg[v];
		for (v = 0; v < V; v++)
			z += exp(lg[v] - mx);
		sum += mx + log(z) - lg[s->ids[t + 1]];
		free(lg);
	}
	acc->sum += sum;
	acc->cnt += s->hi - s->lo;
}

/* Mean NLL per reply token over the tasks first, first + 2, ... */
static double pair_nll(lm *m, lm_cache *c, const pair_seq *seqs, int n,
	int first)
{
	nll_acc acc = { 0.0, 0 };

	seq_each(m, c, seqs, n, first, 2, nll_fn, &acc);
	return acc.sum / (double)acc.cnt;
}

/* The three directions of every block from the two classes' statistics. */
static void make_dirs(int nb, const double *ma, const double *mb,
	const double *sa, const double *sb, double *dirs)
{
	int l;

	for (l = 0; l < nb; l++) {
		double *u = dirs + (size_t)l * NDIR * D;
		double lb;
		double la;
		double dn = 0.0;
		int i;

		for (i = 0; i < D; i++) {
			u[i] = ma[(size_t)l * D + i] - mb[(size_t)l * D + i];
			dn += u[i] * u[i];
		}
		normalize(u);
		top_eig(sb + (size_t)l * D * D, sa + (size_t)l * D * D, 1.0, u + D,
		    &lb);
		top_eig(sb + (size_t)l * D * D, sa + (size_t)l * D * D, -1.0,
		    u + 2 * D, &la);
		printf("block %2d: |m_%s - m_%s| %.3f  %s eig %.3f  %s eig %.3f\n",
		    l, dir_names[2], dir_names[1], sqrt(dn), dir_names[1], lb,
		    dir_names[2], la);
	}
}

int main(int argc, char **argv)
{
	int nwin = env_int("LM_EDIT_WIN", 24);
	int neval = env_int("LM_EDIT_EVAL", 24);
	const char *dpath = getenv("LM_EDIT_DIRS");
	const char *spath = getenv("LM_EDIT_SAVE");
	const char *ppath = getenv("LM_EDIT_PAIRS");
	static pair_seq edit[MAX_PAIRS];
	static pair_seq talk[MAX_PAIRS];
	double e0 = 0.0;
	double t0 = 0.0;
	double r0 = 0.0;
	int np = 0;
	ann_real **keep_w;
	ann_real **keep_b;
	double *dirs;
	lm_cache *c;
	lm_cache *cv;
	lm *m;
	int *ec;
	int *ew;
	double c0;
	double w0;
	FILE *f;
	int nb;
	int l;
	int a;

	g_T = env_int("LM_EDIT_T", 2048);
	B = env_int("LM_EDIT_B", 4);
	if (B < 2 || B > 64)
		B = 4;
	nwin = (nwin + B - 1) / B * B;
	neval = (neval + B - 1) / B * B;
	if (argc < 2) {
		fprintf(stderr, "usage: lm_edit MODEL [L:DIR:K:BETA[+...]]...\n");
		return 2;
	}
	if (!gpu_available()) {
		fprintf(stderr, "lm_edit: no GPU\n");
		return 1;
	}
	m = lm_load_any(argv[1]);
	if (!m)
		return 1;
	nb = m->tr->n_blocks;
	lm_weights_to_device(m);
	attn_set_flash(1);	/* no per-sequence T x T scores at 2048 */
	if (env_int("LM_EDIT_FP32", 0) == 0)
		gpu_set_compute_prec(GPU_PREC_BF16);	/* as trained; FP32 attention is slow */
	c = lm_cache_create_packed(B, g_T);
	/* The packed NLL wants a cache whose rows were never read on the host. */
	cv = lm_cache_create_packed(B, g_T);
	dirs = malloc((size_t)nb * NDIR * D * sizeof(double));
	if (!c || !cv || !dirs)
		return 1;
	if (ppath) {
		dir_names = pair_dirs;
		np = load_pairs(m, ppath, edit, talk, MAX_PAIRS);
		printf("pairs: %d tasks from %s (directions: even, scores: odd)\n",
		    np, ppath);
		if (np < 2)
			return 1;
	}
	f = dpath ? fopen(dpath, "rb") : NULL;
	if (f) {
		if (fread(dirs, sizeof(double), (size_t)nb * NDIR * D, f)
		    != (size_t)nb * NDIR * D)
			return 1;
		fclose(f);
		printf("directions: %s\n", dpath);
	} else {
		double *ma = malloc((size_t)nb * D * sizeof(double));
		double *mb = malloc((size_t)nb * D * sizeof(double));
		double *sa = malloc((size_t)nb * D * D * sizeof(double));
		double *sb = malloc((size_t)nb * D * D * sizeof(double));

		if (!ma || !mb || !sa || !sb)
			return 1;
		if (np > 0) {
			pair_stats(m, c, edit, np, ma, sa);
			pair_stats(m, c, talk, np, mb, sb);
		} else {
			int *ids;
			long n;

			f = fopen(TRAIN, "rb");
			if (!f)
				return 1;
			fseek(f, 0, SEEK_END);
			n = ftell(f) / 4;
			fclose(f);
			ids = read_windows(TRAIN, n - CODE_IDS, CODE_IDS, nwin);
			ffn_stats(m, c, ids, nwin, ma, sa);
			free(ids);
			ids = read_windows(TRAIN, WEB_OFF, 400000000L, nwin);
			ffn_stats(m, c, ids, nwin, mb, sb);
			free(ids);
		}
		make_dirs(nb, ma, mb, sa, sb, dirs);
		free(ma);
		free(mb);
		free(sa);
		free(sb);
		if (dpath && (f = fopen(dpath, "wb"))) {
			fwrite(dirs, sizeof(double), (size_t)nb * NDIR * D, f);
			fclose(f);
		}
	}
	ec = read_windows(VAL_CODE, 0, 0, neval);
	ew = read_windows(VAL_WEB, 0, 0, neval);
	c0 = mean_nll(m, cv, ec, neval);
	w0 = mean_nll(m, cv, ew, neval);
	printf("unedited: code NLL %.4f  web NLL %.4f  (%d + %d windows of %d)\n",
	    c0, w0, neval, neval, g_T);
	if (np > 0) {
		e0 = pair_nll(m, c, edit, np, 1);
		t0 = pair_nll(m, c, talk, np, 1);
		r0 = pair_nll(m, c, edit, np, 0);
		printf("unedited: edit NLL %.4f  talk NLL %.4f  margin %.4f"
		    "  (even tasks: edit NLL %.4f)\n", e0, t0, t0 - e0, r0);
	}
	fflush(stdout);
	keep_w = malloc((size_t)nb * sizeof(*keep_w));
	keep_b = malloc((size_t)nb * sizeof(*keep_b));
	if (!keep_w || !keep_b)
		return 1;
	for (l = 0; l < nb; l++) {
		ffn *fl = m->tr->blocks[l].ffn;

		mat_pull_host(fl->w2);
		keep_w[l] = malloc((size_t)LM_D_FF * D * sizeof(ann_real));
		keep_b[l] = malloc((size_t)D * sizeof(ann_real));
		if (!keep_w[l] || !keep_b[l])
			return 1;
		memcpy(keep_w[l], fl->w2->data, (size_t)LM_D_FF * D * sizeof(ann_real));
		memcpy(keep_b[l], fl->b2->data, (size_t)D * sizeof(ann_real));
	}
	for (a = 2; a < argc; a++) {
		double c1;
		double w1;

		if (apply_spec(m, dirs, argv[a]) != 0) {
			fprintf(stderr, "lm_edit: bad spec %s\n", argv[a]);
			return 2;
		}
		sync_ffn_out(m);
		c1 = mean_nll(m, cv, ec, neval);
		w1 = mean_nll(m, cv, ew, neval);
		if (np > 0) {
			double e1 = pair_nll(m, c, edit, np, 1);
			double t1 = pair_nll(m, c, talk, np, 1);
			double r1 = pair_nll(m, c, edit, np, 0);

			printf("round %2d %-32s edit %.4f (%+.4f)  margin %.4f"
			    " (%+.4f)  even %+.4f  code %+.4f  web %+.4f\n",
			    a - 1, argv[a], e1, e1 - e0, t1 - e1,
			    (t1 - e1) - (t0 - e0), r1 - r0, c1 - c0, w1 - w0);
		} else
			printf("round %2d %-40s code %.4f (%+.4f)  web %.4f (%+.4f)\n",
			    a - 1, argv[a], c1, c1 - c0, w1, w1 - w0);
		fflush(stdout);
		if (a + 1 == argc && spath) {
			if (lm_save(m, spath) != 0) {
				fprintf(stderr, "lm_edit: cannot save %s\n", spath);
				return 1;
			}
			printf("saved %s\n", spath);
			break;
		}
		for (l = 0; l < nb; l++) {
			ffn *fl = m->tr->blocks[l].ffn;

			memcpy(fl->w2->data, keep_w[l],
			    (size_t)LM_D_FF * D * sizeof(ann_real));
			memcpy(fl->b2->data, keep_b[l], (size_t)D * sizeof(ann_real));
		}
		sync_ffn_out(m);
	}
	lm_cache_free(&c);
	lm_cache_free(&cv);
	lm_free(&m);
	return 0;
}
