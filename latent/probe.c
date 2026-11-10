/*
**	anni-probe: dump what a checkpoint holds for the latent space viewer.
**
**	anni-probe --model FILE --out DIR [--data FILE:LABEL]... [--positions N]
**
**	DIR/vocab.npy	float32 [V, D]: the token table
**	DIR/vocab.txt	one token per line, with \n \t \\ escaped
**	DIR/states.npy	float16 [L, N, D]: the residual stream, layer 0 being the
**			input of the first block and layer L-1 the last output
**	DIR/points.tsv	per position: source, window, position, token, next
**			token, predicted token, its probability, NLL of next
**
**	FILE holds raw int32 token ids. Runs on the CPU, reads the model only.
*/
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "lm.h"
#include "lm_serialize.h"
#include "npy.h"

#define WINDOW		256	/* the context of the pretraining */
#define MAX_SOURCES	8
#define TOKEN_TEXT	512

typedef struct	source {
	const char	*path;
	char		label[32];
	int		*ids;
	long		n;
} source;

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

static uint64_t rng_next(void)
{
	uint64_t x = rng_state;

	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	rng_state = x;
	return x;
}

static const char *arg_value(int argc, char **argv, const char *name,
	const char *fallback)
{
	int i;

	for (i = 1; i + 1 < argc; i++) {
		if (strcmp(argv[i], name) == 0)
			return argv[i + 1];
	}
	return fallback;
}

static void path_in(char *out, size_t size, const char *dir, const char *name)
{
	snprintf(out, size, "%s/%s", dir, name);
}

static void write_escaped(FILE *fp, const char *s)
{
	for (; *s; s++) {
		if (*s == '\n')
			fputs("\\n", fp);
		else if (*s == '\t')
			fputs("\\t", fp);
		else if (*s == '\\')
			fputs("\\\\", fp);
		else
			fputc(*s, fp);
	}
	fputc('\n', fp);
}

/* The head is D x V, a token being a column; the dump has a token per row. */
static int dump_vocab(const lm *model, const char *dir)
{
	char path[1024];
	char text[TOKEN_TEXT];
	const mat *w = model->w_head;
	long shape[2];
	float *row;
	FILE *fp;
	int v;
	int d;

	shape[0] = w->cols;
	shape[1] = w->rows;
	path_in(path, sizeof(path), dir, "vocab.npy");
	fp = npy_create(path, 0, 2, shape);
	row = malloc(sizeof(float) * (size_t)w->rows);
	if (!fp || !row)
		return -1;
	for (v = 0; v < w->cols; v++) {
		for (d = 0; d < w->rows; d++)
			row[d] = w->data[(size_t)d * (size_t)w->cols + (size_t)v];
		if (fwrite(row, sizeof(float), (size_t)w->rows, fp) != (size_t)w->rows)
			return -1;
	}
	free(row);
	if (fclose(fp) != 0)
		return -1;
	path_in(path, sizeof(path), dir, "vocab.txt");
	fp = fopen(path, "wb");
	if (!fp)
		return -1;
	for (v = 0; v < w->cols; v++) {
		int n = lm_decode(model, &v, 1, text, sizeof(text));

		text[n > 0 ? n : 0] = '\0';
		write_escaped(fp, text);
	}
	return fclose(fp);
}

static int load_source(source *s, const char *spec)
{
	const char *colon = strrchr(spec, ':');
	static char paths[MAX_SOURCES][512];
	static int used;
	size_t len = colon ? (size_t)(colon - spec) : strlen(spec);
	FILE *fp;

	if (len >= sizeof(paths[0]) || used == MAX_SOURCES)
		return -1;
	memcpy(paths[used], spec, len);
	paths[used][len] = '\0';
	s->path = paths[used++];
	snprintf(s->label, sizeof(s->label), "%s", colon ? colon + 1 : "text");
	fp = fopen(s->path, "rb");
	if (!fp)
		return -1;
	fseek(fp, 0, SEEK_END);
	s->n = ftell(fp) / (long)sizeof(int);
	rewind(fp);
	s->ids = malloc(sizeof(int) * (size_t)s->n);
	if (!s->ids || fread(s->ids, sizeof(int), (size_t)s->n, fp) != (size_t)s->n) {
		fclose(fp);
		return -1;
	}
	fclose(fp);
	return s->n > WINDOW + 1 ? 0 : -1;
}

static void write_states(FILE *fp, const mat *m, long layer, long n_points,
	long point, int rows)
{
	static uint16_t half[LM_D_MODEL];
	long off = (layer * n_points + point) * LM_D_MODEL * 2 + 128;
	int r;
	int d;

	fseek(fp, off, SEEK_SET);
	for (r = 0; r < rows; r++) {
		for (d = 0; d < LM_D_MODEL; d++)
			half[d] = f32_to_f16(m->data[(size_t)r * LM_D_MODEL + (size_t)d]);
		fwrite(half, 2, LM_D_MODEL, fp);
	}
}

/* What the model expects after position t, into the points file. */
static void write_point(FILE *fp, const lm *model, const lm_cache *c, ann_real *row,
	const source *s, int window, int t, int next)
{
	double sum = 0.0;
	float top;
	int best = 0;
	int v;

	for (v = 0; v < model->vocab_size; v++)
		row[v] += model->b_head->data[v];
	top = row[0];
	for (v = 1; v < model->vocab_size; v++) {
		if (row[v] > top) {
			top = row[v];
			best = v;
		}
	}
	for (v = 0; v < model->vocab_size; v++)
		sum += exp((double)row[v] - top);
	fprintf(fp, "%s\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\n", s->label, window, t,
	    c->ids[t], next, best, 1.0 / sum, log(sum) - ((double)row[next] - top));
}

static int dump_states(lm *model, source *src, int n_src, long positions,
	const char *dir)
{
	long windows = (positions + WINDOW - 1) / WINDOW;
	long n_points = windows * WINDOW;
	char path[1024];
	int ids[WINDOW + 1];
	long shape[3];
	lm_cache *c;
	mat *logits;
	FILE *states;
	FILE *points;
	long w;

	shape[0] = LM_N_BLOCKS + 1;
	shape[1] = n_points;
	shape[2] = LM_D_MODEL;
	path_in(path, sizeof(path), dir, "states.npy");
	states = npy_create(path, 1, 3, shape);
	path_in(path, sizeof(path), dir, "points.tsv");
	points = fopen(path, "wb");
	c = lm_cache_create(WINDOW);
	logits = mat_create(WINDOW, model->vocab_size);
	if (!states || !points || !c || !logits)
		return -1;
	if (ftell(states) != 128) {
		fprintf(stderr, "anni-probe: unexpected npy header size\n");
		return -1;
	}
	for (w = 0; w < windows; w++) {
		const source *s = &src[w % n_src];
		long at = (long)(rng_next() % (uint64_t)(s->n - WINDOW - 1));
		int i;

		memcpy(ids, s->ids + at, sizeof(ids));
		if (lm_forward_cache(model, ids, WINDOW, c) != 0)
			return -1;
		for (i = 0; i < LM_N_BLOCKS; i++)
			write_states(states, c->trc->bc[i].x_in, i, n_points,
			    w * WINDOW, WINDOW);
		write_states(states, c->hidden, LM_N_BLOCKS, n_points, w * WINDOW,
		    WINDOW);
		if (mat_mul(logits, c->hidden, model->w_head) != 0)
			return -1;
		for (i = 0; i < WINDOW; i++)
			write_point(points, model, c, logits->data
			    + (size_t)i * (size_t)model->vocab_size, s, (int)w, i, ids[i + 1]);
		fprintf(stderr, "\rwindow %ld/%ld", w + 1, windows);
	}
	fprintf(stderr, "\n");
	mat_free(&logits);
	lm_cache_free(&c);
	fclose(points);
	return fclose(states);
}

int main(int argc, char **argv)
{
	const char *path = arg_value(argc, argv, "--model", NULL);
	const char *dir = arg_value(argc, argv, "--out", NULL);
	long positions = atol(arg_value(argc, argv, "--positions", "20000"));
	source src[MAX_SOURCES];
	int n_src = 0;
	lm *model;
	int i;

	if (!path || !dir) {
		fprintf(stderr, "usage: anni-probe --model FILE --out DIR "
		    "[--data FILE:LABEL]... [--positions N]\n");
		return 2;
	}
	memset(src, 0, sizeof(src));
	for (i = 1; i + 1 < argc; i++) {
		if (strcmp(argv[i], "--data") != 0)
			continue;
		if (n_src == MAX_SOURCES || load_source(&src[n_src++], argv[i + 1]) != 0) {
			fprintf(stderr, "anni-probe: cannot read %s\n", argv[i + 1]);
			return 1;
		}
	}
	if (mkdir(dir, 0777) != 0 && errno != EEXIST) {
		fprintf(stderr, "anni-probe: cannot create %s\n", dir);
		return 1;
	}
	model = lm_load_any(path);
	if (!model) {
		fprintf(stderr, "anni-probe: cannot load %s\n", path);
		return 1;
	}
	if (dump_vocab(model, dir) != 0
	    || (n_src && dump_states(model, src, n_src, positions, dir) != 0)) {
		fprintf(stderr, "anni-probe: dump failed\n");
		return 1;
	}
	lm_free(&model);
	return 0;
}
