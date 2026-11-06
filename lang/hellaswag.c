#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu.h"
#include "hellaswag.h"

#define ENDINGS		4
#define LINE_MAX_BYTES	16384
#define MAX_TOKENS	256	/* the context of the pretraining */
#define BATCH		32
#define BATCH_TOKENS	4096	/* the batch of the training: the cache fits */
#define T_STEP		32	/* window lengths: few, each one needs a cache */

typedef struct	sequence {
	int	*ids;
	int	n;
	int	first;		/* first token of the ending */
	int	bytes;		/* of the ending */
	int	question;
	int	ending;
}		sequence;

typedef struct	question {
	int	answer;
	double	sum[ENDINGS];	/* NLL of each ending */
	int	tokens[ENDINGS];
	int	bytes[ENDINGS];
}		question;

static int encode_pair(const lm *model, const char *context, const char *ending,
	sequence *s)
{
	static int ctx[LINE_MAX_BYTES];
	static int end[LINE_MAX_BYTES];
	static char spaced[LINE_MAX_BYTES + 1];
	int nc;
	int ne;
	int keep;

	/* The space belongs to the ending: this is how text is tokenized. */
	spaced[0] = ' ';
	strncpy(spaced + 1, ending, LINE_MAX_BYTES - 1);
	spaced[LINE_MAX_BYTES] = '\0';
	nc = lm_encode(model, context, ctx, LINE_MAX_BYTES);
	ne = lm_encode(model, spaced, end, LINE_MAX_BYTES);
	if (nc < 1 || ne < 1 || ne >= MAX_TOKENS)
		return -1;
	keep = nc;
	if (keep + ne > MAX_TOKENS)
		keep = MAX_TOKENS - ne;	/* the end of the context matters most */
	s->n = keep + ne;
	s->first = keep;
	s->bytes = (int)strlen(ending);
	s->ids = malloc(sizeof(int) * (size_t)s->n);
	if (!s->ids)
		return -1;
	memcpy(s->ids, ctx + nc - keep, sizeof(int) * (size_t)keep);
	memcpy(s->ids + keep, end, sizeof(int) * (size_t)ne);
	return 0;
}

static int split_line(char *line, char **field)
{
	int i;

	line[strcspn(line, "\r\n")] = '\0';
	for (i = 0; i < 2 + ENDINGS; i++) {
		field[i] = line;
		line = strchr(line, '\t');
		if (!line)
			return i == 1 + ENDINGS ? 0 : -1;
		*line++ = '\0';
	}
	return -1;
}

static int read_questions(const lm *model, const char *path, int limit,
	question **qs, sequence **seqs, int *count)
{
	static char line[LINE_MAX_BYTES];
	char *field[2 + ENDINGS];
	FILE *fp = fopen(path, "rb");
	int cap = 0;
	int n = 0;
	int e;

	*qs = NULL;
	*seqs = NULL;
	if (!fp)
		return -1;
	while ((limit <= 0 || n < limit) && fgets(line, sizeof(line), fp)) {
		if (n == cap) {
			question *q2;
			sequence *s2;

			cap = cap ? cap * 2 : 1024;
			q2 = realloc(*qs, sizeof(question) * (size_t)cap);
			if (q2)
				*qs = q2;
			s2 = realloc(*seqs, sizeof(sequence) * (size_t)cap * ENDINGS);
			if (s2)
				*seqs = s2;
			if (!q2 || !s2)
				goto fail;
		}
		if (split_line(line, field) != 0)
			goto fail;
		memset(&(*qs)[n], 0, sizeof(question));
		(*qs)[n].answer = atoi(field[0]);
		for (e = 0; e < ENDINGS; e++) {
			sequence *s = &(*seqs)[n * ENDINGS + e];

			memset(s, 0, sizeof(*s));
			if (encode_pair(model, field[1], field[2 + e], s) != 0)
				goto fail;
			s->question = n;
			s->ending = e;
			(*qs)[n].bytes[e] = s->bytes;
			(*qs)[n].tokens[e] = s->n - s->first;
		}
		n++;
	}
	fclose(fp);
	*count = n;
	return n > 0 ? 0 : -1;
fail:
	fclose(fp);
	return -1;
}

static int by_length(const void *a, const void *b)
{
	return ((const sequence *)a)->n - ((const sequence *)b)->n;
}

static void add_rows(question *qs, const sequence *s, const float *row)
{
	double sum = 0.0;
	int t;

	/* Row t holds the loss of token t + 1. */
	for (t = s->first - 1; t < s->n - 1; t++)
		sum += (double)row[t];
	qs[s->question].sum[s->ending] = sum;
}

/* All sequences of one window length at most, on the device. */
static int score_batches(lm *model, question *qs, sequence *seqs, int total)
{
	lm_cache *c = NULL;
	int *ids = NULL;
	float *rows = NULL;
	int cache_T = 0;
	int done = 0;
	int rc = -1;

	while (done < total) {
		int T = seqs[done + (total - done < BATCH ? total - done : BATCH) - 1].n;
		int B;
		int count;
		int b;

		T = (T + T_STEP - 1) / T_STEP * T_STEP;
		B = BATCH_TOKENS / T < BATCH ? BATCH_TOKENS / T : BATCH;
		count = total - done < B ? total - done : B;
		if (T != cache_T) {
			lm_cache_free(&c);
			free(ids);
			free(rows);
			c = lm_cache_create_packed(B, T);
			ids = malloc(sizeof(int) * (size_t)B * (size_t)T);
			rows = malloc(sizeof(float) * (size_t)B * (size_t)(T - 1));
			cache_T = T;
			if (!c || !ids || !rows)
				goto out;
		}
		/* Causal: what follows a sequence cannot change its losses. */
		memset(ids, 0, sizeof(int) * (size_t)B * (size_t)T);
		for (b = 0; b < B; b++) {
			const sequence *s = &seqs[done + (b < count ? b : 0)];

			memcpy(ids + (size_t)b * T, s->ids, sizeof(int) * (size_t)s->n);
		}
		if (lm_nll_rows_packed(model, c, ids, B, T, rows) != 0)
			goto out;
		for (b = 0; b < count; b++)
			add_rows(qs, &seqs[done + b], rows + (size_t)b * (T - 1));
		done += count;
		fprintf(stderr, "\rhellaswag %d/%d", done, total);
	}
	fprintf(stderr, "\n");
	rc = 0;
out:
	lm_cache_free(&c);
	free(ids);
	free(rows);
	return rc;
}

/* One sequence at a time on the CPU: slow, for small limits and for tests. */
static int score_host(lm *model, question *qs, sequence *seqs, int total)
{
	int V = model->vocab_size;
	int i;

	for (i = 0; i < total; i++) {
		sequence *s = &seqs[i];
		mat *hidden = mat_create(s->n, LM_D_MODEL);
		mat *logits = mat_create(s->n, V);
		float *rows = malloc(sizeof(float) * (size_t)s->n);
		int t;

		if (!hidden || !logits || !rows
		    || lm_forward(hidden, model, s->ids, s->n) != 0
		    || mat_mul(logits, hidden, model->w_head) != 0) {
			mat_free(&hidden);
			mat_free(&logits);
			free(rows);
			return -1;
		}
		for (t = 0; t < s->n - 1; t++) {
			const ann_real *row = logits->data + (size_t)t * (size_t)V;
			double top = -1e300;
			double sum = 0.0;
			int v;

			for (v = 0; v < V; v++) {
				double x = (double)row[v] + (double)model->b_head->data[v];

				if (x > top)
					top = x;
			}
			for (v = 0; v < V; v++)
				sum += exp((double)row[v]
				    + (double)model->b_head->data[v] - top);
			rows[t] = (float)(top + log(sum)
			    - (double)row[s->ids[t + 1]]
			    - (double)model->b_head->data[s->ids[t + 1]]);
		}
		add_rows(qs, s, rows);
		mat_free(&hidden);
		mat_free(&logits);
		free(rows);
		fprintf(stderr, "\rhellaswag %d/%d", i + 1, total);
	}
	fprintf(stderr, "\n");
	return 0;
}

static void tally(const question *qs, int n, hellaswag_score *score)
{
	double loss = 0.0;
	long tokens = 0;
	int right = 0;
	int right_norm = 0;
	int i;

	for (i = 0; i < n; i++) {
		const question *q = &qs[i];
		int best = 0;
		int best_norm = 0;
		int e;

		for (e = 1; e < ENDINGS; e++) {
			if (q->sum[e] < q->sum[best])
				best = e;
			if (q->sum[e] / q->bytes[e]
			    < q->sum[best_norm] / q->bytes[best_norm])
				best_norm = e;
		}
		right += best == q->answer;
		right_norm += best_norm == q->answer;
		loss += q->sum[q->answer];
		tokens += q->tokens[q->answer];
	}
	score->questions = n;
	score->acc = (double)right / (double)n;
	score->acc_norm = (double)right_norm / (double)n;
	score->nll_token = loss / (double)tokens;
}

int lm_hellaswag(lm *model, const char *path, int limit, hellaswag_score *score)
{
	question *qs;
	sequence *seqs;
	int n = 0;
	int i;
	int rc;

	if (!model || !path || !score)
		return -1;
	rc = read_questions(model, path, limit, &qs, &seqs, &n);
	if (rc == 0) {
		qsort(seqs, (size_t)n * ENDINGS, sizeof(sequence), by_length);
		if (gpu_available())
			rc = score_batches(model, qs, seqs, n * ENDINGS);
		else
			rc = score_host(model, qs, seqs, n * ENDINGS);
	}
	if (rc == 0)
		tally(qs, n, score);
	for (i = 0; seqs && i < n * ENDINGS; i++)
		free(seqs[i].ids);
	free(seqs);
	free(qs);
	return rc;
}
