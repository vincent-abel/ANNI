/*
**	Autoregressive decode (greedy, temperature, top-k / top-p)
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "generate.h"
#include "kv.h"
#include "ops.h"
#include "tokenize.h"
#include "vocab.h"

typedef struct	samp_pair {
	int	id;
	ann_real	p;
} samp_pair;

static int samp_cmp_desc(const void *a, const void *b)
{
	const samp_pair *pa = a;
	const samp_pair *pb = b;

	if (pa->p < pb->p)
		return 1;
	if (pa->p > pb->p)
		return -1;
	return 0;
}

int lm_sample_id(const vec *logits, double temperature, int top_k, double top_p)
{
	int i;
	int best;
	int n;
	int keep;
	double best_v;
	double sum;
	double u;
	double cum;
	vec scaled;
	vec probs;
	ann_real *buf;
	ann_real *pbuf;
	samp_pair *pairs;

	if (!logits || !logits->data || logits->n <= 0)
		return VOCAB_EOS;
	n = logits->n;

	if (temperature <= 0.0) {
		best = 0;
		best_v = (double)logits->data[0];
		for (i = 1; i < n; i++) {
			if ((double)logits->data[i] > best_v) {
				best_v = (double)logits->data[i];
				best = i;
			}
		}
		return best;
	}

	buf = malloc((size_t)n * sizeof(ann_real));
	pbuf = malloc((size_t)n * sizeof(ann_real));
	if (!buf || !pbuf) {
		free(buf);
		free(pbuf);
		return VOCAB_EOS;
	}
	for (i = 0; i < n; i++)
		buf[i] = (ann_real)((double)logits->data[i] / temperature);
	scaled.data = buf;
	scaled.n = n;
	probs.data = pbuf;
	probs.n = n;
	if (vec_softmax(&probs, &scaled) != 0) {
		free(buf);
		free(pbuf);
		return VOCAB_EOS;
	}

	pairs = malloc((size_t)n * sizeof(samp_pair));
	if (!pairs) {
		free(buf);
		free(pbuf);
		return VOCAB_EOS;
	}
	for (i = 0; i < n; i++) {
		pairs[i].id = i;
		pairs[i].p = pbuf[i];
	}
	qsort(pairs, (size_t)n, sizeof(samp_pair), samp_cmp_desc);

	keep = n;
	if (top_k > 0 && top_k < keep)
		keep = top_k;
	if (top_p > 0.0 && top_p < 1.0) {
		cum = 0.0;
		for (i = 0; i < keep; i++) {
			cum += (double)pairs[i].p;
			if (cum >= top_p) {
				keep = i + 1;
				break;
			}
		}
	}
	if (keep < 1)
		keep = 1;

	sum = 0.0;
	for (i = 0; i < keep; i++)
		sum += (double)pairs[i].p;
	if (sum <= 0.0) {
		best = pairs[0].id;
		free(pairs);
		free(buf);
		free(pbuf);
		return best;
	}
	u = (double)rand() / (double)RAND_MAX * sum;
	cum = 0.0;
	best = pairs[keep - 1].id;
	for (i = 0; i < keep; i++) {
		cum += (double)pairs[i].p;
		if (u <= cum) {
			best = pairs[i].id;
			break;
		}
	}
	free(pairs);
	free(buf);
	free(pbuf);
	return best;
}

int lm_generate(lm *model, const char *prompt, char *out, int out_cap,
	int max_new, double temperature, int top_k, double top_p)
{
	int n;
	int i;
	int next;
	int ids[LM_MAX_CTX];
	mat *hidden;
	vec *logits;

	n = lm_generate_kv(model, prompt, out, out_cap, max_new, temperature,
	    top_k, top_p);
	if (n >= 0)
		return n;
	if (!model || !prompt || !out || out_cap <= 1 || max_new < 0)
		return -1;
	n = lm_encode(model, prompt, ids, LM_MAX_CTX);
	if (n <= 0)
		return -1;
	hidden = mat_create(LM_MAX_CTX, LM_D_MODEL);
	logits = vec_create(model->vocab_size);
	if (!hidden || !logits) {
		mat_free(&hidden);
		vec_free(&logits);
		return -1;
	}
	for (i = 0; i < max_new && n < LM_MAX_CTX; i++) {
		hidden->rows = n;
		if (lm_forward(hidden, model, ids, n) != 0) {
			mat_free(&hidden);
			vec_free(&logits);
			return -1;
		}
		if (lm_logits_at(logits, model, hidden, n - 1) != 0) {
			mat_free(&hidden);
			vec_free(&logits);
			return -1;
		}
		next = lm_sample_id(logits, temperature, top_k, top_p);
		if (next == VOCAB_EOS)
			break;
		ids[n++] = next;
	}
	if (lm_decode(model, ids, n, out, out_cap) < 0) {
		mat_free(&hidden);
		vec_free(&logits);
		return -1;
	}
	mat_free(&hidden);
	vec_free(&logits);
	return n;
}

int lm_generate_batch(lm *model, const char *const *prompts, char **outs,
	int *out_caps, int n, int max_new, double temperature, int top_k,
	double top_p)
{
	int i;
	int rc;

	if (!model || !prompts || !outs || !out_caps || n <= 0)
		return -1;
	for (i = 0; i < n; i++) {
		if (!prompts[i] || !outs[i] || out_caps[i] <= 0)
			return -1;
		rc = lm_generate(model, prompts[i], outs[i], out_caps[i],
		    max_new, temperature, top_k, top_p);
		if (rc < 0)
			return -1;
	}
	return 0;
}
