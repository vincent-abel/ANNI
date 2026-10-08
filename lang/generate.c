/*
**	Autoregressive decode (greedy or temperature sampling)
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "generate.h"
#include "kv.h"
#include "ops.h"
#include "tokenize.h"
#include "vocab.h"

static int sample_id(const vec *logits, double temperature)
{
	int i;
	int best;
	double best_v;
	double sum;
	double u;
	vec scaled;
	vec probs;
	double *buf;
	double *pbuf;

	if (!logits || !logits->data || logits->n <= 0)
		return VOCAB_EOS;

	if (temperature <= 0.0) {
		best = 0;
		best_v = logits->data[0];
		for (i = 1; i < logits->n; i++) {
			if (logits->data[i] > best_v) {
				best_v = logits->data[i];
				best = i;
			}
		}
		return best;
	}

	buf = malloc((size_t)logits->n * sizeof(double));
	pbuf = malloc((size_t)logits->n * sizeof(double));
	if (!buf || !pbuf) {
		free(buf);
		free(pbuf);
		return VOCAB_EOS;
	}
	for (i = 0; i < logits->n; i++)
		buf[i] = logits->data[i] / temperature;
	scaled.data = buf;
	scaled.n = logits->n;
	probs.data = pbuf;
	probs.n = logits->n;
	if (vec_softmax(&probs, &scaled) != 0) {
		free(buf);
		free(pbuf);
		return VOCAB_EOS;
	}
	u = (double)rand() / (double)RAND_MAX;
	sum = 0.0;
	best = logits->n - 1;
	for (i = 0; i < logits->n; i++) {
		sum += pbuf[i];
		if (u <= sum) {
			best = i;
			break;
		}
	}
	free(buf);
	free(pbuf);
	return best;
}

int lm_generate(lm *model, const char *prompt, char *out, int out_cap,
	int max_new, double temperature)
{
	/* Prefer KV-cache decode; fall back to full reforward if it fails. */
	int n;
	int i;
	int next;
	int ids[LM_MAX_CTX];
	mat *hidden;
	vec *logits;

	n = lm_generate_kv(model, prompt, out, out_cap, max_new, temperature);
	if (n >= 0)
		return n;
	if (!model || !prompt || !out || out_cap <= 1 || max_new < 0)
		return -1;
	n = tokenize_encode(prompt, ids, LM_MAX_CTX);
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
		next = sample_id(logits, temperature);
		if (next == VOCAB_EOS)
			break;
		ids[n++] = next;
	}
	if (tokenize_decode(ids, n, out, out_cap) < 0) {
		mat_free(&hidden);
		vec_free(&logits);
		return -1;
	}
	mat_free(&hidden);
	vec_free(&logits);
	return n;
}
