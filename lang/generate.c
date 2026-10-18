/*
**	Autoregressive decode (greedy, temperature, top-k / top-p)
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "decode.h"
#include "generate.h"
#include "gpu.h"
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

struct cap_buf {
	char	*out;
	int	cap;
	int	n;
};

static void cap_append(const char *s, int n, void *ud)
{
	struct cap_buf *b;
	int room;

	b = ud;
	if (!b || !b->out || b->cap <= 1 || !s || n <= 0)
		return;
	room = b->cap - 1 - b->n;
	if (room <= 0)
		return;
	if (n > room)
		n = room;
	memcpy(b->out + b->n, s, (size_t)n);
	b->n += n;
	b->out[b->n] = '\0';
}

int lm_generate(lm *model, const char *prompt, char *out, int out_cap,
	int max_new, double temperature, int top_k, double top_p)
{
	struct cap_buf b;

	if (!model || !prompt || !out || out_cap <= 1 || max_new < 0)
		return -1;
	b.out = out;
	b.cap = out_cap;
	b.n = 0;
	out[0] = '\0';
	cap_append(prompt, (int)strlen(prompt), &b);
	/* Same forward as lm_generate_stream. KV decode does not match it. */
	return lm_generate_stream(model, prompt, max_new, temperature, top_k,
	    top_p, cap_append, &b);
}

/*
** ANN_DECODE_CHECK=N: for the first N decode steps, also run the full
** forward and print the max |logit| difference to stderr.
*/
static void decode_check(lm *model, const int *ids, int n, const vec *logits,
	int step)
{
	static int limit = -1;
	lm_cache *c;
	vec *ref;
	double diff;
	double d;
	int ref_arg;
	int dec_arg;
	int v;

	if (limit < 0)
		limit = getenv("ANN_DECODE_CHECK")
		    ? atoi(getenv("ANN_DECODE_CHECK")) : 0;
	if (step >= limit)
		return;
	c = lm_cache_create(n);
	ref = vec_create(model->vocab_size);
	if (c && ref && lm_forward_cache(model, ids, n, c) == 0
	    && lm_logits_at(ref, model, c->hidden, n - 1) == 0) {
		diff = 0.0;
		ref_arg = 0;
		dec_arg = 0;
		for (v = 0; v < ref->n; v++) {
			d = fabs((double)ref->data[v] - (double)logits->data[v]);
			if (d > diff)
				diff = d;
			if (ref->data[v] > ref->data[ref_arg])
				ref_arg = v;
			if (logits->data[v] > logits->data[dec_arg])
				dec_arg = v;
		}
		fprintf(stderr, "[decode-check] step %d pos %d max|dlogit| %.3g"
		    " argmax ref %d dec %d\n", step, n - 1, diff, ref_arg,
		    dec_arg);
	}
	vec_free(&ref);
	lm_cache_free(&c);
}

/* KV-cache decode on the GPU; -2 when the decoder is unavailable. */
static int generate_stream_dec(lm *model, int *ids, int n, int max_new,
	double temperature, int top_k, double top_p,
	void (*on_text)(const char *s, int n, void *ud), void *ud)
{
	lm_decoder *dec;
	vec *logits;
	char piece[1024];
	int nnew;
	int next;
	int pn;
	int i;

	dec = lm_decoder_create(model);
	if (!dec)
		return -2;
	logits = vec_create(model->vocab_size);
	if (!logits || lm_decoder_prefill(dec, ids, n, logits) != 0) {
		vec_free(&logits);
		lm_decoder_free(&dec);
		return -2;
	}
	nnew = 0;
	for (i = 0; i < max_new && n < LM_MAX_CTX; i++) {
		if (i > 0) {
			if (lm_decoder_step(dec, ids[n - 1], logits) != 0)
				goto fail;
			decode_check(model, ids, n, logits, i - 1);
		}
		next = lm_sample_id(logits, temperature, top_k, top_p);
		if (next == VOCAB_EOS)
			break;
		ids[n++] = next;
		nnew++;
		pn = lm_decode(model, &next, 1, piece, (int)sizeof(piece));
		if (pn < 0)
			goto fail;
		if (on_text && pn > 0)
			on_text(piece, pn, ud);
	}
	vec_free(&logits);
	lm_decoder_free(&dec);
	return nnew;
fail:
	vec_free(&logits);
	lm_decoder_free(&dec);
	return -1;
}

int lm_generate_stream(lm *model, const char *prompt, int max_new,
	double temperature, int top_k, double top_p,
	void (*on_text)(const char *s, int n, void *ud), void *ud)
{
	int ids[LM_MAX_CTX];
	int n;
	int i;
	int next;
	int pn;
	int nnew;
	char piece[1024];
	lm_cache *cache;
	vec *logits;
	int sync_was;

	if (!model || !prompt || max_new < 0)
		return -1;
	n = lm_encode(model, prompt, ids, LM_MAX_CTX);
	if (n <= 0)
		return -1;
	logits = vec_create(model->vocab_size);
	if (!logits)
		return -1;
	/* One upload; later steps read the device twin.
	** BATCH is the mode the B=1 attention pull was fixed for. */
	lm_weights_to_device(model);
	sync_was = gpu_sync_mode();
	gpu_set_sync_mode(GPU_SYNC_BATCH);
	nnew = generate_stream_dec(model, ids, n, max_new, temperature, top_k,
	    top_p, on_text, ud);
	if (nnew != -2) {
		gpu_set_sync_mode(sync_was);
		vec_free(&logits);
		return nnew;
	}
	nnew = 0;
	for (i = 0; i < max_new && n < LM_MAX_CTX; i++) {
		cache = lm_cache_create(n);
		if (!cache)
			goto fail;
		if (lm_forward_cache(model, ids, n, cache) != 0
		    || lm_logits_at(logits, model, cache->hidden, n - 1) != 0) {
			lm_cache_free(&cache);
			goto fail;
		}
		lm_cache_free(&cache);
		next = lm_sample_id(logits, temperature, top_k, top_p);
		if (next == VOCAB_EOS)
			break;
		ids[n++] = next;
		nnew++;
		pn = lm_decode(model, &next, 1, piece, (int)sizeof(piece));
		if (pn < 0)
			goto fail;
		if (on_text && pn > 0)
			on_text(piece, pn, ud);
	}
	gpu_set_sync_mode(sync_was);
	vec_free(&logits);
	return nnew;
fail:
	gpu_set_sync_mode(sync_was);
	vec_free(&logits);
	return -1;
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
