/*
**	KV cache incremental decode (attention over cached K/V)
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "attn.h"
#include "embed.h"
#include "ffn.h"
#include "generate.h"
#include "kv.h"
#include "layernorm.h"
#include "ops.h"
#include "posenc.h"
#include "resid.h"
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

lm_kv *lm_kv_create(const lm *model, int max_T)
{
	lm_kv *kv;
	int i;

	if (!model || !model->tr || max_T <= 0)
		return NULL;
	kv = malloc(sizeof(lm_kv));
	if (!kv)
		return NULL;
	memset(kv, 0, sizeof(*kv));
	kv->n_blocks = model->tr->n_blocks;
	kv->max_T = max_T;
	kv->len = 0;
	kv->k = calloc((size_t)kv->n_blocks, sizeof(mat *));
	kv->v = calloc((size_t)kv->n_blocks, sizeof(mat *));
	if (!kv->k || !kv->v) {
		lm_kv_free(&kv);
		return NULL;
	}
	for (i = 0; i < kv->n_blocks; i++) {
		kv->k[i] = mat_create(max_T, LM_D_MODEL);
		kv->v[i] = mat_create(max_T, LM_D_MODEL);
		if (!kv->k[i] || !kv->v[i]) {
			lm_kv_free(&kv);
			return NULL;
		}
	}
	return kv;
}

void lm_kv_free(lm_kv **kv)
{
	int i;

	if (!kv || !*kv)
		return;
	if ((*kv)->k) {
		for (i = 0; i < (*kv)->n_blocks; i++)
			mat_free(&(*kv)->k[i]);
		free((*kv)->k);
	}
	if ((*kv)->v) {
		for (i = 0; i < (*kv)->n_blocks; i++)
			mat_free(&(*kv)->v[i]);
		free((*kv)->v);
	}
	free(*kv);
	*kv = NULL;
}

void lm_kv_reset(lm_kv *kv)
{
	if (kv)
		kv->len = 0;
}

static int softmax_row(double *row, int n, double *tmp)
{
	vec in;
	vec out;
	int i;
	int rc;

	in.data = row;
	in.n = n;
	out.data = tmp;
	out.n = n;
	rc = vec_softmax(&out, &in);
	if (rc == 0) {
		for (i = 0; i < n; i++)
			row[i] = tmp[i];
	}
	return rc;
}

/* Attend one query row against K/V cache length len; write D_MODEL out. */
static int attn_decode_row(double *out, const attn *a, const double *xrow,
	mat *k_cache, mat *v_cache, int len, double *scratch)
{
	double *q;
	double *k;
	double *v;
	double *scores;
	double *otmp;
	double *stmp;
	double scale;
	int h;
	int j;
	int t;
	int d;
	int off;
	double sum;

	if (len <= 0 || len > k_cache->rows)
		return -1;
	d = a->d_head;
	scale = 1.0 / sqrt((double)d);
	q = scratch;
	k = scratch + LM_D_MODEL;
	v = scratch + 2 * LM_D_MODEL;
	scores = scratch + 3 * LM_D_MODEL;
	otmp = scores + len;
	stmp = otmp + LM_D_MODEL;
	/* q = x * Wq ; k = x * Wk ; v = x * Wv  (1 x D) */
	for (j = 0; j < LM_D_MODEL; j++) {
		sum = 0.0;
		for (t = 0; t < LM_D_MODEL; t++)
			sum += xrow[t] * a->wq->data[t * LM_D_MODEL + j];
		q[j] = sum;
		sum = 0.0;
		for (t = 0; t < LM_D_MODEL; t++)
			sum += xrow[t] * a->wk->data[t * LM_D_MODEL + j];
		k[j] = sum;
		sum = 0.0;
		for (t = 0; t < LM_D_MODEL; t++)
			sum += xrow[t] * a->wv->data[t * LM_D_MODEL + j];
		v[j] = sum;
	}
	memcpy(&k_cache->data[(len - 1) * LM_D_MODEL], k,
	    (size_t)LM_D_MODEL * sizeof(double));
	memcpy(&v_cache->data[(len - 1) * LM_D_MODEL], v,
	    (size_t)LM_D_MODEL * sizeof(double));
	memset(otmp, 0, (size_t)LM_D_MODEL * sizeof(double));
	for (h = 0; h < a->n_heads; h++) {
		off = h * d;
		for (j = 0; j < len; j++) {
			sum = 0.0;
			for (t = 0; t < d; t++)
				sum += q[off + t]
				    * k_cache->data[j * LM_D_MODEL + off + t];
			scores[j] = sum * scale;
		}
		if (softmax_row(scores, len, stmp) != 0)
			return -1;
		for (t = 0; t < d; t++) {
			sum = 0.0;
			for (j = 0; j < len; j++)
				sum += scores[j]
				    * v_cache->data[j * LM_D_MODEL + off + t];
			otmp[off + t] = sum;
		}
	}
	for (j = 0; j < LM_D_MODEL; j++) {
		sum = 0.0;
		for (t = 0; t < LM_D_MODEL; t++)
			sum += otmp[t] * a->wo->data[t * LM_D_MODEL + j];
		out[j] = sum;
	}
	return 0;
}

static int ln_row(double *x, const layernorm *ln)
{
	int j;
	double mean;
	double var;
	double inv;

	mean = 0.0;
	for (j = 0; j < LM_D_MODEL; j++)
		mean += x[j];
	mean /= (double)LM_D_MODEL;
	var = 0.0;
	for (j = 0; j < LM_D_MODEL; j++)
		var += (x[j] - mean) * (x[j] - mean);
	var /= (double)LM_D_MODEL;
	inv = 1.0 / sqrt(var + ln->eps);
	for (j = 0; j < LM_D_MODEL; j++)
		x[j] = (x[j] - mean) * inv * ln->gamma->data[j]
		    + ln->beta->data[j];
	return 0;
}

static int ffn_row(double *x, const ffn *f, double *hid)
{
	int j;
	int k;
	double sum;

	for (j = 0; j < LM_D_FF; j++) {
		sum = f->b1->data[j];
		for (k = 0; k < LM_D_MODEL; k++)
			sum += x[k] * f->w1->data[k * LM_D_FF + j];
		hid[j] = sum > 0.0 ? sum : 0.0;
	}
	for (j = 0; j < LM_D_MODEL; j++) {
		sum = f->b2->data[j];
		for (k = 0; k < LM_D_FF; k++)
			sum += hid[k] * f->w2->data[k * LM_D_MODEL + j];
		x[j] = sum;
	}
	return 0;
}

static int decode_token(lm *model, lm_kv *kv, int id, int pos, double *x,
	double *scratch)
{
	tblock *b;
	double *lnbuf;
	double *attn_out;
	double *ffn_hid;
	int i;
	int j;

	if (pos < 0 || pos >= kv->max_T)
		return -1;
	if (embed_lookup_row(x, model->embed, id) != 0)
		return -1;
	posenc_add_row(x, pos);
	lnbuf = scratch + 8 * LM_D_MODEL;
	attn_out = lnbuf + LM_D_MODEL;
	ffn_hid = attn_out + LM_D_MODEL; /* size LM_D_FF */
	kv->len = pos + 1;
	for (i = 0; i < model->tr->n_blocks; i++) {
		b = &model->tr->blocks[i];
		memcpy(lnbuf, x, (size_t)LM_D_MODEL * sizeof(double));
		ln_row(lnbuf, b->ln1);
		if (attn_decode_row(attn_out, b->attn, lnbuf, kv->k[i], kv->v[i],
			kv->len, scratch) != 0)
			return -1;
		for (j = 0; j < LM_D_MODEL; j++)
			x[j] += attn_out[j];
		memcpy(lnbuf, x, (size_t)LM_D_MODEL * sizeof(double));
		ln_row(lnbuf, b->ln2);
		memcpy(attn_out, lnbuf, (size_t)LM_D_MODEL * sizeof(double));
		ffn_row(attn_out, b->ffn, ffn_hid);
		for (j = 0; j < LM_D_MODEL; j++)
			x[j] += attn_out[j];
	}
	return 0;
}

int lm_generate_kv(lm *model, const char *prompt, char *out, int out_cap,
	int max_new, double temperature)
{
	int ids[LM_MAX_CTX];
	int n;
	int i;
	int next;
	lm_kv *kv;
	vec *logits;
	double *x;
	double *scratch;
	mat hrow;

	if (!model || !prompt || !out || out_cap <= 1 || max_new < 0)
		return -1;
	n = tokenize_encode(prompt, ids, LM_MAX_CTX);
	if (n <= 0)
		return -1;
	kv = lm_kv_create(model, LM_MAX_CTX);
	logits = vec_create(model->vocab_size);
	x = calloc((size_t)LM_D_MODEL, sizeof(double));
	scratch = calloc((size_t)(12 * LM_D_MODEL + LM_D_FF + LM_MAX_CTX * 2),
	    sizeof(double));
	if (!kv || !logits || !x || !scratch) {
		lm_kv_free(&kv);
		vec_free(&logits);
		free(x);
		free(scratch);
		return -1;
	}
	for (i = 0; i < n; i++) {
		if (decode_token(model, kv, ids[i], i, x, scratch) != 0)
			goto fail;
	}
	for (i = 0; i < max_new && n < LM_MAX_CTX; i++) {
		hrow.data = x;
		hrow.rows = 1;
		hrow.cols = LM_D_MODEL;
		if (lm_logits_at(logits, model, &hrow, 0) != 0)
			goto fail;
		next = sample_id(logits, temperature);
		if (next == VOCAB_EOS)
			break;
		ids[n] = next;
		if (decode_token(model, kv, next, n, x, scratch) != 0)
			goto fail;
		n++;
	}
	if (tokenize_decode(ids, n, out, out_cap) < 0)
		goto fail;
	lm_kv_free(&kv);
	vec_free(&logits);
	free(x);
	free(scratch);
	return n;
fail:
	lm_kv_free(&kv);
	vec_free(&logits);
	free(x);
	free(scratch);
	return -1;
}
