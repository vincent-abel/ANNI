/*
**	Language model forward + backward through Transformer
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "gradutil.h"
#include "lm_adam.h"
#include "lm.h"
#include "ops.h"
#include "posenc.h"
#include "tokenize.h"

static double rand_w(void)
{
	return ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * 0.02;
}

static void copy_mat(mat *dst, const mat *src)
{
	memcpy(dst->data, src->data,
	    (size_t)dst->rows * (size_t)dst->cols * sizeof(double));
}

static void softmax_inplace(double *x, int n)
{
	vec in;
	vec out;
	double *tmp;
	int i;

	tmp = malloc((size_t)n * sizeof(double));
	if (!tmp)
		return;
	in.data = x;
	in.n = n;
	out.data = tmp;
	out.n = n;
	if (vec_softmax(&out, &in) == 0) {
		for (i = 0; i < n; i++)
			x[i] = tmp[i];
	}
	free(tmp);
}

lm *lm_create(void)
{
	lm *model;
	int i;
	int n;

	model = malloc(sizeof(lm));
	if (!model)
		return NULL;
	model->vocab = vocab_byte_create();
	if (!model->vocab) {
		free(model);
		return NULL;
	}
	model->vocab_size = vocab_size(model->vocab);
	model->embed = embed_create(model->vocab_size);
	model->tr = transformer_create();
	model->w_head = mat_create(LM_D_MODEL, model->vocab_size);
	model->b_head = vec_create(model->vocab_size);
	model->dw_head = mat_create(LM_D_MODEL, model->vocab_size);
	model->db_head = vec_create(model->vocab_size);
	if (!model->embed || !model->tr || !model->w_head || !model->b_head
	    || !model->dw_head || !model->db_head) {
		lm_free(&model);
		return NULL;
	}
	n = model->w_head->rows * model->w_head->cols;
	for (i = 0; i < n; i++)
		model->w_head->data[i] = rand_w();
	vec_fill(model->b_head, 0.0);
	lm_zero_grad(model);
	return model;
}

void lm_free(lm **model)
{
	if (!model || !*model)
		return;
	vocab_free(&(*model)->vocab);
	embed_free(&(*model)->embed);
	transformer_free(&(*model)->tr);
	mat_free(&(*model)->w_head);
	vec_free(&(*model)->b_head);
	mat_free(&(*model)->dw_head);
	vec_free(&(*model)->db_head);
	free(*model);
	*model = NULL;
}

void lm_zero_grad(lm *model)
{
	if (!model)
		return;
	embed_zero_grad(model->embed);
	transformer_zero_grad(model->tr);
	mat_zero(model->dw_head);
	vec_zero(model->db_head);
}

lm_cache *lm_cache_create(int n)
{
	lm_cache *c;

	if (n <= 0)
		return NULL;
	c = malloc(sizeof(lm_cache));
	if (!c)
		return NULL;
	memset(c, 0, sizeof(*c));
	c->n = n;
	c->ids = calloc((size_t)n, sizeof(int));
	c->hidden = mat_create(n, LM_D_MODEL);
	if (!c->ids || !c->hidden) {
		lm_cache_free(&c);
		return NULL;
	}
	return c;
}

void lm_cache_free(lm_cache **c)
{
	if (!c || !*c)
		return;
	free((*c)->ids);
	mat_free(&(*c)->hidden);
	transformer_cache_free(&(*c)->trc);
	free(*c);
	*c = NULL;
}

int lm_forward_cache(lm *model, const int *ids, int n, lm_cache *c)
{
	int i;

	if (!model || !ids || !c || n <= 0 || c->n != n)
		return -1;
	for (i = 0; i < n; i++)
		c->ids[i] = ids[i];
	transformer_cache_free(&c->trc);
	c->trc = transformer_cache_create(n, model->tr);
	if (!c->trc)
		return -1;
	if (embed_lookup(c->hidden, model->embed, ids, n) != 0)
		return -1;
	posenc_sinusoidal_add(c->hidden);
	if (transformer_forward_cache(c->hidden, model->tr, c->trc) != 0)
		return -1;
	return 0;
}

int lm_forward(mat *hidden, const lm *model, const int *ids, int n)
{
	lm_cache *c;
	int rc;

	if (!hidden || !model || !ids || n <= 0)
		return -1;
	c = lm_cache_create(n);
	if (!c)
		return -1;
	rc = lm_forward_cache((lm *)model, ids, n, c);
	if (rc == 0)
		copy_mat(hidden, c->hidden);
	lm_cache_free(&c);
	return rc;
}

int lm_logits_at(vec *logits, const lm *model, const mat *hidden, int t)
{
	int j;
	int v;
	double sum;

	if (!logits || !model || !hidden || !logits->data)
		return -1;
	if (t < 0 || t >= hidden->rows)
		return -1;
	if (logits->n != model->vocab_size)
		return -1;
	for (v = 0; v < model->vocab_size; v++) {
		sum = model->b_head->data[v];
		for (j = 0; j < LM_D_MODEL; j++)
			sum += mat_get(hidden, t, j) * mat_get(model->w_head, j, v);
		logits->data[v] = sum;
	}
	return 0;
}

int lm_loss_backward(lm *model, const lm_cache *c, const int *ids, int n,
	mat *dhidden)
{
	vec *logits;
	double *delta;
	int t;
	int j;
	int v;

	if (!model || !c || !ids || !dhidden || n < 2)
		return -1;
	logits = vec_create(model->vocab_size);
	delta = malloc((size_t)model->vocab_size * sizeof(double));
	if (!logits || !delta) {
		vec_free(&logits);
		free(delta);
		return -1;
	}
	mat_zero(dhidden);
	for (t = 0; t < n - 1; t++) {
		if (lm_logits_at(logits, model, c->hidden, t) != 0) {
			vec_free(&logits);
			free(delta);
			return -1;
		}
		memcpy(delta, logits->data,
		    (size_t)model->vocab_size * sizeof(double));
		softmax_inplace(delta, model->vocab_size);
		delta[ids[t + 1]] -= 1.0;
		for (v = 0; v < model->vocab_size; v++) {
			model->db_head->data[v] += delta[v];
			for (j = 0; j < LM_D_MODEL; j++) {
				mat_set(model->dw_head, j, v,
				    mat_get(model->dw_head, j, v)
				    + mat_get(c->hidden, t, j) * delta[v]);
				mat_set(dhidden, t, j,
				    mat_get(dhidden, t, j)
				    + mat_get(model->w_head, j, v) * delta[v]);
			}
		}
	}
	vec_free(&logits);
	free(delta);
	return 0;
}

int lm_backward(lm *model, const mat *dhidden, const lm_cache *c)
{
	mat *dx;

	if (!model || !dhidden || !c || !c->trc)
		return -1;
	dx = mat_create(c->n, LM_D_MODEL);
	if (!dx)
		return -1;
	if (transformer_backward(dx, dhidden, model->tr, c->trc) != 0) {
		mat_free(&dx);
		return -1;
	}
	/* posenc is constant add — gradient passes through unchanged */
	if (embed_backward(model->embed, dx, c->ids, c->n) != 0) {
		mat_free(&dx);
		return -1;
	}
	mat_free(&dx);
	return 0;
}

double lm_nll(const lm *model, const int *ids, int n)
{
	mat *hidden;
	vec *logits;
	double loss;
	int t;
	int count;

	if (!model || !ids || n < 2)
		return -1.0;
	hidden = mat_create(n, LM_D_MODEL);
	logits = vec_create(model->vocab_size);
	if (!hidden || !logits) {
		mat_free(&hidden);
		vec_free(&logits);
		return -1.0;
	}
	if (lm_forward(hidden, model, ids, n) != 0) {
		mat_free(&hidden);
		vec_free(&logits);
		return -1.0;
	}
	loss = 0.0;
	count = 0;
	for (t = 0; t < n - 1; t++) {
		if (lm_logits_at(logits, model, hidden, t) != 0) {
			mat_free(&hidden);
			vec_free(&logits);
			return -1.0;
		}
		loss += vec_softmax_cross_entropy(logits, ids[t + 1]);
		count++;
	}
	mat_free(&hidden);
	vec_free(&logits);
	if (count == 0)
		return -1.0;
	return loss / (double)count;
}

double lm_perplexity(const lm *model, const int *ids, int n)
{
	double nll;

	nll = lm_nll(model, ids, n);
	if (nll < 0.0)
		return -1.0;
	return exp(nll);
}

double lm_train_steps(lm *model, const char *text, int steps, double lr)
{
	int ids[LM_MAX_CTX];
	int n;
	int step;
	lm_cache *cache;
	mat *dhidden;
	double loss;

	if (!model || !text || steps <= 0 || lr <= 0.0)
		return -1.0;
	n = tokenize_encode(text, ids, LM_MAX_CTX);
	if (n < 2)
		return -1.0;

	cache = lm_cache_create(n);
	dhidden = mat_create(n, LM_D_MODEL);
	if (!cache || !dhidden) {
		lm_cache_free(&cache);
		mat_free(&dhidden);
		return -1.0;
	}

	{
		lm_adam *opt;

		opt = lm_adam_create(model);
		if (!opt) {
			lm_cache_free(&cache);
			mat_free(&dhidden);
			return -1.0;
		}
		for (step = 0; step < steps; step++) {
			lm_adam_zero_grad(model);
			if (lm_forward_cache(model, ids, n, cache) != 0)
				break;
			if (lm_loss_backward(model, cache, ids, n, dhidden) != 0)
				break;
			if (lm_backward(model, dhidden, cache) != 0)
				break;
			lm_adam_step(opt, lr);
		}
		lm_adam_free(&opt);
	}

	loss = lm_nll(model, ids, n);
	lm_cache_free(&cache);
	mat_free(&dhidden);
	return loss;
}
