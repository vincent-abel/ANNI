/*
**	Language model forward + next-token head training
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "lm.h"
#include "ops.h"
#include "posenc.h"
#include "tokenize.h"

static double rand_w(void)
{
	return ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * 0.02;
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
	if (!model->embed || !model->tr || !model->w_head || !model->b_head) {
		lm_free(&model);
		return NULL;
	}
	n = model->w_head->rows * model->w_head->cols;
	for (i = 0; i < n; i++)
		model->w_head->data[i] = rand_w();
	vec_fill(model->b_head, 0.0);
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
	free(*model);
	*model = NULL;
}

int lm_forward(mat *hidden, const lm *model, const int *ids, int n)
{
	if (!hidden || !model || !ids || n <= 0)
		return -1;
	if (hidden->rows != n || hidden->cols != LM_D_MODEL)
		return -1;
	if (embed_lookup(hidden, model->embed, ids, n) != 0)
		return -1;
	posenc_sinusoidal_add(hidden);
	if (transformer_forward(hidden, model->tr) != 0)
		return -1;
	return 0;
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

double lm_train_steps(lm *model, const char *text, int steps, double lr)
{
	int ids[LM_MAX_CTX];
	int n;
	int step;
	int t;
	int j;
	int v;
	mat *hidden;
	vec *logits;
	double *delta;
	double loss;
	double before;
	double dh[LM_D_MODEL];

	if (!model || !text || steps <= 0 || lr <= 0.0)
		return -1.0;
	n = tokenize_encode(text, ids, LM_MAX_CTX);
	if (n < 2)
		return -1.0;

	before = lm_nll(model, ids, n);
	hidden = mat_create(n, LM_D_MODEL);
	logits = vec_create(model->vocab_size);
	delta = malloc((size_t)model->vocab_size * sizeof(double));
	if (!hidden || !logits || !delta) {
		mat_free(&hidden);
		vec_free(&logits);
		free(delta);
		return -1.0;
	}

	for (step = 0; step < steps; step++) {
		if (lm_forward(hidden, model, ids, n) != 0)
			break;
		for (t = 0; t < n - 1; t++) {
			if (lm_logits_at(logits, model, hidden, t) != 0)
				break;
			memcpy(delta, logits->data,
			    (size_t)model->vocab_size * sizeof(double));
			softmax_inplace(delta, model->vocab_size);
			delta[ids[t + 1]] -= 1.0;
			for (j = 0; j < LM_D_MODEL; j++)
				dh[j] = 0.0;
			for (v = 0; v < model->vocab_size; v++) {
				model->b_head->data[v] -= lr * delta[v];
				for (j = 0; j < LM_D_MODEL; j++) {
					dh[j] += mat_get(model->w_head, j, v) * delta[v];
					mat_set(model->w_head, j, v,
					    mat_get(model->w_head, j, v)
					    - lr * mat_get(hidden, t, j) * delta[v]);
				}
			}
			/* rough embed update (treat transformer as identity) */
			for (j = 0; j < LM_D_MODEL; j++)
				mat_set(model->embed->table, ids[t], j,
				    mat_get(model->embed->table, ids[t], j)
				    - lr * dh[j]);
		}
	}

	loss = lm_nll(model, ids, n);
	mat_free(&hidden);
	vec_free(&logits);
	free(delta);
	(void)before;
	return loss;
}
