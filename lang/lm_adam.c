/*
**	Adam optimizer over flat LM param/grad slots
*/
#include <math.h>
#include <stdlib.h>
#include "gpu.h"
#include "lm_adam.h"

static void add_mat(lm_adam_slot *slots, int *i, mat *w, mat *dw)
{
	int k;
	int n;

	if (!w || !dw || !w->data || !dw->data)
		return;
	n = w->rows * w->cols;
	for (k = 0; k < n; k++) {
		slots[*i].param = &w->data[k];
		slots[*i].grad = &dw->data[k];
		slots[*i].m = 0.0;
		slots[*i].v = 0.0;
		(*i)++;
	}
}

static void add_vec(lm_adam_slot *slots, int *i, vec *w, vec *dw)
{
	int k;

	if (!w || !dw || !w->data || !dw->data)
		return;
	for (k = 0; k < w->n; k++) {
		slots[*i].param = &w->data[k];
		slots[*i].grad = &dw->data[k];
		slots[*i].m = 0.0;
		slots[*i].v = 0.0;
		(*i)++;
	}
}

static int count_params(lm *model)
{
	int n;
	int i;

	n = model->w_head->rows * model->w_head->cols + model->b_head->n;
	n += model->embed->table->rows * model->embed->table->cols;
	for (i = 0; i < model->tr->n_blocks; i++) {
		n += 4 * LM_D_MODEL * LM_D_MODEL;
		n += LM_D_MODEL * LM_D_FF + LM_D_FF + LM_D_FF * LM_D_MODEL + LM_D_MODEL;
		n += 4 * LM_D_MODEL;
	}
	return n;
}

lm_adam *lm_adam_create(lm *model)
{
	lm_adam *opt;
	int i;
	int n;
	tblock *b;

	if (!model)
		return NULL;
	n = count_params(model);
	opt = malloc(sizeof(lm_adam));
	if (!opt)
		return NULL;
	opt->slots = calloc((size_t)n, sizeof(lm_adam_slot));
	if (!opt->slots) {
		free(opt);
		return NULL;
	}
	opt->n = n;
	opt->t = 0;
	opt->beta1 = 0.9;
	opt->beta2 = 0.999;
	opt->eps = 1e-8;
	i = 0;
	add_mat(opt->slots, &i, model->w_head, model->dw_head);
	add_vec(opt->slots, &i, model->b_head, model->db_head);
	add_mat(opt->slots, &i, model->embed->table, model->embed->dtable);
	for (n = 0; n < model->tr->n_blocks; n++) {
		b = &model->tr->blocks[n];
		add_mat(opt->slots, &i, b->attn->wq, b->attn->dwq);
		add_mat(opt->slots, &i, b->attn->wk, b->attn->dwk);
		add_mat(opt->slots, &i, b->attn->wv, b->attn->dwv);
		add_mat(opt->slots, &i, b->attn->wo, b->attn->dwo);
		add_mat(opt->slots, &i, b->ffn->w1, b->ffn->dw1);
		add_vec(opt->slots, &i, b->ffn->b1, b->ffn->db1);
		add_mat(opt->slots, &i, b->ffn->w2, b->ffn->dw2);
		add_vec(opt->slots, &i, b->ffn->b2, b->ffn->db2);
		add_vec(opt->slots, &i, b->ln1->gamma, b->ln1->dgamma);
		add_vec(opt->slots, &i, b->ln1->beta, b->ln1->dbeta);
		add_vec(opt->slots, &i, b->ln2->gamma, b->ln2->dgamma);
		add_vec(opt->slots, &i, b->ln2->beta, b->ln2->dbeta);
	}
	opt->n = i;
	return opt;
}

void lm_adam_free(lm_adam **opt)
{
	if (!opt || !*opt)
		return;
	free((*opt)->slots);
	free(*opt);
	*opt = NULL;
}

void lm_adam_zero_grad(lm *model)
{
	lm_zero_grad(model);
}

static int adam_step_gpu(lm_adam *opt, double lr)
{
	double *param;
	double *grad;
	double *m;
	double *v;
	int i;
	int n;
	size_t nbytes;
	int rc;

	n = opt->n;
	nbytes = (size_t)n * sizeof(double);
	param = malloc(nbytes);
	grad = malloc(nbytes);
	m = malloc(nbytes);
	v = malloc(nbytes);
	if (!param || !grad || !m || !v) {
		free(param);
		free(grad);
		free(m);
		free(v);
		return -1;
	}
	for (i = 0; i < n; i++) {
		param[i] = *opt->slots[i].param;
		grad[i] = *opt->slots[i].grad;
		m[i] = opt->slots[i].m;
		v[i] = opt->slots[i].v;
	}
	rc = gpu_adam_step(param, grad, m, v, n, lr, opt->beta1, opt->beta2,
	    opt->eps, opt->t);
	if (rc == 0) {
		for (i = 0; i < n; i++) {
			*opt->slots[i].param = param[i];
			*opt->slots[i].grad = 0.0;
			opt->slots[i].m = m[i];
			opt->slots[i].v = v[i];
		}
	}
	free(param);
	free(grad);
	free(m);
	free(v);
	return rc;
}

void lm_adam_step(lm_adam *opt, double lr)
{
	int i;
	double g;
	double mhat;
	double vhat;
	double bc1;
	double bc2;

	if (!opt || lr <= 0.0)
		return;
	opt->t++;
	if (gpu_available() && adam_step_gpu(opt, lr) == 0)
		return;
	bc1 = 1.0 - pow(opt->beta1, (double)opt->t);
	bc2 = 1.0 - pow(opt->beta2, (double)opt->t);
	for (i = 0; i < opt->n; i++) {
		g = *opt->slots[i].grad;
		opt->slots[i].m = opt->beta1 * opt->slots[i].m + (1.0 - opt->beta1) * g;
		opt->slots[i].v = opt->beta2 * opt->slots[i].v + (1.0 - opt->beta2) * g * g;
		mhat = opt->slots[i].m / bc1;
		vhat = opt->slots[i].v / bc2;
		*opt->slots[i].param -= lr * mhat / (sqrt(vhat) + opt->eps);
		*opt->slots[i].grad = 0.0;
	}
}
