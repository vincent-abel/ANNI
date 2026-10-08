/*
**	LM weight (+ optional Adam) binary serialization
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lm_serialize.h"

static int write_i32(FILE *fp, int v)
{
	return fwrite(&v, sizeof(int), 1, fp) == 1 ? 0 : -1;
}

static int read_i32(FILE *fp, int *v)
{
	return fread(v, sizeof(int), 1, fp) == 1 ? 0 : -1;
}

static int write_f64(FILE *fp, double v)
{
	return fwrite(&v, sizeof(double), 1, fp) == 1 ? 0 : -1;
}

static int read_f64(FILE *fp, double *v)
{
	return fread(v, sizeof(double), 1, fp) == 1 ? 0 : -1;
}

static int write_mat(FILE *fp, const mat *m)
{
	int n;

	if (!m || !m->data)
		return -1;
	if (write_i32(fp, m->rows) != 0 || write_i32(fp, m->cols) != 0)
		return -1;
	n = m->rows * m->cols;
	return fwrite(m->data, sizeof(double), (size_t)n, fp) == (size_t)n ? 0 : -1;
}

static int read_mat_into(FILE *fp, mat *m)
{
	int rows;
	int cols;
	int n;

	if (read_i32(fp, &rows) != 0 || read_i32(fp, &cols) != 0)
		return -1;
	if (!m || rows != m->rows || cols != m->cols)
		return -1;
	n = rows * cols;
	return fread(m->data, sizeof(double), (size_t)n, fp) == (size_t)n ? 0 : -1;
}

static int write_vec(FILE *fp, const vec *v)
{
	if (!v || !v->data)
		return -1;
	if (write_i32(fp, v->n) != 0)
		return -1;
	return fwrite(v->data, sizeof(double), (size_t)v->n, fp) == (size_t)v->n ? 0 : -1;
}

static int read_vec_into(FILE *fp, vec *v)
{
	int n;

	if (read_i32(fp, &n) != 0 || !v || n != v->n)
		return -1;
	return fread(v->data, sizeof(double), (size_t)n, fp) == (size_t)n ? 0 : -1;
}

static int write_block(FILE *fp, const tblock *b)
{
	if (write_mat(fp, b->attn->wq) != 0 || write_mat(fp, b->attn->wk) != 0
	    || write_mat(fp, b->attn->wv) != 0 || write_mat(fp, b->attn->wo) != 0)
		return -1;
	if (write_mat(fp, b->ffn->w1) != 0 || write_vec(fp, b->ffn->b1) != 0
	    || write_mat(fp, b->ffn->w2) != 0 || write_vec(fp, b->ffn->b2) != 0)
		return -1;
	if (write_vec(fp, b->ln1->gamma) != 0 || write_vec(fp, b->ln1->beta) != 0
	    || write_vec(fp, b->ln2->gamma) != 0 || write_vec(fp, b->ln2->beta) != 0)
		return -1;
	return 0;
}

static int read_block(FILE *fp, tblock *b)
{
	if (read_mat_into(fp, b->attn->wq) != 0 || read_mat_into(fp, b->attn->wk) != 0
	    || read_mat_into(fp, b->attn->wv) != 0 || read_mat_into(fp, b->attn->wo) != 0)
		return -1;
	if (read_mat_into(fp, b->ffn->w1) != 0 || read_vec_into(fp, b->ffn->b1) != 0
	    || read_mat_into(fp, b->ffn->w2) != 0 || read_vec_into(fp, b->ffn->b2) != 0)
		return -1;
	if (read_vec_into(fp, b->ln1->gamma) != 0 || read_vec_into(fp, b->ln1->beta) != 0
	    || read_vec_into(fp, b->ln2->gamma) != 0 || read_vec_into(fp, b->ln2->beta) != 0)
		return -1;
	return 0;
}

static int write_weights(FILE *fp, const lm *model)
{
	int i;

	if (write_i32(fp, LM_MAGIC) != 0)
		return -1;
	if (write_i32(fp, LM_D_MODEL) != 0 || write_i32(fp, LM_N_HEADS) != 0
	    || write_i32(fp, LM_D_FF) != 0 || write_i32(fp, LM_MAX_CTX) != 0
	    || write_i32(fp, LM_N_BLOCKS) != 0 || write_i32(fp, model->vocab_size) != 0)
		return -1;
	if (write_mat(fp, model->embed->table) != 0)
		return -1;
	for (i = 0; i < model->tr->n_blocks; i++) {
		if (write_block(fp, &model->tr->blocks[i]) != 0)
			return -1;
	}
	if (write_mat(fp, model->w_head) != 0 || write_vec(fp, model->b_head) != 0)
		return -1;
	return 0;
}

static int read_weights(FILE *fp, lm *model)
{
	int magic;
	int d_model;
	int n_heads;
	int d_ff;
	int max_ctx;
	int n_blocks;
	int vsize;
	int i;

	if (read_i32(fp, &magic) != 0 || magic != (int)LM_MAGIC)
		return -1;
	if (read_i32(fp, &d_model) != 0 || read_i32(fp, &n_heads) != 0
	    || read_i32(fp, &d_ff) != 0 || read_i32(fp, &max_ctx) != 0
	    || read_i32(fp, &n_blocks) != 0 || read_i32(fp, &vsize) != 0)
		return -1;
	if (d_model != LM_D_MODEL || n_heads != LM_N_HEADS || d_ff != LM_D_FF
	    || max_ctx != LM_MAX_CTX || n_blocks != LM_N_BLOCKS
	    || vsize != model->vocab_size)
		return -1;
	if (read_mat_into(fp, model->embed->table) != 0)
		return -1;
	for (i = 0; i < model->tr->n_blocks; i++) {
		if (read_block(fp, &model->tr->blocks[i]) != 0)
			return -1;
	}
	if (read_mat_into(fp, model->w_head) != 0 || read_vec_into(fp, model->b_head) != 0)
		return -1;
	return 0;
}

int lm_save(const lm *model, const char *path)
{
	FILE *fp;
	int rc;

	if (!model || !path)
		return -1;
	fp = fopen(path, "wb");
	if (!fp)
		return -1;
	rc = write_weights(fp, model);
	if (rc == 0)
		rc = write_i32(fp, 0); /* no adam section */
	fclose(fp);
	return rc;
}

lm *lm_load(const char *path)
{
	FILE *fp;
	lm *model;
	int has_adam;

	if (!path)
		return NULL;
	fp = fopen(path, "rb");
	if (!fp)
		return NULL;
	model = lm_create();
	if (!model) {
		fclose(fp);
		return NULL;
	}
	if (read_weights(fp, model) != 0) {
		lm_free(&model);
		fclose(fp);
		return NULL;
	}
	if (read_i32(fp, &has_adam) != 0)
		has_adam = 0;
	fclose(fp);
	(void)has_adam;
	return model;
}

int lm_save_checkpoint(const lm *model, const lm_adam *opt, int epoch, int step,
	const char *path)
{
	FILE *fp;
	int i;

	if (!model || !opt || !path)
		return -1;
	fp = fopen(path, "wb");
	if (!fp)
		return -1;
	if (write_weights(fp, model) != 0) {
		fclose(fp);
		return -1;
	}
	if (write_i32(fp, 1) != 0 || write_i32(fp, epoch) != 0 || write_i32(fp, step) != 0
	    || write_i32(fp, opt->t) != 0 || write_i32(fp, opt->n) != 0) {
		fclose(fp);
		return -1;
	}
	for (i = 0; i < opt->n; i++) {
		if (write_f64(fp, opt->slots[i].m) != 0 || write_f64(fp, opt->slots[i].v) != 0) {
			fclose(fp);
			return -1;
		}
	}
	fclose(fp);
	return 0;
}

static void add_mat(mat *acc, const mat *x)
{
	int i;
	int n;

	if (!acc || !x || !acc->data || !x->data)
		return;
	if (acc->rows != x->rows || acc->cols != x->cols)
		return;
	n = acc->rows * acc->cols;
	for (i = 0; i < n; i++)
		acc->data[i] += x->data[i];
}

static void add_vec(vec *acc, const vec *x)
{
	int i;

	if (!acc || !x || !acc->data || !x->data || acc->n != x->n)
		return;
	for (i = 0; i < acc->n; i++)
		acc->data[i] += x->data[i];
}

static void scale_mat(mat *m, double s)
{
	int i;
	int n;

	if (!m || !m->data)
		return;
	n = m->rows * m->cols;
	for (i = 0; i < n; i++)
		m->data[i] *= s;
}

static void scale_vec(vec *v, double s)
{
	int i;

	if (!v || !v->data)
		return;
	for (i = 0; i < v->n; i++)
		v->data[i] *= s;
}

static void add_block(tblock *acc, const tblock *x)
{
	add_mat(acc->attn->wq, x->attn->wq);
	add_mat(acc->attn->wk, x->attn->wk);
	add_mat(acc->attn->wv, x->attn->wv);
	add_mat(acc->attn->wo, x->attn->wo);
	add_mat(acc->ffn->w1, x->ffn->w1);
	add_vec(acc->ffn->b1, x->ffn->b1);
	add_mat(acc->ffn->w2, x->ffn->w2);
	add_vec(acc->ffn->b2, x->ffn->b2);
	add_vec(acc->ln1->gamma, x->ln1->gamma);
	add_vec(acc->ln1->beta, x->ln1->beta);
	add_vec(acc->ln2->gamma, x->ln2->gamma);
	add_vec(acc->ln2->beta, x->ln2->beta);
}

static void scale_block(tblock *b, double s)
{
	scale_mat(b->attn->wq, s);
	scale_mat(b->attn->wk, s);
	scale_mat(b->attn->wv, s);
	scale_mat(b->attn->wo, s);
	scale_mat(b->ffn->w1, s);
	scale_vec(b->ffn->b1, s);
	scale_mat(b->ffn->w2, s);
	scale_vec(b->ffn->b2, s);
	scale_vec(b->ln1->gamma, s);
	scale_vec(b->ln1->beta, s);
	scale_vec(b->ln2->gamma, s);
	scale_vec(b->ln2->beta, s);
}

static void copy_mat(mat *dst, const mat *src)
{
	int n;

	if (!dst || !src || !dst->data || !src->data)
		return;
	if (dst->rows != src->rows || dst->cols != src->cols)
		return;
	n = dst->rows * dst->cols;
	memcpy(dst->data, src->data, (size_t)n * sizeof(double));
}

static void copy_vec(vec *dst, const vec *src)
{
	if (!dst || !src || !dst->data || !src->data || dst->n != src->n)
		return;
	memcpy(dst->data, src->data, (size_t)src->n * sizeof(double));
}

static void copy_block(tblock *dst, const tblock *src)
{
	copy_mat(dst->attn->wq, src->attn->wq);
	copy_mat(dst->attn->wk, src->attn->wk);
	copy_mat(dst->attn->wv, src->attn->wv);
	copy_mat(dst->attn->wo, src->attn->wo);
	copy_mat(dst->ffn->w1, src->ffn->w1);
	copy_vec(dst->ffn->b1, src->ffn->b1);
	copy_mat(dst->ffn->w2, src->ffn->w2);
	copy_vec(dst->ffn->b2, src->ffn->b2);
	copy_vec(dst->ln1->gamma, src->ln1->gamma);
	copy_vec(dst->ln1->beta, src->ln1->beta);
	copy_vec(dst->ln2->gamma, src->ln2->gamma);
	copy_vec(dst->ln2->beta, src->ln2->beta);
}

static void copy_lm_weights(lm *dst, const lm *src)
{
	int i;

	copy_mat(dst->embed->table, src->embed->table);
	for (i = 0; i < dst->tr->n_blocks; i++)
		copy_block(&dst->tr->blocks[i], &src->tr->blocks[i]);
	copy_mat(dst->w_head, src->w_head);
	copy_vec(dst->b_head, src->b_head);
}

static void add_lm_weights(lm *acc, const lm *src)
{
	int i;

	add_mat(acc->embed->table, src->embed->table);
	for (i = 0; i < acc->tr->n_blocks; i++)
		add_block(&acc->tr->blocks[i], &src->tr->blocks[i]);
	add_mat(acc->w_head, src->w_head);
	add_vec(acc->b_head, src->b_head);
}

static void scale_lm_weights(lm *m, double s)
{
	int i;

	scale_mat(m->embed->table, s);
	for (i = 0; i < m->tr->n_blocks; i++)
		scale_block(&m->tr->blocks[i], s);
	scale_mat(m->w_head, s);
	scale_vec(m->b_head, s);
}

int lm_merge_average(lm *const *models, int n, lm **out)
{
	lm *merged;
	int i;

	if (!models || !out || n < 2)
		return -1;
	for (i = 0; i < n; i++) {
		if (!models[i] || !models[i]->tr
		    || models[i]->tr->n_blocks != models[0]->tr->n_blocks
		    || models[i]->vocab_size != models[0]->vocab_size)
			return -1;
	}
	merged = lm_create();
	if (!merged)
		return -1;
	copy_lm_weights(merged, models[0]);
	for (i = 1; i < n; i++)
		add_lm_weights(merged, models[i]);
	scale_lm_weights(merged, 1.0 / (double)n);
	*out = merged;
	return 0;
}

lm *lm_load_checkpoint(const char *path, lm_adam **opt_out, int *epoch_out,
	int *step_out)
{
	FILE *fp;
	lm *model;
	lm_adam *opt;
	int has_adam;
	int epoch;
	int step;
	int t;
	int n;
	int i;

	if (!path)
		return NULL;
	fp = fopen(path, "rb");
	if (!fp)
		return NULL;
	model = lm_create();
	if (!model) {
		fclose(fp);
		return NULL;
	}
	if (read_weights(fp, model) != 0) {
		lm_free(&model);
		fclose(fp);
		return NULL;
	}
	if (read_i32(fp, &has_adam) != 0 || has_adam != 1) {
		lm_free(&model);
		fclose(fp);
		return NULL;
	}
	if (read_i32(fp, &epoch) != 0 || read_i32(fp, &step) != 0
	    || read_i32(fp, &t) != 0 || read_i32(fp, &n) != 0) {
		lm_free(&model);
		fclose(fp);
		return NULL;
	}
	opt = lm_adam_create(model);
	if (!opt || opt->n != n) {
		lm_adam_free(&opt);
		lm_free(&model);
		fclose(fp);
		return NULL;
	}
	opt->t = t;
	for (i = 0; i < n; i++) {
		if (read_f64(fp, &opt->slots[i].m) != 0
		    || read_f64(fp, &opt->slots[i].v) != 0) {
			lm_adam_free(&opt);
			lm_free(&model);
			fclose(fp);
			return NULL;
		}
	}
	fclose(fp);
	if (opt_out)
		*opt_out = opt;
	else
		lm_adam_free(&opt);
	if (epoch_out)
		*epoch_out = epoch;
	if (step_out)
		*step_out = step;
	return model;
}

lm *lm_load_any(const char *path)
{
	lm *model;
	lm_adam *opt;

	model = lm_load(path);
	if (model)
		return model;
	opt = NULL;
	model = lm_load_checkpoint(path, &opt, NULL, NULL);
	lm_adam_free(&opt);
	return model;
}
