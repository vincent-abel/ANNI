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
