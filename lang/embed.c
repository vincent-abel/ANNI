/*
**	Token embedding table + backward
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "embed.h"
#include "gpu.h"
#include "gradutil.h"

static double embed_rand_weight(void)
{
	double u;

	u = (double)rand() / (double)RAND_MAX;
	return (u * 2.0 - 1.0) * 0.02;
}

embed *embed_create(int vocab_size)
{
	embed *e;
	int i;
	int n;

	if (vocab_size <= 0)
		return NULL;
	e = malloc(sizeof(embed));
	if (!e)
		return NULL;
	e->tied = 0;
	e->scale = 1.0;
	e->table = mat_create(vocab_size, LM_D_MODEL);
	e->dtable = mat_create(vocab_size, LM_D_MODEL);
	if (!e->table || !e->dtable) {
		embed_free(&e);
		return NULL;
	}
	n = e->table->rows * e->table->cols;
	for (i = 0; i < n; i++)
		e->table->data[i] = embed_rand_weight();
	embed_zero_grad(e);
	mat_try_pin(e->table);
	mat_try_pin(e->dtable);
	return e;
}

embed *embed_create_tied(mat *w_head, mat *dw_head)
{
	embed *e;

	if (!w_head || !dw_head)
		return NULL;
	e = malloc(sizeof(embed));
	if (!e)
		return NULL;
	e->table = w_head;
	e->dtable = dw_head;
	e->tied = 1;
	e->scale = (ann_real)sqrt((double)LM_D_MODEL);
	return e;
}

void embed_free(embed **e)
{
	if (!e || !*e)
		return;
	if (!(*e)->tied) {
		mat_free(&(*e)->table);
		mat_free(&(*e)->dtable);
	}
	free(*e);
	*e = NULL;
}

void embed_zero_grad(embed *e)
{
	if (!e || e->tied)	/* the head clears its own grad */
		return;
	mat_zero_dev(e->dtable);
}

static int embed_vocab(const embed *e)
{
	return e->tied ? e->table->cols : e->table->rows;
}

static void gather_row(ann_real *out, const embed *e, int id)
{
	size_t V;
	int j;

	if (!e->tied) {
		memcpy(out, &e->table->data[(size_t)id * LM_D_MODEL],
		    (size_t)LM_D_MODEL * sizeof(ann_real));
		return;
	}
	V = (size_t)e->table->cols;
	for (j = 0; j < LM_D_MODEL; j++)
		out[j] = e->table->data[(size_t)j * V + (size_t)id] * e->scale;
}

int embed_lookup_row(ann_real *out, const embed *e, int id)
{
	if (!out || !e || !e->table || id < 0 || id >= embed_vocab(e))
		return -1;
	(void)mat_ensure_host((mat *)e->table);
	gather_row(out, e, id);
	return 0;
}

int embed_lookup(mat *out, const embed *e, const int *ids, int n)
{
	int i;
	int id;

	if (!out || !e || !e->table || !ids || n <= 0)
		return -1;
	if (out->rows != n || out->cols != LM_D_MODEL)
		return -1;
	mat_try_pin(out);
	if (gpu_embed_lookup(out, e->table, ids, n, e->tied, e->scale) == 0)
		return 0;
	/* Device-only mats have NULL host data — allocate before CPU gather. */
	if (mat_ensure_host(out) != 0 || mat_ensure_host((mat *)e->table) != 0)
		return -1;
	for (i = 0; i < n; i++) {
		id = ids[i];
		if (id < 0 || id >= embed_vocab(e))
			return -1;
		gather_row(&out->data[i * LM_D_MODEL], e, id);
	}
	mat_host_dirty(out);
	return 0;
}

int embed_backward(embed *e, const mat *dout, const int *ids, int n)
{
	int i;
	int j;
	int id;
	ann_real *row;

	if (!e || !dout || !ids || n <= 0)
		return -1;
	if (dout->rows != n || dout->cols != LM_D_MODEL)
		return -1;
	if (gpu_embed_scatter(e->dtable, dout, ids, n, e->tied, e->scale) == 0)
		return 0;
	(void)mat_ensure_host((mat *)dout);
	(void)mat_ensure_host(e->dtable);
	for (i = 0; i < n; i++) {
		id = ids[i];
		if (id < 0 || id >= embed_vocab(e))
			return -1;
		if (e->tied) {
			size_t V = (size_t)e->dtable->cols;

			for (j = 0; j < LM_D_MODEL; j++)
				e->dtable->data[(size_t)j * V + (size_t)id]
				    += dout->data[i * LM_D_MODEL + j] * e->scale;
			continue;
		}
		row = &e->dtable->data[id * LM_D_MODEL];
		for (j = 0; j < LM_D_MODEL; j++)
			row[j] += dout->data[i * LM_D_MODEL + j];
	}
	mat_host_dirty(e->dtable);
	return 0;
}
