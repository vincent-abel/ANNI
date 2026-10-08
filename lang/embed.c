/*
**	Token embedding table + backward
*/
#include <stdlib.h>
#include <string.h>
#include "embed.h"
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
	return e;
}

void embed_free(embed **e)
{
	if (!e || !*e)
		return;
	mat_free(&(*e)->table);
	mat_free(&(*e)->dtable);
	free(*e);
	*e = NULL;
}

void embed_zero_grad(embed *e)
{
	if (!e)
		return;
	mat_zero(e->dtable);
}

int embed_lookup(mat *out, const embed *e, const int *ids, int n)
{
	int i;
	int id;

	if (!out || !e || !e->table || !ids || n <= 0)
		return -1;
	if (out->rows != n || out->cols != LM_D_MODEL)
		return -1;
	for (i = 0; i < n; i++) {
		id = ids[i];
		if (id < 0 || id >= e->table->rows)
			return -1;
		memcpy(&out->data[i * LM_D_MODEL],
		    &e->table->data[id * LM_D_MODEL],
		    (size_t)LM_D_MODEL * sizeof(double));
	}
	return 0;
}

int embed_backward(embed *e, const mat *dout, const int *ids, int n)
{
	int i;
	int j;
	int id;
	double *row;

	if (!e || !dout || !ids || n <= 0)
		return -1;
	if (dout->rows != n || dout->cols != LM_D_MODEL)
		return -1;
	for (i = 0; i < n; i++) {
		id = ids[i];
		if (id < 0 || id >= e->dtable->rows)
			return -1;
		row = &e->dtable->data[id * LM_D_MODEL];
		for (j = 0; j < LM_D_MODEL; j++)
			row[j] += dout->data[i * LM_D_MODEL + j];
	}
	return 0;
}
