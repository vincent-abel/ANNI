/*
**	Token embedding table
*/
#include <stdlib.h>
#include "embed.h"

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
	if (!e->table) {
		free(e);
		return NULL;
	}
	n = e->table->rows * e->table->cols;
	for (i = 0; i < n; i++)
		e->table->data[i] = embed_rand_weight();
	return e;
}

void embed_free(embed **e)
{
	if (!e || !*e)
		return;
	mat_free(&(*e)->table);
	free(*e);
	*e = NULL;
}

int embed_lookup(mat *out, const embed *e, const int *ids, int n)
{
	int i;
	int j;
	int id;

	if (!out || !e || !e->table || !ids || n <= 0)
		return -1;
	if (out->rows != n || out->cols != LM_D_MODEL)
		return -1;
	for (i = 0; i < n; i++) {
		id = ids[i];
		if (id < 0 || id >= e->table->rows)
			return -1;
		for (j = 0; j < LM_D_MODEL; j++)
			mat_set(out, i, j, mat_get(e->table, id, j));
	}
	return 0;
}
