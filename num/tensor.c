/*
**	Tensor Code
*/
#include "main.h"

static int shape_size(const int *shape, int ndim)
{
	int i;
	int size;

	if (!shape || ndim <= 0)
		return -1;
	size = 1;
	for (i = 0; i < ndim; i++) {
		if (shape[i] <= 0)
			return -1;
		size *= shape[i];
	}
	return size;
}

tensor *tensor_create(const int *shape, int ndim)
{
	tensor *t;
	int i;
	int size;

	size = shape_size(shape, ndim);
	if (size < 0)
		return NULL;
	t = malloc(sizeof(tensor));
	if (!t)
		return NULL;
	t->shape = malloc((size_t)ndim * sizeof(int));
	t->data = calloc((size_t)size, sizeof(ann_real));
	if (!t->shape || !t->data) {
		free(t->shape);
		free(t->data);
		free(t);
		return NULL;
	}
	for (i = 0; i < ndim; i++)
		t->shape[i] = shape[i];
	t->ndim = ndim;
	t->size = size;
	return t;
}

void tensor_free(tensor **t)
{
	if (!t || !*t)
		return;
	free((*t)->data);
	free((*t)->shape);
	free(*t);
	*t = NULL;
}

void tensor_fill(tensor *t, ann_real value)
{
	int i;

	if (!t || !t->data)
		return;
	for (i = 0; i < t->size; i++)
		t->data[i] = value;
}

ann_real *tensor_at(tensor *t, const int *idx)
{
	int i;
	int flat;
	int stride;

	if (!t || !t->data || !t->shape || !idx)
		return NULL;
	flat = 0;
	stride = 1;
	for (i = t->ndim - 1; i >= 0; i--) {
		if (idx[i] < 0 || idx[i] >= t->shape[i])
			return NULL;
		flat += idx[i] * stride;
		stride *= t->shape[i];
	}
	return &t->data[flat];
}

int tensor_as_vec(vec *view, tensor *t)
{
	if (!view || !t || !t->data || t->ndim != 1)
		return -1;
	view->data = t->data;
	view->n = t->shape[0];
	return 0;
}

int tensor_as_mat(mat *view, tensor *t)
{
	if (!view || !t || !t->data || t->ndim != 2)
		return -1;
	view->data = t->data;
	view->d_data = NULL;
	view->rows = t->shape[0];
	view->cols = t->shape[1];
	view->on_device = 0;
	return 0;
}
