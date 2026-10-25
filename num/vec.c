/*
**	Vector Code
*/
#include "main.h"
#include "simd.h"
#include "gpu.h"

vec *vec_create(int n)
{
	vec *v;

	if (n <= 0)
		return NULL;
	v = malloc(sizeof(vec));
	if (!v)
		return NULL;
	v->data = calloc((size_t)n, sizeof(ann_real));
	if (!v->data) {
		free(v);
		return NULL;
	}
	v->n = n;
	return v;
}

void vec_free(vec **v)
{
	if (!v || !*v)
		return;
	gpu_host_forget((*v)->data);
	free((*v)->data);
	free(*v);
	*v = NULL;
}

void vec_fill(vec *v, ann_real value)
{
	int i;

	if (!v || !v->data)
		return;
	for (i = 0; i < v->n; i++)
		v->data[i] = value;
}

int vec_copy(vec *dst, const vec *src)
{
	int i;

	if (!dst || !src || !dst->data || !src->data || dst->n != src->n)
		return -1;
	for (i = 0; i < src->n; i++)
		dst->data[i] = src->data[i];
	return 0;
}

ann_real vec_dot(const vec *a, const vec *b)
{
	if (!a || !b || !a->data || !b->data || a->n != b->n)
		return 0.0f;
	return ann_dot(a->data, b->data, a->n);
}

void vec_axpy(vec *y, ann_real a, const vec *x)
{
	if (!y || !x || !y->data || !x->data || y->n != x->n)
		return;
	ann_axpy(y->data, a, x->data, y->n);
}

void vec_scale(vec *v, ann_real s)
{
	if (!v || !v->data)
		return;
	ann_scale(v->data, s, v->n);
}

int vec_add(vec *out, const vec *a, const vec *b)
{
	int i;

	if (!out || !a || !b || !out->data || !a->data || !b->data)
		return -1;
	if (out->n != a->n || a->n != b->n)
		return -1;
	for (i = 0; i < a->n; i++)
		out->data[i] = a->data[i] + b->data[i];
	return 0;
}
