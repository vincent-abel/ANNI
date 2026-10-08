/*
**	Vector Code
*/
#include "main.h"

vec *vec_create(int n)
{
	vec *v;

	if (n <= 0)
		return NULL;
	v = malloc(sizeof(vec));
	if (!v)
		return NULL;
	v->data = calloc((size_t)n, sizeof(double));
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
	free((*v)->data);
	free(*v);
	*v = NULL;
}

void vec_fill(vec *v, double value)
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

double vec_dot(const vec *a, const vec *b)
{
	double sum;
	int i;

	if (!a || !b || !a->data || !b->data || a->n != b->n)
		return 0.0;
	sum = 0.0;
	for (i = 0; i < a->n; i++)
		sum += a->data[i] * b->data[i];
	return sum;
}

void vec_axpy(vec *y, double a, const vec *x)
{
	int i;

	if (!y || !x || !y->data || !x->data || y->n != x->n)
		return;
	for (i = 0; i < y->n; i++)
		y->data[i] += a * x->data[i];
}

void vec_scale(vec *v, double s)
{
	int i;

	if (!v || !v->data)
		return;
	for (i = 0; i < v->n; i++)
		v->data[i] *= s;
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
