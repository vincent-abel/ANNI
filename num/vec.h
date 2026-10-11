/*
**	Vector Header
*/
#ifndef VEC_H__
#define VEC_H__

#include "real.h"

typedef struct	vec {
	ann_real	*data;
	int	n;
} vec;

vec	*vec_create(int n);
void	vec_free(vec **v);
void	vec_fill(vec *v, ann_real value);
int	vec_copy(vec *dst, const vec *src);
ann_real	vec_dot(const vec *a, const vec *b);
void	vec_axpy(vec *y, ann_real a, const vec *x);
void	vec_scale(vec *v, ann_real s);
int	vec_add(vec *out, const vec *a, const vec *b);

#endif
