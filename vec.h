/*
**	Vector Header
*/
#ifndef VEC_H__
#define VEC_H__

typedef struct	vec {
	double	*data;
	int	n;
} vec;

vec	*vec_create(int n);
void	vec_free(vec **v);
void	vec_fill(vec *v, double value);
int	vec_copy(vec *dst, const vec *src);
double	vec_dot(const vec *a, const vec *b);
void	vec_axpy(vec *y, double a, const vec *x);
void	vec_scale(vec *v, double s);
int	vec_add(vec *out, const vec *a, const vec *b);

#endif
