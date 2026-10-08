/*
**	Small helpers for LM gradient buffers
*/
#ifndef GRADUTIL_H__
#define GRADUTIL_H__

#include "mat.h"
#include "vec.h"

void	mat_zero(mat *m);
void	vec_zero(vec *v);
void	mat_axpy(mat *y, double a, const mat *x);
int	mat_outer_add(mat *dw, const mat *a, const mat *b);
int	mat_mul_at(mat *c, const mat *a, const mat *b);
/* mat_mul_bt lives in mat.h */

#endif
