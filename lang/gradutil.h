/*
**	Small helpers for LM gradient buffers
*/
#ifndef GRADUTIL_H__
#define GRADUTIL_H__

#include "mat.h"
#include "vec.h"

void	mat_zero(mat *m);
void	mat_zero_dev(mat *m);	/* zero host + device twin (state 2) */
void	vec_zero(vec *v);
void	mat_axpy(mat *y, double a, const mat *x);
int	mat_outer_add(mat *dw, const mat *a, const mat *b);
int	mat_mul_at(mat *c, const mat *a, const mat *b);
/* c = a * w^T (backward through y = x W); GPU OP_T first, else host transpose. */
int	mat_mul_wt(mat *c, const mat *a, const mat *w);
/* c += a w^T: one accumulating GEMM on the GPU, else mat_mul_wt into tmp and add. */
int	mat_mul_wt_add(mat *c, const mat *a, const mat *w, mat *tmp);
/* mat_mul_bt lives in mat.h */

#endif
