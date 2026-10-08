/*
**	Matrix Header
*/
#ifndef MAT_H__
#define MAT_H__

#include "vec.h"

typedef struct	mat {
	double	*data;
	int	rows;
	int	cols;
} mat;

mat	*mat_create(int rows, int cols);
void	mat_free(mat **m);
double	mat_get(const mat *m, int row, int col);
void	mat_set(mat *m, int row, int col, double value);
void	mat_fill(mat *m, double value);
double	*mat_row(mat *m, int row);
const double	*mat_row_const(const mat *m, int row);
int	mat_vec_mul(vec *y, const mat *a, const vec *x);
int	mat_mul(mat *c, const mat *a, const mat *b);
int	mat_mul_naive(mat *c, const mat *a, const mat *b);
int	mat_mul_bt(mat *c, const mat *a, const mat *b);

#endif
