/*
**	Matrix Header
*/
#ifndef MAT_H__
#define MAT_H__

#include "vec.h"

typedef struct	mat {
	double	*data;		/* host */
	double	*d_data;	/* device twin (NULL if none) */
	int	rows;
	int	cols;
	int	on_device;	/* 1 if d_data is authoritative */
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

/* Device twin helpers (no-op / fail when GPU stub). */
int	mat_device_alloc(mat *m);
void	mat_device_free(mat *m);
int	mat_to_device(mat *m);		/* H2D; sets on_device */
int	mat_to_host(mat *m);		/* D2H; clears on_device */
int	mat_ensure_host(mat *m);	/* sync down if needed */
/* Alloc device twin if GPU present; ignore failure (CPU remains valid). */
void	mat_try_pin(mat *m);

#endif
