/*
**	Matrix Header
*/
#ifndef MAT_H__
#define MAT_H__

#include "real.h"
#include "vec.h"

typedef struct	mat {
	ann_real	*data;		/* host */
	ann_real	*d_data;	/* device twin (NULL if none) */
	int	rows;
	int	cols;
	int	on_device;	/* MAT_DEV_* residency state (see below) */
} mat;

/*
** on_device: 0 = host authoritative (d_data, if any, may be stale);
** 1 = device authoritative (host may be stale: mat_ensure_host first);
** 2 = host and device twin identical (used for weights between optimizer
** steps). Any host write must leave the mat at 0 (see mat_host_dirty).
*/
#define MAT_DEV_HOST	0
#define MAT_DEV_DEVICE	1
#define MAT_DEV_SYNCED	2

mat	*mat_create(int rows, int cols);
void	mat_free(mat **m);
ann_real	mat_get(const mat *m, int row, int col);
void	mat_set(mat *m, int row, int col, ann_real value);
void	mat_fill(mat *m, ann_real value);
ann_real	*mat_row(mat *m, int row);
const ann_real	*mat_row_const(const mat *m, int row);
int	mat_vec_mul(vec *y, const mat *a, const vec *x);
int	mat_mul(mat *c, const mat *a, const mat *b);
int	mat_mul_naive(mat *c, const mat *a, const mat *b);
int	mat_mul_bt(mat *c, const mat *a, const mat *b);

/* Device twin helpers (no-op / fail when GPU stub). */
int	mat_device_alloc(mat *m);
void	mat_device_free(mat *m);
int	mat_to_device(mat *m);		/* H2D; sets on_device */
int	mat_to_host(mat *m);		/* D2H; clears on_device */
int	mat_ensure_host(mat *m);	/* D2H only when device-authoritative */
int	mat_sync_device(mat *m);	/* H2D; host+device identical (state 2) */
/* D2H only when device-authoritative; result is state 2 (host == twin).
** Use for read-only host access; use mat_ensure_host if the host will write. */
int	mat_pull_host(mat *m);
void	mat_host_dirty(mat *m);		/* call after host writes: state 0 */
/* Alloc device twin if GPU present; ignore failure (CPU remains valid). */
void	mat_try_pin(mat *m);
/* Device twin when authoritative or synced; else host (caller may stage). */
ann_real	*mat_compute_ptr(mat *m);
const ann_real	*mat_compute_ptr_const(const mat *m);
/* After an in-place device write through mat_compute_ptr / d_data. */
void	mat_mark_device(mat *m);

#endif
