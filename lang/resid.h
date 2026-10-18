/*
**	Residual stream helpers (pre-norm wiring)
*/
#ifndef RESID_H__
#define RESID_H__

#include "mat.h"

int	resid_add(mat *dst, const mat *src);
/* dst = a + b (one kernel when GPU); falls back to copy+add. */
int	resid_sum(mat *dst, const mat *a, const mat *b);
int	resid_copy(mat *dst, const mat *src);

#endif
