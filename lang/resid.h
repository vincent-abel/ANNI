/*
**	Residual stream helpers (pre-norm wiring)
*/
#ifndef RESID_H__
#define RESID_H__

#include "mat.h"

int	resid_add(mat *dst, const mat *src);
int	resid_copy(mat *dst, const mat *src);

#endif
