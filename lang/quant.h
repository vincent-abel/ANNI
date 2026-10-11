/*
**	Int8 weight quantization for inference
*/
#ifndef QUANT_H__
#define QUANT_H__

#include "mat.h"
#include "vec.h"
#include <stdint.h>

typedef struct	qmat {
	int8_t	*data;
	int	rows;
	int	cols;
	double	scale;
} qmat;

qmat	*qmat_from_mat(const mat *m);
void	qmat_free(qmat **q);
/* y[j] += scale * sum_i x[i] * q[i, j]  for q shaped (rows=x.n, cols=y.n) */
int	qmat_gemv_t(vec *y, const qmat *q, const ann_real *x);

#endif
