/*
**	Layer normalization (per token row)
*/
#ifndef LAYERNORM_H__
#define LAYERNORM_H__

#include "lm_config.h"
#include "mat.h"
#include "vec.h"

typedef struct	layernorm {
	vec	*gamma;
	vec	*beta;
	double	eps;
} layernorm;

layernorm	*layernorm_create(void);
void		layernorm_free(layernorm **ln);
int		layernorm_forward(mat *x, const layernorm *ln);

#endif
