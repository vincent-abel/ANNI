/*
**	Layer normalization (per token row) + backward
*/
#ifndef LAYERNORM_H__
#define LAYERNORM_H__

#include "lm_config.h"
#include "mat.h"
#include "vec.h"

typedef struct	layernorm {
	vec	*gamma;
	vec	*beta;
	vec	*dgamma;
	vec	*dbeta;
	double	eps;
} layernorm;

typedef struct	ln_cache {
	mat	*x_hat;
	vec	*mean;
	vec	*inv_std;
	int	T;
} ln_cache;

layernorm	*layernorm_create(void);
void		layernorm_free(layernorm **ln);
void		layernorm_zero_grad(layernorm *ln);
int		layernorm_forward(mat *x, const layernorm *ln);
int		layernorm_forward_cache(mat *x, layernorm *ln, ln_cache *c);
/* out = LN(in); avoids copy_mat(out, in) before in-place LN. */
int		layernorm_forward_into(mat *out, const mat *in, layernorm *ln,
			ln_cache *c);
int		layernorm_backward(mat *dx, const mat *dy, layernorm *ln,
			const ln_cache *c);

ln_cache	*ln_cache_create(int T);
void		ln_cache_free(ln_cache **c);

#endif
