/*
**	Position-wise feed-forward block + backward
*/
#ifndef FFN_H__
#define FFN_H__

#include "lm_config.h"
#include "mat.h"
#include "vec.h"

typedef struct	ffn {
	mat	*w1;
	vec	*b1;
	mat	*w2;
	vec	*b2;
	mat	*dw1;
	vec	*db1;
	mat	*dw2;
	vec	*db2;
} ffn;

typedef struct	ffn_cache {
	mat	*x_in;
	mat	*pre_act;
	mat	*hidden;
	int	T;
} ffn_cache;

ffn	*ffn_create(void);
void	ffn_free(ffn **f);
void	ffn_zero_grad(ffn *f);
int	ffn_forward(mat *x, const ffn *f);
int	ffn_forward_cache(mat *out, const mat *x, ffn *f, ffn_cache *c);
int	ffn_backward(mat *dx, const mat *dout, ffn *f, const ffn_cache *c);

ffn_cache	*ffn_cache_create(int T);
void		ffn_cache_free(ffn_cache **c);

#endif
