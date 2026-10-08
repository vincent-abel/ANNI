/*
**	Position-wise feed-forward block (Transformer MLP)
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
} ffn;

ffn	*ffn_create(void);
void	ffn_free(ffn **f);
int	ffn_forward(mat *x, const ffn *f);

#endif
