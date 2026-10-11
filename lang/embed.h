/*
**	Token embedding table + backward
*/
#ifndef EMBED_H__
#define EMBED_H__

#include "lm_config.h"
#include "mat.h"
#include "vocab.h"

typedef struct	embed {
	mat	*table;
	mat	*dtable;
} embed;

embed	*embed_create(int vocab_size);
void	embed_free(embed **e);
void	embed_zero_grad(embed *e);
int	embed_lookup(mat *out, const embed *e, const int *ids, int n);
int	embed_lookup_row(ann_real *out, const embed *e, int id);
int	embed_backward(embed *e, const mat *dout, const int *ids, int n);

#endif
