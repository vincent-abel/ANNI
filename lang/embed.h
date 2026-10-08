/*
**	Token embedding table
*/
#ifndef EMBED_H__
#define EMBED_H__

#include "lm_config.h"
#include "mat.h"
#include "vocab.h"

typedef struct	embed {
	mat	*table;
} embed;

embed	*embed_create(int vocab_size);
void	embed_free(embed **e);
int	embed_lookup(mat *out, const embed *e, const int *ids, int n);

#endif
