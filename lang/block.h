/*
**	Transformer decoder block + stack
*/
#ifndef BLOCK_H__
#define BLOCK_H__

#include "attn.h"
#include "ffn.h"
#include "layernorm.h"
#include "lm_config.h"
#include "mat.h"

typedef struct	tblock {
	layernorm	*ln1;
	attn		*attn;
	layernorm	*ln2;
	ffn		*ffn;
} tblock;

typedef struct	transformer {
	tblock	*blocks;
	int	n_blocks;
} transformer;

tblock		*tblock_create(void);
void		tblock_free(tblock **b);
int		tblock_forward(mat *x, tblock *b);

transformer	*transformer_create(void);
void		transformer_free(transformer **t);
int		transformer_forward(mat *x, transformer *t);

#endif
