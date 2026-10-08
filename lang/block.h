/*
**	Transformer decoder block + stack + backward caches
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

typedef struct	block_cache {
	mat		*x_in;
	mat		*ln1_out;
	mat		*attn_out;
	mat		*x_mid;
	mat		*ln2_out;
	mat		*ffn_out;
	ln_cache	*ln1c;
	attn_cache	*attnc;
	ln_cache	*ln2c;
	ffn_cache	*ffnc;
	int		T;
} block_cache;

typedef struct	transformer {
	tblock	*blocks;
	int	n_blocks;
} transformer;

typedef struct	transformer_cache {
	block_cache	*bc;
	int		n_blocks;
	int		T;
} transformer_cache;

tblock		*tblock_create(void);
void		tblock_free(tblock **b);
void		tblock_zero_grad(tblock *b);
int		tblock_forward(mat *x, tblock *b);
int		tblock_forward_cache(mat *x, tblock *b, block_cache *c);
int		tblock_backward(mat *dx, const mat *dout, tblock *b,
			const block_cache *c);

block_cache	*block_cache_create(int T, const attn *a);
void		block_cache_free(block_cache **c);

transformer	*transformer_create(void);
void		transformer_free(transformer **t);
void		transformer_zero_grad(transformer *t);
int		transformer_forward(mat *x, transformer *t);
int		transformer_forward_cache(mat *x, transformer *t,
			transformer_cache *c);
int		transformer_backward(mat *dx, const mat *dout, transformer *t,
			const transformer_cache *c);

transformer_cache	*transformer_cache_create(int T, const transformer *t);
void			transformer_cache_free(transformer_cache **c);

#endif
