/*
**	Transformer decoder block (pre-norm causal MHA + FFN)
*/
#include <stdlib.h>
#include <string.h>
#include "block.h"
#include "resid.h"

tblock *tblock_create(void)
{
	tblock *b;

	b = malloc(sizeof(tblock));
	if (!b)
		return NULL;
	b->ln1 = layernorm_create();
	b->attn = attn_create();
	b->ln2 = layernorm_create();
	b->ffn = ffn_create();
	if (!b->ln1 || !b->attn || !b->ln2 || !b->ffn) {
		tblock_free(&b);
		return NULL;
	}
	return b;
}

void tblock_free(tblock **b)
{
	if (!b || !*b)
		return;
	layernorm_free(&(*b)->ln1);
	attn_free(&(*b)->attn);
	layernorm_free(&(*b)->ln2);
	ffn_free(&(*b)->ffn);
	free(*b);
	*b = NULL;
}

int tblock_forward(mat *x, tblock *b)
{
	mat *tmp;
	mat *sub;
	int T;

	if (!x || !b || !x->data)
		return -1;
	if (x->cols != LM_D_MODEL)
		return -1;
	T = x->rows;
	tmp = mat_create(T, LM_D_MODEL);
	sub = mat_create(T, LM_D_MODEL);
	if (!tmp || !sub) {
		mat_free(&tmp);
		mat_free(&sub);
		return -1;
	}
	if (resid_copy(tmp, x) != 0)
		goto fail;
	if (layernorm_forward(tmp, b->ln1) != 0)
		goto fail;
	if (attn_mha(sub, b->attn, tmp, 1) != 0)
		goto fail;
	if (resid_add(x, sub) != 0)
		goto fail;

	if (resid_copy(tmp, x) != 0)
		goto fail;
	if (layernorm_forward(tmp, b->ln2) != 0)
		goto fail;
	if (ffn_forward(tmp, b->ffn) != 0)
		goto fail;
	if (resid_add(x, tmp) != 0)
		goto fail;

	mat_free(&tmp);
	mat_free(&sub);
	return 0;
fail:
	mat_free(&tmp);
	mat_free(&sub);
	return -1;
}

transformer *transformer_create(void)
{
	transformer *t;
	int i;

	t = malloc(sizeof(transformer));
	if (!t)
		return NULL;
	t->n_blocks = LM_N_BLOCKS;
	t->blocks = calloc((size_t)LM_N_BLOCKS, sizeof(tblock));
	if (!t->blocks) {
		free(t);
		return NULL;
	}
	for (i = 0; i < LM_N_BLOCKS; i++) {
		t->blocks[i].ln1 = layernorm_create();
		t->blocks[i].attn = attn_create();
		t->blocks[i].ln2 = layernorm_create();
		t->blocks[i].ffn = ffn_create();
		if (!t->blocks[i].ln1 || !t->blocks[i].attn
		    || !t->blocks[i].ln2 || !t->blocks[i].ffn) {
			transformer_free(&t);
			return NULL;
		}
	}
	return t;
}

void transformer_free(transformer **t)
{
	int i;

	if (!t || !*t)
		return;
	if ((*t)->blocks) {
		for (i = 0; i < (*t)->n_blocks; i++) {
			layernorm_free(&(*t)->blocks[i].ln1);
			attn_free(&(*t)->blocks[i].attn);
			layernorm_free(&(*t)->blocks[i].ln2);
			ffn_free(&(*t)->blocks[i].ffn);
		}
		free((*t)->blocks);
	}
	free(*t);
	*t = NULL;
}

int transformer_forward(mat *x, transformer *t)
{
	int i;

	if (!x || !t || !t->blocks)
		return -1;
	for (i = 0; i < t->n_blocks; i++) {
		if (tblock_forward(x, &t->blocks[i]) != 0)
			return -1;
	}
	return 0;
}
