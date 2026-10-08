/*
**	Attention: QKV, self-attn, multi-head, causal
*/
#ifndef ATTN_H__
#define ATTN_H__

#include "lm_config.h"
#include "mat.h"

typedef struct	attn {
	mat	*wq;
	mat	*wk;
	mat	*wv;
	mat	*wo;
	int	n_heads;
	int	d_head;
} attn;

attn	*attn_create(void);
void	attn_free(attn **a);
int	attn_qkv_project(mat *q, mat *k, mat *v, const attn *a, const mat *x);
int	attn_self(mat *out, const mat *q, const mat *k, const mat *v, int causal);
int	attn_mha(mat *out, const attn *a, const mat *x, int causal);

#endif
