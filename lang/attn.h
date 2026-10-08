/*
**	Attention: QKV, self-attn, multi-head, causal + backward
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
	mat	*dwq;
	mat	*dwk;
	mat	*dwv;
	mat	*dwo;
	int	n_heads;
	int	d_head;
} attn;

typedef struct	attn_cache {
	mat	*x_in;
	mat	*q;
	mat	*k;
	mat	*v;
	mat	*concat;
	mat	**attn_w;	/* n_heads of T x T softmax weights */
	mat	**v_h;		/* n_heads of T x d_head */
	int	T;
	int	n_heads;
	int	d_head;
	int	causal;
} attn_cache;

attn	*attn_create(void);
void	attn_free(attn **a);
void	attn_zero_grad(attn *a);
int	attn_qkv_project(mat *q, mat *k, mat *v, const attn *a, const mat *x);
int	attn_self(mat *out, const mat *q, const mat *k, const mat *v, int causal);
int	attn_mha(mat *out, const attn *a, const mat *x, int causal);
int	attn_mha_forward_cache(mat *out, attn *a, const mat *x, int causal,
		attn_cache *c);
int	attn_mha_backward(mat *dx, const mat *dout, attn *a, const attn_cache *c);

attn_cache	*attn_cache_create(int T, int n_heads, int d_head);
void		attn_cache_free(attn_cache **c);

#endif
