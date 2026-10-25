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
	mat	**q_h;		/* n_heads of T x d_head scratch (thread-safe) */
	mat	**k_h;
	mat	**o_h;
	ann_real	*softmax_tmp;	/* length n_heads * T */
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

/*
** Token-parallel packed attention. x is (B*T) x D: QKV / Wo projections are
** fat GEMMs over all B*T rows, but scores/softmax/AV stay causal within each
** length-T sequence (B caches of T x T per head, never (B*T)^2). seq[b] only
** carries attn_w / head scratch (x_in/q/k/v/concat are NULL). On a GPU build
** the same per-sequence weights live in w_all and one fused kernel launch
** (gpu_attn_causal_fwd/bwd, M1) covers all (b, h); seq[] is then only built
** as the fallback when the fused path is unavailable.
*/
typedef struct	attn_pcache {
	mat		*x_in;		/* (B*T) x D */
	mat		*q;
	mat		*k;
	mat		*v;
	mat		*concat;
	attn_cache	**seq;		/* B per-sequence score caches (host path) */
	mat		*w_all;		/* (B*H*T) x T weights, fused GPU path */
	mat		*lse;		/* (B*H*T) x 1 causal logsumexp, not T×T */
	mat		*dq;		/* backward scratch, lazily created */
	mat		*dk;
	mat		*dv;
	mat		*dconcat;
	mat		*dtmp;
	int		fused;		/* last forward used gpu_attn_causal_fwd */
	int		x_in_alias;	/* x_in points at the caller's input, not owned */
	int		B;
	int		T;		/* per-sequence length */
	int		n;		/* B*T */
	int		n_heads;
	int		d_head;
} attn_pcache;

/* S7: skip storing B·H·T·T weights; recompute in backward. */
void		attn_set_flash(int on);
int		attn_flash_enabled(void);
attn_pcache	*attn_pcache_create(int B, int T, int n_heads, int d_head);
void		attn_pcache_free(attn_pcache **c);
/* Same, with the block residual fused into the output projection (res may be NULL). */
int		attn_mha_forward_packed_res(mat *out, attn *a, const mat *x,
		attn_pcache *c, const mat *res);
int		attn_mha_forward_packed(mat *out, attn *a, const mat *x,
			attn_pcache *c);
int		attn_mha_backward_packed(mat *dx, const mat *dout, attn *a,
			const attn_pcache *c);

#endif
