/*
**	Adam optimizer over flat LM param/grad slots
*/
#ifndef LM_ADAM_H__
#define LM_ADAM_H__

#include "lm.h"

typedef struct	lm_adam_slot {
	ann_real	*param;
	ann_real	*grad;
	ann_real	m;
	ann_real	v;
} lm_adam_slot;

/*
** One contiguous run of the flat param/grad vectors: a weight mat (+ its dW)
** or a bias/LN vec. `off` is the element offset into d_param/d_grad/d_m/d_v
** (byte offset = off * sizeof(ann_real)), so each run can be gathered from /
** scattered to its mat's d_data with device-to-device copies only.
*/
typedef struct	lm_adam_seg {
	mat	*w;
	mat	*dw;
	vec	*wv;
	vec	*dwv;
	int	off;
	int	n;
	int	host_use;	/* host code reads/writes it every step (embed table) */
} lm_adam_seg;

typedef struct	lm_adam {
	lm_adam_slot	*slots;
	lm_adam_seg	*segs;
	int	n_segs;
	int	n;
	int	t;
	double	beta1;
	double	beta2;
	double	eps;
	/* Persistent GPU state (lazy; NULL on CPU builds). */
	ann_real	*h_m;	/* host staging for m/v checkpoint sync only */
	ann_real	*h_v;
	ann_real	*d_param;	/* A1: source of truth between steps */
	void		*d_param_bf16;	/* Adam-fused BF16 twins (GEMM path) */
	ann_real	*d_grad;
	ann_real	*d_m;	/* device m/v stay resident across steps */
	ann_real	*d_v;
	int	dev_mv;	/* 1: d_m/d_v newer than slots[].m/v */
	int	params_valid;	/* 1: d_param already mirrors the weights */
	int	offload_mv;	/* S13: keep m/v on host between steps */
	double	weight_decay;	/* E2 AdamW-style; 0 = off */
	double	grad_clip;	/* E2 global L2 clip; 0 = off */
} lm_adam;

lm_adam	*lm_adam_create(lm *model);
/* Total learnable scalars (embed + blocks + head), matching Adam slot count. */
int	lm_param_count(const lm *model);
void	lm_adam_free(lm_adam **opt);
void	lm_adam_zero_grad(lm *model);
void	lm_adam_step(lm_adam *opt, double lr);
/*
** Adam with g = grad_scale * grad (e.g. 1/B), folded into the update.
** GPU path: device-authoritative dW is gathered D2D into d_grad, the update
** runs on d_param/d_m/d_v, and d_param is scattered D2D back into each
** weight's d_data (weight left device-authoritative; no D2H). Only host-side
** consumers (embedding table, bias/LN vecs) are copied back per step.
*/
void	lm_adam_step_scaled(lm_adam *opt, double lr, double grad_scale);
/* Force the next step to re-read weights (call after host rewrites them). */
void	lm_adam_invalidate(lm_adam *opt);
/*
** Pull device-resident m/v into slots and make host weights current
** (call before checkpoint save).
*/
void	lm_adam_sync_host(lm_adam *opt);
/* S13: when on, pull m/v to host and free device m/v after each step. */
void	lm_adam_set_offload(int on);

#endif
