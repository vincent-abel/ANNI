/*
**	Adam optimizer over flat LM param/grad slots
*/
#ifndef LM_ADAM_H__
#define LM_ADAM_H__

#include "lm.h"

typedef struct	lm_adam_slot {
	double	*param;
	double	*grad;
	double	m;
	double	v;
} lm_adam_slot;

typedef struct	lm_adam {
	lm_adam_slot	*slots;
	int	n;
	int	t;
	double	beta1;
	double	beta2;
	double	eps;
} lm_adam;

lm_adam	*lm_adam_create(lm *model);
void	lm_adam_free(lm_adam **opt);
void	lm_adam_zero_grad(lm *model);
void	lm_adam_step(lm_adam *opt, double lr);

#endif
