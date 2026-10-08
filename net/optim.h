/*
**	Optimizer Header
*/
#ifndef OPTIM_H__
#define OPTIM_H__

#include "layer.h"

typedef struct	adam_slot {
	double	*param;
	double	*grad;
	double	m;
	double	v;
} adam_slot;

typedef struct	adam {
	adam_slot	*slots;
	int	n;
	int	t;
	double	beta1;
	double	beta2;
	double	eps;
} adam;

adam	*adam_create(network *net);
void	adam_free(adam **opt);
void	adam_step(adam *opt, double lr);
void	network_train_adam(network *net, adam *opt, const double *samples,
		const double *targets, int n_samples, int n_target,
		int epochs, double lr);

#endif
