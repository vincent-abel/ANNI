/*
**	Layer Header
*/
#ifndef LAYER_H__
#define LAYER_H__

#include "neuron.h"

typedef struct	nlayer {
	struct	neuron *neuron;
	struct	nlayer *next;
	struct	nlayer *prev;
} nlayer;

typedef struct	network {
	double	*inputs;
	int	n_inputs;
	nlayer	*layers;
	nlayer	*out;
} network;

void	nlayer_show(nlayer *);
void	nlayer_append(nlayer **, nlayer **, neuron *);
void	nlayer_clear(nlayer **);
void	nlayer_forward(nlayer *);
network	*network_create(int n_inputs, int n_neurons, int n_layers);
void	network_forward(network *);
void	network_show(network *);
double	network_mse(network *, const double *target, int n_target);
void	network_backward(network *, const double *target, int n_target);
void	network_update(network *, double lr);
void	network_set_inputs(network *, const double *values, int n_values);
void	network_train(network *, const double *samples, const double *targets,
		int n_samples, int n_target, int epochs, double lr);
int		network_output_size(network *);
void	network_free(network **);

#endif
