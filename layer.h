/*
**	Layer Header
*/
#ifndef LAYER_H__
#define LAYER_H__

#include "neuron.h"

typedef struct	nlayer {
	struct	neuron *neuron;
	struct	nlayer *next;
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
int		network_output_size(network *);
void	network_free(network **);

#endif
