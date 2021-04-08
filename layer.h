/*
**	Layer Header
*/
#ifndef LAYER_H__
#define LAYER_H__

#include "neuron.h"

typedef struct	nlayer {
	struct	neuron *neuron;
	struct	nlayer *prev;
} nlayer;

typedef	struct	brain {
	void**	neuron_layer;
} brain;

void 	nlayer_show(nlayer *nlayer);
void	nlayer_setInput(nlayer *);
void nlayer_push(nlayer **p);

#endif 