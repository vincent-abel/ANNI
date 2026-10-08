/*
**	Optimizer Code
*/
#include "main.h"

static int count_params(network *net)
{
	nlayer *layer;
	neuron *n;
	input *in;
	int c;

	c = 0;
	layer = net->layers;
	while (layer) {
		n = layer->neuron;
		while (n) {
			c++;
			in = n->input;
			while (in) {
				c++;
				in = in->prev;
			}
			n = n->prev;
		}
		layer = layer->next;
	}
	return c;
}

adam *adam_create(network *net)
{
	adam *opt;
	nlayer *layer;
	neuron *n;
	input *in;
	int i;
	int n_params;

	if (!net)
		return NULL;
	n_params = count_params(net);
	if (n_params <= 0)
		return NULL;
	opt = malloc(sizeof(adam));
	if (!opt)
		return NULL;
	opt->slots = calloc((size_t)n_params, sizeof(adam_slot));
	if (!opt->slots) {
		free(opt);
		return NULL;
	}
	opt->n = n_params;
	opt->t = 0;
	opt->beta1 = 0.9;
	opt->beta2 = 0.999;
	opt->eps = 1e-8;

	i = 0;
	layer = net->layers;
	while (layer) {
		n = layer->neuron;
		while (n) {
			opt->slots[i].param = &n->bias;
			opt->slots[i].grad = &n->dbias;
			opt->slots[i].m = 0.0;
			opt->slots[i].v = 0.0;
			i++;
			in = n->input;
			while (in) {
				opt->slots[i].param = &in->w;
				opt->slots[i].grad = &in->dw;
				opt->slots[i].m = 0.0;
				opt->slots[i].v = 0.0;
				i++;
				in = in->prev;
			}
			n = n->prev;
		}
		layer = layer->next;
	}
	return opt;
}

void adam_free(adam **opt)
{
	if (!opt || !*opt)
		return;
	free((*opt)->slots);
	free(*opt);
	*opt = NULL;
}

void adam_step(adam *opt, double lr)
{
	int i;
	double g;
	double mhat;
	double vhat;
	double bc1;
	double bc2;

	if (!opt || !opt->slots || opt->n <= 0)
		return;

	opt->t++;
	bc1 = 1.0 - pow(opt->beta1, (double)opt->t);
	bc2 = 1.0 - pow(opt->beta2, (double)opt->t);
	if (bc1 == 0.0 || bc2 == 0.0)
		return;

	for (i = 0; i < opt->n; i++) {
		g = *(opt->slots[i].grad);
		opt->slots[i].m = opt->beta1 * opt->slots[i].m
			+ (1.0 - opt->beta1) * g;
		opt->slots[i].v = opt->beta2 * opt->slots[i].v
			+ (1.0 - opt->beta2) * g * g;
		mhat = opt->slots[i].m / bc1;
		vhat = opt->slots[i].v / bc2;
		*(opt->slots[i].param) -= lr * mhat / (sqrt(vhat) + opt->eps);
		*(opt->slots[i].grad) = 0.0;
	}
}

void network_train_adam(network *net, adam *opt, const double *samples,
	const double *targets, int n_samples, int n_target,
	int epochs, double lr)
{
	int e;
	int s;

	if (!net || !opt || !samples || !targets || n_samples <= 0
		|| n_target <= 0 || epochs <= 0)
		return;

	for (e = 0; e < epochs; e++) {
		for (s = 0; s < n_samples; s++) {
			network_set_inputs(net, samples + s * net->n_inputs,
				net->n_inputs);
			network_forward(net);
			network_backward(net, targets + s * n_target, n_target);
			adam_step(opt, lr);
		}
	}
}
