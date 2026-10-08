/*
**	Layer Code
*/
#include "main.h"

static neuron *layer_neurons_create(int count, double *values, int n_values, neuron *prev_neurons)
{
	neuron *head = NULL;

	while (count--) {
		if (!neuron_push(&head)) {
			neuron_clear(&head);
			return NULL;
		}
		if (prev_neurons) {
			if (neuron_connect_layer(head, prev_neurons) != 0) {
				neuron_clear(&head);
				return NULL;
			}
		} else if (neuron_connect_values(head, values, n_values) != 0) {
			neuron_clear(&head);
			return NULL;
		}
	}
	return head;
}

int nlayer_append(nlayer **head, nlayer **tail, neuron *neurons)
{
	nlayer *layer = malloc(sizeof(nlayer));

	if (!layer)
		return -1;
	layer->neuron = neurons;
	layer->next = NULL;
	layer->prev = *tail;
	if (!*head)
		*head = layer;
	else
		(*tail)->next = layer;
	*tail = layer;
	return 0;
}

void nlayer_clear(nlayer **head)
{
	nlayer *tmp;

	while (*head) {
		neuron_clear(&(*head)->neuron);
		tmp = (*head)->next;
		free(*head);
		*head = tmp;
	}
}

void nlayer_forward(nlayer *layer)
{
	while (layer) {
		neuron_output(layer->neuron);
		layer = layer->next;
	}
}

static int count_neurons(neuron *n)
{
	int c = 0;

	while (n) {
		c++;
		n = n->prev;
	}
	return c;
}

static int count_layers(nlayer *layer)
{
	int c = 0;

	while (layer) {
		c++;
		layer = layer->next;
	}
	return c;
}

void nlayer_show(nlayer *layer)
{
	int i = 0;
	int n_layers = count_layers(layer);

	while (layer) {
		const char *role;

		i++;
		if (n_layers == 1)
			role = "input/output";
		else if (i == 1)
			role = "input";
		else if (!layer->next)
			role = "output";
		else
			role = "hidden";

		printf("\nLayer %d / %d  (%s, %d neurons)\n\n",
			i, n_layers, role, count_neurons(layer->neuron));
		neuron_show(layer->neuron);
		layer = layer->next;
	}
}

void network_show(network *net)
{
	int i;
	int n_layers;
	int n_neurons;

	if (!net || !net->layers)
		return;

	n_layers = count_layers(net->layers);
	n_neurons = count_neurons(net->layers->neuron);

	puts("");
	printf("ANN  inputs=%d  neurons=%d  layers=%d\n",
		net->n_inputs, n_neurons, n_layers);
	printf("sample");
	for (i = 0; i < net->n_inputs; i++)
		printf("  %10.4f", net->inputs[i]);
	puts("");
	nlayer_show(net->layers);
	puts("");
}

int network_output_size(network *net)
{
	if (!net || !net->out)
		return 0;
	return count_neurons(net->out->neuron);
}

double network_mse(network *net, const double *target, int n_target)
{
	neuron *n;
	double sum;
	double d;
	int i;

	if (!net || !net->out || !target || n_target <= 0)
		return -1.0;

	n = net->out->neuron;
	sum = 0.0;
	i = 0;
	while (n) {
		if (i >= n_target)
			return -1.0;
		d = n->output - target[i];
		sum += d * d;
		i++;
		n = n->prev;
	}
	if (i != n_target)
		return -1.0;
	return sum / (double)n_target;
}

void network_backward(network *net, const double *target, int n_target)
{
	nlayer *layer;

	if (!net || !net->out || !target || n_target <= 0)
		return;
	if (count_neurons(net->out->neuron) != n_target)
		return;

	layer = net->layers;
	while (layer) {
		neuron_zero_delta(layer->neuron);
		layer = layer->next;
	}

	layer = net->out;
	neuron_backward_output(layer->neuron, target, n_target);
	layer = layer->prev;
	while (layer) {
		neuron_backward_hidden(layer->neuron);
		layer = layer->prev;
	}
}

network *network_create_sizes(int n_inputs, const int *sizes, int n_layers)
{
	network *net;
	nlayer *tail = NULL;
	neuron *prev = NULL;
	int i;

	if (!sizes || n_inputs <= 0 || n_layers <= 0)
		return NULL;

	net = malloc(sizeof(network));
	if (!net)
		return NULL;
	net->n_inputs = n_inputs;
	net->inputs = NULL;
	net->layers = NULL;
	net->out = NULL;
	net->inputs = malloc((size_t)n_inputs * sizeof(double));
	if (!net->inputs) {
		network_free(&net);
		return NULL;
	}

	for (i = 0; i < n_inputs; i++)
		net->inputs[i] = randfrom(0, 1);

	for (i = 0; i < n_layers; i++) {
		neuron *neurons;

		if (sizes[i] <= 0) {
			network_free(&net);
			return NULL;
		}
		neurons = layer_neurons_create(
			sizes[i],
			net->inputs,
			n_inputs,
			prev);
		if (!neurons || nlayer_append(&net->layers, &tail, neurons) != 0) {
			neuron_clear(&neurons);
			network_free(&net);
			return NULL;
		}
		prev = neurons;
	}
	net->out = tail;
	return net;
}

network *network_create(int n_inputs, int n_neurons, int n_layers)
{
	int *sizes;
	network *net;
	int i;

	if (n_inputs <= 0 || n_neurons <= 0 || n_layers <= 0)
		exit(EXIT_FAILURE);
	sizes = malloc((size_t)n_layers * sizeof(int));
	if (!sizes)
		exit(EXIT_FAILURE);
	for (i = 0; i < n_layers; i++)
		sizes[i] = n_neurons;
	net = network_create_sizes(n_inputs, sizes, n_layers);
	free(sizes);
	if (!net)
		exit(EXIT_FAILURE);
	return net;
}

void network_forward(network *net)
{
	if (net)
		nlayer_forward(net->layers);
}

void network_update(network *net, double lr)
{
	nlayer *layer;

	if (!net)
		return;
	layer = net->layers;
	while (layer) {
		neuron_update(layer->neuron, lr);
		layer = layer->next;
	}
}

void network_set_inputs(network *net, const double *values, int n_values)
{
	int i;

	if (!net || !values || n_values != net->n_inputs)
		return;
	for (i = 0; i < n_values; i++)
		net->inputs[i] = values[i];
}

void network_train(network *net, const double *samples, const double *targets,
	int n_samples, int n_target, int epochs, double lr)
{
	int e;
	int s;

	if (!net || !samples || !targets || n_samples <= 0 || n_target <= 0
		|| epochs <= 0)
		return;

	for (e = 0; e < epochs; e++) {
		for (s = 0; s < n_samples; s++) {
			network_set_inputs(net, samples + s * net->n_inputs,
				net->n_inputs);
			network_forward(net);
			network_backward(net, targets + s * n_target, n_target);
			network_update(net, lr);
		}
	}
}

double network_mse_dataset(network *net, const double *samples,
	const double *targets, int n_samples, int n_target)
{
	double sum;
	int s;

	if (!net || !samples || !targets || n_samples <= 0 || n_target <= 0)
		return -1.0;

	sum = 0.0;
	for (s = 0; s < n_samples; s++) {
		network_set_inputs(net, samples + s * net->n_inputs, net->n_inputs);
		network_forward(net);
		sum += network_mse(net, targets + s * n_target, n_target);
	}
	return sum / (double)n_samples;
}

void network_free(network **net)
{
	if (!net || !*net)
		return;
	nlayer_clear(&(*net)->layers);
	(*net)->out = NULL;
	free((*net)->inputs);
	(*net)->inputs = NULL;
	(*net)->n_inputs = 0;
	free(*net);
	*net = NULL;
}
