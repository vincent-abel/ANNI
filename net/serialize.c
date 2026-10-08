/*
**	Serialize Code
*/
#include "main.h"

#define ANN_MAGIC 0x314E4E41

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

static int write_i32(FILE *fp, int v)
{
	return fwrite(&v, sizeof(int), 1, fp) == 1 ? 0 : -1;
}

static int write_f64(FILE *fp, double v)
{
	return fwrite(&v, sizeof(double), 1, fp) == 1 ? 0 : -1;
}

static int read_i32(FILE *fp, int *v)
{
	return fread(v, sizeof(int), 1, fp) == 1 ? 0 : -1;
}

static int read_f64(FILE *fp, double *v)
{
	return fread(v, sizeof(double), 1, fp) == 1 ? 0 : -1;
}

static int write_neuron(FILE *fp, neuron *n)
{
	input *in;

	if (write_f64(fp, n->bias) != 0)
		return -1;
	if (write_i32(fp, input_size(n->input)) != 0)
		return -1;
	in = n->input;
	while (in) {
		if (write_f64(fp, in->w) != 0)
			return -1;
		in = in->prev;
	}
	return 0;
}

static int read_neuron(FILE *fp, neuron *n)
{
	input *in;
	int n_in;
	int i;
	double w;

	if (read_f64(fp, &n->bias) != 0)
		return -1;
	if (read_i32(fp, &n_in) != 0)
		return -1;
	if (n_in != input_size(n->input))
		return -1;
	in = n->input;
	i = 0;
	while (in && i < n_in) {
		if (read_f64(fp, &w) != 0)
			return -1;
		in->w = w;
		in = in->prev;
		i++;
	}
	return (in == NULL && i == n_in) ? 0 : -1;
}

int network_save(network *net, const char *path)
{
	FILE *fp;
	nlayer *layer;
	neuron *n;
	int n_layers;
	int sizes[64];
	int i;

	if (!net || !path || !net->layers)
		return -1;
	n_layers = count_layers(net->layers);
	if (n_layers <= 0 || n_layers > 64)
		return -1;

	i = 0;
	layer = net->layers;
	while (layer) {
		sizes[i++] = count_neurons(layer->neuron);
		layer = layer->next;
	}

	fp = fopen(path, "wb");
	if (!fp)
		return -1;
	if (write_i32(fp, ANN_MAGIC) != 0
		|| write_i32(fp, net->n_inputs) != 0
		|| write_i32(fp, n_layers) != 0) {
		fclose(fp);
		return -1;
	}
	for (i = 0; i < n_layers; i++) {
		if (write_i32(fp, sizes[i]) != 0) {
			fclose(fp);
			return -1;
		}
	}

	layer = net->layers;
	while (layer) {
		n = layer->neuron;
		while (n) {
			if (write_neuron(fp, n) != 0) {
				fclose(fp);
				return -1;
			}
			n = n->prev;
		}
		layer = layer->next;
	}
	fclose(fp);
	return 0;
}

network *network_load(const char *path)
{
	FILE *fp;
	network *net;
	nlayer *layer;
	neuron *n;
	int magic;
	int n_inputs;
	int n_layers;
	int sizes[64];
	int i;

	if (!path)
		return NULL;
	fp = fopen(path, "rb");
	if (!fp)
		return NULL;
	if (read_i32(fp, &magic) != 0 || magic != ANN_MAGIC
		|| read_i32(fp, &n_inputs) != 0
		|| read_i32(fp, &n_layers) != 0
		|| n_layers <= 0 || n_layers > 64) {
		fclose(fp);
		return NULL;
	}
	for (i = 0; i < n_layers; i++) {
		if (read_i32(fp, &sizes[i]) != 0 || sizes[i] <= 0) {
			fclose(fp);
			return NULL;
		}
	}

	net = network_create_sizes(n_inputs, sizes, n_layers);
	if (!net) {
		fclose(fp);
		return NULL;
	}

	layer = net->layers;
	while (layer) {
		n = layer->neuron;
		while (n) {
			if (read_neuron(fp, n) != 0) {
				network_free(&net);
				fclose(fp);
				return NULL;
			}
			n = n->prev;
		}
		layer = layer->next;
	}
	fclose(fp);
	return net;
}
