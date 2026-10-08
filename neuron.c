/*
**	Neuron Code
*/
#include "main.h"

#define RESPONSE 0.5

double randfrom(double min, double max)
{
	double range = (max - min);
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

void input_push(input **p, double *in, double weigh, neuron *src)
{
	input *new_input = malloc(sizeof(input));
	if (!new_input)
		exit(EXIT_FAILURE);
	new_input->i = in;
	new_input->w = weigh;
	new_input->dw = 0.0;
	new_input->src = src;
	new_input->prev = *p;
	*p = new_input;
}

void neuron_push(neuron **p)
{
	neuron *n = malloc(sizeof(neuron));
	if (!n)
		exit(EXIT_FAILURE);
	n->Tinput = 0;
	n->output = 0;
	n->bias = randfrom(-0.5, 0.5);
	n->delta = 0.0;
	n->dbias = 0.0;
	n->input = NULL;
	n->prev = *p;
	*p = n;
}

void neuron_clear(neuron **p)
{
	neuron *tmp;

	while (*p) {
		input_clear(&(*p)->input);
		tmp = (*p)->prev;
		free(*p);
		*p = tmp;
	}
}

void neuron_connect_values(neuron *n, double *values, int count)
{
	int i;

	for (i = 0; i < count; i++)
		input_push(&n->input, &values[i], randfrom(-2, 2), NULL);
}

void neuron_connect_layer(neuron *n, neuron *src)
{
	while (src) {
		input_push(&n->input, &src->output, randfrom(-2, 2), src);
		src = src->prev;
	}
}

static void neuron_table_sep(void)
{
	puts("+--------+------------+------------+------------+");
}

static void conn_table_sep(void)
{
	puts("+--------+--------+------------+------------+");
}

static void conn_table_rows(int neuron_i, input *p, int *src)
{
	if (!p)
		return;
	conn_table_rows(neuron_i, p->prev, src);
	printf("| %6d | %6d | %10.4f | %10.4f |\n",
		neuron_i, *src, *(p->i), p->w);
	(*src)++;
}

void neuron_show(neuron *p)
{
	neuron *n;
	int i;

	neuron_table_sep();
	puts("| Neuron |      Input |       Bias |   Output \xCE\xA3 |");
	neuron_table_sep();
	n = p;
	i = 0;
	while (n) {
		printf("| %6d | %10.4f | %10.4f | %10.4f |\n",
			i++, n->Tinput, n->bias, n->output);
		n = n->prev;
	}
	neuron_table_sep();

	puts("");
	conn_table_sep();
	puts("| Neuron |     in |      input |      weigh |");
	conn_table_sep();
	n = p;
	i = 0;
	while (n) {
		int src = 0;

		conn_table_rows(i++, n->input, &src);
		n = n->prev;
	}
	conn_table_sep();
}

int input_delete(input **p)
{
	input *tmp;

	if (!*p)
		return -1;
	tmp = (*p)->prev;
	free(*p);
	*p = tmp;
	return 0;
}

int input_size(input *p)
{
	int n = 0;

	while (p) {
		n++;
		p = p->prev;
	}
	return n;
}

void input_clear(input **p)
{
	input *tmp;

	while (*p) {
		tmp = (*p)->prev;
		free(*p);
		*p = tmp;
	}
}

void input_show(input *p)
{
	int src = 0;

	conn_table_sep();
	puts("| Neuron |     in |      input |      weigh |");
	conn_table_sep();
	conn_table_rows(0, p, &src);
	conn_table_sep();
}

double neuron_Tinput(neuron *n)
{
	input *i = n->input;

	n->Tinput = n->bias;
	while (i) {
		n->Tinput += *(i->i) * i->w;
		i = i->prev;
	}
	return n->Tinput;
}

void neuron_output(neuron *n)
{
	while (n) {
		n->output = 1.0 / (1.0 + exp(-neuron_Tinput(n) / RESPONSE));
		n = n->prev;
	}
}

static double sigmoid_prime(double output)
{
	return output * (1.0 - output);
}

static void neuron_backward_accumulate(neuron *n)
{
	input *in;

	n->dbias += n->delta;
	in = n->input;
	while (in) {
		in->dw += n->delta * *(in->i);
		if (in->src)
			in->src->delta += n->delta * in->w;
		in = in->prev;
	}
}

void neuron_zero_delta(neuron *n)
{
	input *in;

	while (n) {
		n->delta = 0.0;
		n->dbias = 0.0;
		in = n->input;
		while (in) {
			in->dw = 0.0;
			in = in->prev;
		}
		n = n->prev;
	}
}

void neuron_backward_output(neuron *n, const double *target, int n_target)
{
	int i;

	i = 0;
	while (n && i < n_target) {
		n->delta = (n->output - target[i]) * sigmoid_prime(n->output);
		neuron_backward_accumulate(n);
		i++;
		n = n->prev;
	}
}

void neuron_backward_hidden(neuron *n)
{
	while (n) {
		n->delta *= sigmoid_prime(n->output);
		neuron_backward_accumulate(n);
		n = n->prev;
	}
}

void neuron_update(neuron *n, double lr)
{
	input *in;

	while (n) {
		n->bias -= lr * n->dbias;
		n->dbias = 0.0;
		in = n->input;
		while (in) {
			in->w -= lr * in->dw;
			in->dw = 0.0;
			in = in->prev;
		}
		n->delta = 0.0;
		n = n->prev;
	}
}
