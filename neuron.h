/*
**	Neuron Header
*/
#ifndef NEURON_H__
#define NEURON_H__

typedef struct	input {
	double	*i;
	double	w;
	double	dw;
	struct	neuron *src;
	struct	input *prev;
} input;

typedef struct	neuron {
	double	Tinput;
	double	output;
	double	bias;
	double	delta;
	double	dbias;
	input	*input;
	struct	neuron *prev;
} neuron;

double	randfrom(double min, double max);
void	input_push(input **, double *, double, neuron *);
int	input_delete(input **);
int	input_size(input *);
void	input_clear(input **);
void	input_show(input *);
void	neuron_push(neuron **);
void	neuron_clear(neuron **);
void	neuron_connect_values(neuron *, double *, int);
void	neuron_connect_layer(neuron *, neuron *);
void	neuron_output(neuron *);
void	neuron_zero_delta(neuron *);
void	neuron_backward_output(neuron *, const double *, int);
void	neuron_backward_hidden(neuron *);
double	neuron_Tinput(neuron *);
void	neuron_show(neuron *);

#endif
