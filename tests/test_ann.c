#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../main.h"

static void test_mse_known(void)
{
	network *net;
	const int sizes[] = {1};
	const double target[] = {0.25};
	double mse;

	srand(1);
	net = network_create_sizes(1, sizes, 1);
	net->inputs[0] = 0.0;
	network_set_inputs(net, net->inputs, 1);
	network_forward(net);
	net->out->neuron->output = 0.75;
	mse = network_mse(net, target, 1);
	assert(fabs(mse - 0.25) < 1e-9);
	network_free(&net);
}

static void test_forward_shape(void)
{
	network *net;
	const int sizes[] = {3, 2};
	const double in[] = {0.2, 0.8};

	srand(2);
	net = network_create_sizes(2, sizes, 2);
	assert(network_output_size(net) == 2);
	network_set_inputs(net, in, 2);
	network_forward(net);
	assert(net->out->neuron->output >= 0.0);
	assert(net->out->neuron->output <= 1.0);
	assert(net->out->neuron->prev != NULL);
	assert(net->out->neuron->prev->output >= 0.0);
	assert(net->out->neuron->prev->output <= 1.0);
	network_free(&net);
}

static void test_create_free_cycle(void)
{
	network *net;
	const int sizes[] = {3, 2, 1};
	int i;

	for (i = 0; i < 20; i++) {
		srand((unsigned int)i + 1);
		net = network_create_sizes(2, sizes, 3);
		network_forward(net);
		network_free(&net);
		assert(net == NULL);
	}
}

static void test_xor_trains(void)
{
	network *net;
	const int sizes[] = {4, 1};
	double mse;
	int s;
	int correct;
	double out;

	srand(1);
	net = network_create_sizes(XOR_INPUTS, sizes, 2);
	network_train(net, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS, 5000, 0.5);
	mse = network_mse_dataset(net, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS);
	assert(mse < 0.01);

	correct = 0;
	for (s = 0; s < XOR_SAMPLES; s++) {
		network_set_inputs(net, xor_inputs + s * XOR_INPUTS, XOR_INPUTS);
		network_forward(net);
		out = net->out->neuron->output;
		if ((out >= 0.5) == (xor_targets[s] >= 0.5))
			correct++;
	}
	assert(correct == XOR_SAMPLES);
	network_free(&net);
}

int main(void)
{
	test_mse_known();
	test_forward_shape();
	test_create_free_cycle();
	test_xor_trains();
	puts("ok");
	return 0;
}
