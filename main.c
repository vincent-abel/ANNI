#include "main.h"

int main(int argc, char *argv[])
{
	network *net;
	double *samples;
	double *targets;
	double *target;
	int n_inputs;
	int n_neurons;
	int n_layers;
	int n_out;
	int i;

	setvbuf(stdout, NULL, _IONBF, 0);
	srand((unsigned int)time(NULL));

	if (argc < 4) {
		printf("Usage: ./ann <inputs> <neurons-per-layer> <layers>\n");
		return 1;
	}

	n_inputs = atoi(argv[1]);
	n_neurons = atoi(argv[2]);
	n_layers = atoi(argv[3]);
	if (n_inputs <= 0 || n_neurons <= 0 || n_layers <= 0) {
		printf("All sizes must be positive integers.\n");
		return 1;
	}

	net = network_create(n_inputs, n_neurons, n_layers);
	n_out = network_output_size(net);

	samples = malloc((size_t)n_inputs * sizeof(double));
	targets = calloc((size_t)n_out, sizeof(double));
	target = calloc((size_t)n_out, sizeof(double));
	if (!samples || !targets || !target)
		exit(EXIT_FAILURE);
	for (i = 0; i < n_inputs; i++)
		samples[i] = net->inputs[i];

	network_forward(net);
	printf("MSE before train  %10.6f\n", network_mse(net, target, n_out));
	network_train(net, samples, targets, 1, n_out, 50, 0.5);
	network_set_inputs(net, samples, n_inputs);
	network_forward(net);
	network_show(net);
	printf("MSE after train   %10.6f\n\n", network_mse(net, target, n_out));

	free(samples);
	free(targets);
	free(target);
	network_free(&net);
	return 0;
}
