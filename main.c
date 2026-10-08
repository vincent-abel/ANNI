#include "main.h"

int main(int argc, char *argv[])
{
	network *net;
	int n_inputs;
	int n_neurons;
	int n_layers;

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
	network_forward(net);
	network_show(net);
	network_free(&net);
	return 0;
}
