#include "main.h"




static int run_lm(int argc, char **argv)
{
	lm *model;
	char out[256];
	const char *phrase;
	double loss0;
	double loss1;
	int ids[LM_MAX_CTX];
	int n;

	srand(1);
	model = lm_create();
	if (!model)
		return 1;
	phrase = (argc >= 3) ? argv[2] : "hello";
	n = tokenize_encode(phrase, ids, LM_MAX_CTX);
	if (n < 2) {
		lm_free(&model);
		return 1;
	}
	loss0 = lm_nll(model, ids, n);
	loss1 = lm_train_steps(model, phrase, 100, 0.01);
	printf("LM NLL before %.4f after %.4f\n", loss0, loss1);
	if (lm_generate(model, phrase, out, (int)sizeof(out), 8, 0.0) < 0) {
		lm_free(&model);
		return 1;
	}
	printf("LM gen: %s\n", out);
	lm_free(&model);
	return 0;
}

static int run_xor(void)
{
	network *net;
	const int sizes[] = {4, 1};
	int s;
	double out;
	double mse;
	int correct;

	srand(1);
	net = network_create_sizes(XOR_INPUTS, sizes, 2);
	if (!net)
		exit(EXIT_FAILURE);
	mse = network_mse_dataset(net, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS);
	printf("XOR MSE before  %10.6f\n", mse);

	network_train(net, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS, 5000, 0.5);

	mse = network_mse_dataset(net, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS);
	printf("XOR MSE after   %10.6f\n\n", mse);

	correct = 0;
	for (s = 0; s < XOR_SAMPLES; s++) {
		network_set_inputs(net, xor_inputs + s * XOR_INPUTS, XOR_INPUTS);
		network_forward(net);
		out = net->out->neuron->output;
		printf("  %.0f xor %.0f -> %.4f  (target %.0f)\n",
			xor_inputs[s * XOR_INPUTS],
			xor_inputs[s * XOR_INPUTS + 1],
			out,
			xor_targets[s]);
		if ((out >= 0.5) == (xor_targets[s] >= 0.5))
			correct++;
	}
	printf("\naccuracy  %d / %d\n\n", correct, XOR_SAMPLES);
	network_show(net);
	network_free(&net);
	return correct == XOR_SAMPLES ? 0 : 1;
}

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

	if (argc >= 2 && strcmp(argv[1], "xor") == 0)
		return run_xor();
	if (argc >= 2 && strcmp(argv[1], "lm") == 0)
		return run_lm(argc, argv);

	srand((unsigned int)time(NULL));

	if (argc < 4) {
		printf("Usage: ./ann <inputs> <neurons-per-layer> <layers>\n");
		printf("       ./ann xor\n");
		printf("       ./ann lm [phrase]\n");
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
	if (!samples || !targets || !target) {
		free(samples);
		free(targets);
		free(target);
		network_free(&net);
		exit(EXIT_FAILURE);
	}
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
