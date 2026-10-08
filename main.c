#include "main.h"

static int arg_eq(const char *a, const char *b)
{
	return a && b && strcmp(a, b) == 0;
}

static const char *arg_value(int argc, char **argv, const char *key, const char *def)
{
	int i;

	for (i = 0; i < argc - 1; i++) {
		if (arg_eq(argv[i], key))
			return argv[i + 1];
	}
	return def;
}

static int arg_flag(int argc, char **argv, const char *short_key, const char *long_key)
{
	int i;

	for (i = 0; i < argc; i++) {
		if (arg_eq(argv[i], short_key) || arg_eq(argv[i], long_key))
			return 1;
	}
	return 0;
}

static void train_dbg(int debug, const char *msg)
{
	time_t now;
	struct tm *tm;
	char buf[32];

	if (!debug)
		return;
	now = time(NULL);
	tm = localtime(&now);
	if (tm)
		strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
	else
		snprintf(buf, sizeof(buf), "unknown-time");
	fprintf(stderr, "[%s] %s\n", buf, msg);
}

static int run_lm_train(int argc, char **argv)
{
	lm *model;
	lm_adam *opt;
	text_dataset *ds;
	const char *data;
	const char *out;
	const char *resume;
	double loss;
	int epochs;
	int batch;
	int debug;
	int nw;
	int prev_epoch;
	int prev_step;
	int total_epochs;
	char msg[256];

	data = arg_value(argc, argv, "--data", "data/toy.txt");
	resume = arg_value(argc, argv, "--resume", NULL);
	out = arg_value(argc, argv, "--out", resume ? resume : "model.bin");
	epochs = atoi(arg_value(argc, argv, "--epochs", "3"));
	batch = atoi(arg_value(argc, argv, "--batch", "4"));
	debug = arg_flag(argc, argv, "-d", "--debug");
	if (epochs <= 0)
		epochs = 3;
	if (batch <= 0)
		batch = 4;
	prev_epoch = 0;
	prev_step = 0;
	model = NULL;
	opt = NULL;

	train_dbg(debug, "lm train starting");
	snprintf(msg, sizeof(msg), "will ingest file: %s", data);
	train_dbg(debug, msg);
	snprintf(msg, sizeof(msg),
	    "params: epochs=%d batch=%d lr=0.01 out=%s%s%s",
	    epochs, batch, out,
	    resume ? " resume=" : "",
	    resume ? resume : "");
	train_dbg(debug, msg);

	srand(1);
	if (resume) {
		snprintf(msg, sizeof(msg), "resuming from checkpoint: %s", resume);
		train_dbg(debug, msg);
		model = lm_load_checkpoint(resume, &opt, &prev_epoch, &prev_step);
		if (!model) {
			/* weights-only file: warm restart with fresh Adam */
			model = lm_load(resume);
			if (model) {
				opt = lm_adam_create(model);
				train_dbg(debug,
				    "warm restart (no Adam state in file)");
			}
		} else {
			snprintf(msg, sizeof(msg),
			    "loaded checkpoint epoch=%d step=%d adam_t=%d",
			    prev_epoch, prev_step, opt ? opt->t : -1);
			train_dbg(debug, msg);
		}
	} else {
		train_dbg(debug, "creating model and Adam optimizer");
		model = lm_create();
		opt = model ? lm_adam_create(model) : NULL;
	}
	train_dbg(debug, "loading dataset / tokenizing");
	ds = text_dataset_load(data, 8);
	if (!model || !ds || !opt) {
		lm_free(&model);
		text_dataset_free(&ds);
		lm_adam_free(&opt);
		fprintf(stderr, "lm train: failed to init%s\n",
		    resume ? " (bad --resume path?)" : "");
		return 1;
	}
	nw = text_dataset_n_windows(ds);
	snprintf(msg, sizeof(msg),
	    "dataset ready: %d token ids, %d windows (len 8)",
	    ds->n_ids, nw);
	train_dbg(debug, msg);
	train_dbg(debug, "running mini-batch corpus training");
	loss = lm_train_corpus(model, ds, opt, epochs, batch, 0.01, debug);
	printf("LM train NLL %.4f\n", loss);
	total_epochs = prev_epoch + epochs;
	train_dbg(debug, "saving checkpoint");
	if (lm_save_checkpoint(model, opt, total_epochs, prev_step, out) != 0) {
		fprintf(stderr, "lm train: save failed\n");
		lm_free(&model);
		text_dataset_free(&ds);
		lm_adam_free(&opt);
		return 1;
	}
	printf("LM saved %s\n", out);
	snprintf(msg, sizeof(msg), "lm train finished (wrote %s)", out);
	train_dbg(debug, msg);
	lm_free(&model);
	text_dataset_free(&ds);
	lm_adam_free(&opt);
	return 0;
}

static int run_lm_generate(int argc, char **argv)
{
	lm *model;
	char out[512];
	const char *path;
	const char *prompt;
	int nnew;

	path = arg_value(argc, argv, "--model", "model.bin");
	prompt = arg_value(argc, argv, "--prompt", "hello");
	nnew = atoi(arg_value(argc, argv, "--tokens", "16"));
	if (nnew < 0)
		nnew = 16;

	model = lm_load(path);
	if (!model) {
		/* try checkpoint format */
		model = lm_load_checkpoint(path, NULL, NULL, NULL);
	}
	if (!model) {
		fprintf(stderr, "lm generate: cannot load %s\n", path);
		return 1;
	}
	if (lm_generate(model, prompt, out, (int)sizeof(out), nnew, 0.0) < 0) {
		lm_free(&model);
		return 1;
	}
	printf("%s\n", out);
	lm_free(&model);
	return 0;
}

static int run_lm_load(int argc, char **argv)
{
	lm *model;
	const char *path;
	int ids[LM_MAX_CTX];
	int n;
	double nll;
	double ppl;

	path = arg_value(argc, argv, "--model", "model.bin");
	model = lm_load(path);
	if (!model)
		model = lm_load_checkpoint(path, NULL, NULL, NULL);
	if (!model) {
		fprintf(stderr, "lm load: cannot load %s\n", path);
		return 1;
	}
	n = tokenize_encode("hello", ids, LM_MAX_CTX);
	nll = lm_nll(model, ids, n);
	ppl = lm_perplexity(model, ids, n);
	printf("LM loaded %s NLL %.4f PPL %.4f\n", path, nll, ppl);
	lm_free(&model);
	return 0;
}

static int run_lm_demo(int argc, char **argv)
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
	loss1 = lm_train_steps(model, phrase, 20, 0.01);
	printf("LM NLL before %.4f after %.4f\n", loss0, loss1);
	if (lm_generate(model, phrase, out, (int)sizeof(out), 8, 0.0) < 0) {
		lm_free(&model);
		return 1;
	}
	printf("LM gen: %s\n", out);
	lm_free(&model);
	return 0;
}

static int run_lm(int argc, char **argv)
{
	if (argc >= 3 && arg_eq(argv[2], "train"))
		return run_lm_train(argc, argv);
	if (argc >= 3 && arg_eq(argv[2], "generate"))
		return run_lm_generate(argc, argv);
	if (argc >= 3 && arg_eq(argv[2], "load"))
		return run_lm_load(argc, argv);
	return run_lm_demo(argc, argv);
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
		printf("       ./ann lm train [--data PATH] [--epochs N] [--out PATH]\n");
		printf("                 [--resume PATH] [-d|--debug]\n");
		printf("       ./ann lm generate --model model.bin --prompt hello\n");
		printf("       ./ann lm load --model model.bin\n");
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
