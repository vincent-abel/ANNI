/*
**	GUI bridge
**	One command per line on stdin. One JSON object per line on stdout.
**	The network, the loss, and the updates are the same code as ./ann.
*/
#include "main.h"

#define MAX_IN	784
#define MAX_N	64
#define MAX_L	8
#define MAX_STEPS	200000
#define LOSS_CAP	800

static network	*net;
static double	*samples;
static double	*targets;
static int	n_samples;
static int	n_inputs;
static int	n_target;
static int	sample_i;
static int	epoch;
static double	lr;
static int	arch[MAX_L];
static int	n_arch;
static char	dataset_name[16];
static double	*loss_v;
static int	*loss_e;
static int	n_loss;
static int	loss_cap;
static int	ready;
static char	linebuf[65536];

static void emit_error(const char *msg)
{
	printf("{\"ok\":false,\"error\":\"%s\"}\n", msg);
	fflush(stdout);
}

static void emit_num(double x)
{
	if (isnan(x) || isinf(x) || x > 1.0e6 || x < -1.0e6)
		x = 0.0;
	printf("%.6f", x);
}

static void emit_doubles(const double *v, int n)
{
	int	i;

	putchar('[');
	for (i = 0; i < n; i++) {
		if (i)
			putchar(',');
		emit_num(v[i]);
	}
	putchar(']');
}

static void session_free(void)
{
	network_free(&net);
	free(samples);
	samples = NULL;
	free(targets);
	targets = NULL;
	free(loss_v);
	loss_v = NULL;
	free(loss_e);
	loss_e = NULL;
	n_loss = 0;
	loss_cap = 0;
	n_samples = 0;
	n_inputs = 0;
	n_target = 0;
	n_arch = 0;
	epoch = 0;
	sample_i = 0;
	ready = 0;
	dataset_name[0] = '\0';
}

static void loss_compact(void)
{
	int	r;
	int	w;
	double	last_v;
	int	last_e;

	if (n_loss < 2)
		return;
	last_v = loss_v[n_loss - 1];
	last_e = loss_e[n_loss - 1];
	w = 0;
	for (r = 0; r < n_loss; r += 2) {
		loss_v[w] = loss_v[r];
		loss_e[w] = loss_e[r];
		w++;
	}
	n_loss = w;
	if (n_loss > 0 && loss_e[n_loss - 1] != last_e && n_loss < loss_cap) {
		loss_v[n_loss] = last_v;
		loss_e[n_loss] = last_e;
		n_loss++;
	}
}

static void loss_add(int ep, double mse)
{
	if (!loss_v || !loss_e)
		return;
	if (n_loss >= loss_cap)
		loss_compact();
	if (n_loss >= loss_cap)
		return;
	loss_v[n_loss] = mse;
	loss_e[n_loss] = ep;
	n_loss++;
}

static int next_tok(char **ps, char *buf, int buflen)
{
	char	*s;
	int	n;

	s = *ps;
	n = 0;
	while (*s == ' ' || *s == '\t')
		s++;
	if (!*s) {
		*ps = s;
		return 0;
	}
	while (*s && *s != ' ' && *s != '\t' && n < buflen - 1)
		buf[n++] = *s++;
	buf[n] = '\0';
	*ps = s;
	return 1;
}

static int parse_int(const char *s, int *out)
{
	char	*end;
	long	v;

	if (!s || !*s)
		return -1;
	v = strtol(s, &end, 10);
	if (*end != '\0')
		return -1;
	*out = (int)v;
	return 0;
}

static int parse_double(const char *s, double *out)
{
	char	*end;
	double	v;

	if (!s || !*s)
		return -1;
	v = strtod(s, &end);
	if (*end != '\0')
		return -1;
	*out = v;
	return 0;
}

static void read_outputs(double *dst, int n)
{
	neuron	*neu;
	int	i;

	neu = net->out->neuron;
	i = 0;
	while (neu && i < n) {
		dst[i] = neu->output;
		neu = neu->prev;
		i++;
	}
	while (i < n) {
		dst[i] = 0.0;
		i++;
	}
}

static int neurons_of(neuron *head, neuron **arr)
{
	int	n;

	n = 0;
	while (head && n < MAX_N) {
		arr[n] = head;
		head = head->prev;
		n++;
	}
	return n;
}

static int find_src(const input *in, neuron **prev, int n_prev)
{
	int	k;

	if (!in->i)
		return -1;
	if (!prev) {
		for (k = 0; k < n_inputs; k++) {
			if (in->i == &net->inputs[k])
				return k;
		}
		return -1;
	}
	for (k = 0; k < n_prev; k++) {
		if (in->i == &prev[k]->output)
			return k;
	}
	return -1;
}

static void emit_state(void)
{
	int	s;
	int	i;
	int	li;
	int	ni;
	int	k;
	int	ncount;
	int	nprev;
	int	src_n;
	int	first;
	double	sum;
	double	mse;
	double	outs[MAX_N];
	double	row[MAX_IN];
	neuron	*arr[MAX_N];
	neuron	*prev[MAX_N];
	const input	*in;
	nlayer	*L;

	if (!ready || !net) {
		printf("{\"ok\":true,\"ready\":false}\n");
		fflush(stdout);
		return;
	}

	printf("{\"ok\":true,\"ready\":true,\"dataset\":\"%s\"", dataset_name);
	printf(",\"epoch\":%d,\"lr\":", epoch);
	emit_num(lr);
	printf(",\"sample\":%d,\"n_samples\":%d,\"n_inputs\":%d,\"sizes\":[",
		sample_i, n_samples, n_inputs);
	for (i = 0; i < n_arch; i++) {
		if (i)
			putchar(',');
		printf("%d", arch[i]);
	}
	printf("],\"rows\":[");

	sum = 0.0;
	for (s = 0; s < n_samples; s++) {
		network_set_inputs(net, samples + s * n_inputs, n_inputs);
		network_forward(net);
		sum += network_mse(net, targets + s * n_target, n_target);
		read_outputs(outs, n_target);
		if (s)
			putchar(',');
		printf("{\"in\":");
		emit_doubles(samples + s * n_inputs, n_inputs);
		printf(",\"tg\":");
		emit_doubles(targets + s * n_target, n_target);
		printf(",\"out\":");
		emit_doubles(outs, n_target);
		putchar('}');
	}
	putchar(']');
	mse = n_samples > 0 ? sum / (double)n_samples : 0.0;
	printf(",\"mse\":");
	emit_num(mse);

	network_set_inputs(net, samples + sample_i * n_inputs, n_inputs);
	network_forward(net);

	printf(",\"loss_epoch\":[");
	for (i = 0; i < n_loss; i++) {
		if (i)
			putchar(',');
		printf("%d", loss_e[i]);
	}
	printf("],\"loss\":[");
	for (i = 0; i < n_loss; i++) {
		if (i)
			putchar(',');
		emit_num(loss_v[i]);
	}
	printf("],\"layers\":[");

	L = net->layers;
	nprev = 0;
	li = 0;
	first = 1;
	while (L) {
		ncount = neurons_of(L->neuron, arr);
		if (!first)
			putchar(',');
		first = 0;
		printf("{\"a\":[");
		for (ni = 0; ni < ncount; ni++) {
			if (ni)
				putchar(',');
			emit_num(arr[ni]->output);
		}
		printf("],\"b\":[");
		for (ni = 0; ni < ncount; ni++) {
			if (ni)
				putchar(',');
			emit_num(arr[ni]->bias);
		}
		src_n = (li == 0) ? n_inputs : nprev;
		if (src_n > MAX_IN)
			src_n = MAX_IN;
		printf("],\"w\":[");
		for (ni = 0; ni < ncount; ni++) {
			if (ni)
				putchar(',');
			for (k = 0; k < src_n; k++)
				row[k] = 0.0;
			in = arr[ni]->input;
			while (in) {
				if (li == 0)
					k = find_src(in, NULL, 0);
				else
					k = find_src(in, prev, nprev);
				if (k >= 0 && k < src_n)
					row[k] = in->w;
				in = in->prev;
			}
			emit_doubles(row, src_n);
		}
		printf("]}");
		nprev = ncount;
		for (ni = 0; ni < ncount; ni++)
			prev[ni] = arr[ni];
		L = L->next;
		li++;
	}
	printf("]}\n");
	fflush(stdout);
}

static void set_name(const char *s)
{
	int	i;

	for (i = 0; i < 15 && s[i]; i++)
		dataset_name[i] = s[i];
	dataset_name[i] = '\0';
}

static void cmd_create(char *args)
{
	char	tok[64];
	char	kind[16];
	int	seed;
	int	n_in;
	int	i;
	int	sz[MAX_L];
	int	n_sz;
	double	learn;
	double	m;

	if (!next_tok(&args, kind, (int)sizeof(kind))) {
		emit_error("missing dataset");
		return;
	}
	if (strcmp(kind, "xor") != 0 && strcmp(kind, "custom") != 0) {
		emit_error("dataset must be xor or custom");
		return;
	}
	if (!next_tok(&args, tok, (int)sizeof(tok)) || parse_int(tok, &seed) != 0) {
		emit_error("seed must be an integer");
		return;
	}
	if (!next_tok(&args, tok, (int)sizeof(tok)) || parse_double(tok, &learn) != 0
		|| !(learn > 0.0)) {
		emit_error("learning rate must be a positive number");
		return;
	}
	if (!next_tok(&args, tok, (int)sizeof(tok)) || parse_int(tok, &n_in) != 0
		|| n_in < 1 || n_in > MAX_IN) {
		emit_error("inputs must be an integer from 1 to 784");
		return;
	}
	n_sz = 0;
	while (next_tok(&args, tok, (int)sizeof(tok))) {
		if (n_sz >= MAX_L) {
			emit_error("at most 8 layers");
			return;
		}
		if (parse_int(tok, &sz[n_sz]) != 0 || sz[n_sz] < 1 || sz[n_sz] > MAX_N) {
			emit_error("each layer must have 1 to 64 neurons");
			return;
		}
		n_sz++;
	}
	if (n_sz < 1) {
		emit_error("provide at least one layer size");
		return;
	}
	if (strcmp(kind, "xor") == 0 && (n_in != XOR_INPUTS || sz[n_sz - 1] != 1)) {
		emit_error("xor needs 2 inputs and a final layer of 1");
		return;
	}

	session_free();
	srand((unsigned int)seed);
	net = network_create_sizes(n_in, sz, n_sz);
	if (!net) {
		emit_error("could not create the network");
		return;
	}
	n_inputs = n_in;
	n_target = sz[n_sz - 1];
	n_arch = n_sz;
	for (i = 0; i < n_sz; i++)
		arch[i] = sz[i];
	set_name(kind);
	lr = learn;
	epoch = 0;
	sample_i = 0;

	if (strcmp(kind, "xor") == 0) {
		n_samples = XOR_SAMPLES;
		samples = malloc((size_t)XOR_SAMPLES * XOR_INPUTS * sizeof(double));
		targets = malloc((size_t)XOR_SAMPLES * XOR_OUTPUTS * sizeof(double));
		if (samples)
			memcpy(samples, xor_inputs,
				(size_t)XOR_SAMPLES * XOR_INPUTS * sizeof(double));
		if (targets)
			memcpy(targets, xor_targets,
				(size_t)XOR_SAMPLES * XOR_OUTPUTS * sizeof(double));
	} else {
		n_samples = 1;
		samples = calloc((size_t)n_inputs, sizeof(double));
		targets = calloc((size_t)n_target, sizeof(double));
	}
	loss_cap = LOSS_CAP;
	n_loss = 0;
	loss_v = malloc((size_t)loss_cap * sizeof(double));
	loss_e = malloc((size_t)loss_cap * sizeof(int));
	if (!samples || !targets || !loss_v || !loss_e) {
		session_free();
		emit_error("out of memory");
		return;
	}
	ready = 1;
	m = network_mse_dataset(net, samples, targets, n_samples, n_target);
	loss_add(0, m);
	emit_state();
}

static void cmd_train(char *args)
{
	char	tok[64];
	int	steps;
	int	e;
	int	s;
	int	stride;
	double	learn;
	double	m;

	if (!ready) {
		emit_error("launch the network first");
		return;
	}
	if (!next_tok(&args, tok, (int)sizeof(tok)) || parse_int(tok, &steps) != 0
		|| steps < 1) {
		emit_error("steps must be a positive integer");
		return;
	}
	if (next_tok(&args, tok, (int)sizeof(tok))) {
		if (parse_double(tok, &learn) != 0 || !(learn > 0.0)) {
			emit_error("learning rate must be a positive number");
			return;
		}
		lr = learn;
	}
	if (steps > MAX_STEPS)
		steps = MAX_STEPS;
	stride = steps > 400 ? steps / 200 : 1;
	if (stride < 1)
		stride = 1;
	for (e = 1; e <= steps; e++) {
		for (s = 0; s < n_samples; s++) {
			network_set_inputs(net, samples + s * n_inputs, n_inputs);
			network_forward(net);
			network_backward(net, targets + s * n_target, n_target);
			network_update(net, lr);
		}
		epoch++;
		if ((e % stride) == 0 || e == steps) {
			m = network_mse_dataset(net, samples, targets,
				n_samples, n_target);
			loss_add(epoch, m);
		}
	}
	emit_state();
}

static void cmd_sample(char *args)
{
	int	idx;

	if (!ready) {
		emit_error("launch the network first");
		return;
	}
	if (parse_int(args, &idx) != 0 || idx < 0 || idx >= n_samples) {
		emit_error("sample index out of range");
		return;
	}
	sample_i = idx;
	emit_state();
}

static void cmd_values(char *args)
{
	char	tok[64];
	int	nin;
	int	ntg;
	int	i;
	double	inbuf[MAX_IN];
	double	tgbuf[MAX_N];
	double	m;

	if (!ready) {
		emit_error("launch the network first");
		return;
	}
	if (strcmp(dataset_name, "custom") != 0) {
		emit_error("sample values are for the custom dataset");
		return;
	}
	if (!next_tok(&args, tok, (int)sizeof(tok)) || parse_int(tok, &nin) != 0
		|| nin != n_inputs) {
		emit_error("value counts do not match the network");
		return;
	}
	for (i = 0; i < nin; i++) {
		if (!next_tok(&args, tok, (int)sizeof(tok))
			|| parse_double(tok, &inbuf[i]) != 0) {
			emit_error("missing input value");
			return;
		}
	}
	if (!next_tok(&args, tok, (int)sizeof(tok)) || parse_int(tok, &ntg) != 0
		|| ntg != n_target) {
		emit_error("value counts do not match the network");
		return;
	}
	for (i = 0; i < ntg; i++) {
		if (!next_tok(&args, tok, (int)sizeof(tok))
			|| parse_double(tok, &tgbuf[i]) != 0) {
			emit_error("missing target value");
			return;
		}
	}
	for (i = 0; i < n_inputs; i++)
		samples[i] = inbuf[i];
	for (i = 0; i < n_target; i++)
		targets[i] = tgbuf[i];
	sample_i = 0;
	if (epoch == 0)
		n_loss = 0;
	m = network_mse_dataset(net, samples, targets, n_samples, n_target);
	loss_add(epoch, m);
	emit_state();
}

static int handle(char *line)
{
	char	*args;

	while (*line == ' ' || *line == '\t')
		line++;
	if (!*line)
		return 1;
	args = line;
	while (*args && *args != ' ' && *args != '\t')
		args++;
	if (*args) {
		*args = '\0';
		args++;
		while (*args == ' ' || *args == '\t')
			args++;
	}
	if (strcmp(line, "quit") == 0) {
		printf("{\"ok\":true,\"ready\":false}\n");
		fflush(stdout);
		return 0;
	}
	if (strcmp(line, "create") == 0)
		cmd_create(args);
	else if (strcmp(line, "train") == 0)
		cmd_train(args);
	else if (strcmp(line, "sample") == 0)
		cmd_sample(args);
	else if (strcmp(line, "values") == 0)
		cmd_values(args);
	else if (strcmp(line, "state") == 0)
		emit_state();
	else
		emit_error("unknown command");
	return 1;
}

int main(void)
{
	char	*nl;

	setvbuf(stdout, NULL, _IONBF, 0);
	fprintf(stderr, "ann-gui ready\n");
	while (fgets(linebuf, (int)sizeof(linebuf), stdin)) {
		nl = strchr(linebuf, '\n');
		if (nl)
			*nl = '\0';
		nl = strchr(linebuf, '\r');
		if (nl)
			*nl = '\0';
		if (!handle(linebuf))
			break;
	}
	session_free();
	return 0;
}
