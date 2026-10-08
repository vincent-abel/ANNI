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

static void test_create_rejects_empty(void)
{
	const int bad[] = {2, 0};
	network *net;

	net = network_create_sizes(2, bad, 2);
	assert(net == NULL);
	net = network_create_sizes(1, bad, 0);
	assert(net == NULL);
	net = network_create_sizes(0, bad, 1);
	assert(net == NULL);
}

static void test_cross_entropy(void)
{
	vec *probs;
	vec *logits;
	double ce;
	double sce;

	probs = vec_create(3);
	logits = vec_create(2);
	assert(probs && logits);
	probs->data[0] = 0.1;
	probs->data[1] = 0.7;
	probs->data[2] = 0.2;
	ce = vec_cross_entropy(probs, 1);
	assert(fabs(ce - (-log(0.7))) < 1e-9);
	logits->data[0] = 1.0;
	logits->data[1] = 2.0;
	sce = vec_softmax_cross_entropy(logits, 1);
	assert(fabs(sce - (-log(exp(2.0) / (exp(1.0) + exp(2.0))))) < 1e-9);
	vec_free(&probs);
	vec_free(&logits);
}

static void test_softmax(void)
{
	vec *in;
	vec *out;
	double sum;

	in = vec_create(2);
	out = vec_create(2);
	assert(in && out);
	in->data[0] = 1.0;
	in->data[1] = 2.0;
	assert(vec_softmax(out, in) == 0);
	sum = out->data[0] + out->data[1];
	assert(fabs(sum - 1.0) < 1e-9);
	assert(fabs(out->data[0] - exp(1.0) / (exp(1.0) + exp(2.0))) < 1e-9);
	assert(fabs(out->data[1] - exp(2.0) / (exp(1.0) + exp(2.0))) < 1e-9);
	vec_free(&in);
	vec_free(&out);
}

static void test_mat_mul_blocked_matches_naive(void)
{
	mat *a;
	mat *b;
	mat *c_block;
	mat *c_naive;
	int i;
	int n = 37;

	a = mat_create(n, n);
	b = mat_create(n, n);
	c_block = mat_create(n, n);
	c_naive = mat_create(n, n);
	assert(a && b && c_block && c_naive);
	srand(3);
	for (i = 0; i < n * n; i++) {
		a->data[i] = (double)(rand() % 11) - 5.0;
		b->data[i] = (double)(rand() % 11) - 5.0;
	}
	assert(mat_mul(c_block, a, b) == 0);
	assert(mat_mul_naive(c_naive, a, b) == 0);
	for (i = 0; i < n * n; i++)
		assert(fabs(c_block->data[i] - c_naive->data[i]) < 1e-9);
	mat_free(&a);
	mat_free(&b);
	mat_free(&c_block);
	mat_free(&c_naive);
}

static void test_tensor_views(void)
{
	const int shape1[] = {4};
	const int shape2[] = {2, 3};
	tensor *t1;
	tensor *t2;
	vec view_v;
	mat view_m;
	int idx[] = {1, 2};
	double *p;

	t1 = tensor_create(shape1, 1);
	t2 = tensor_create(shape2, 2);
	assert(t1 && t2);
	assert(t1->size == 4);
	assert(t2->size == 6);
	assert(tensor_as_vec(&view_v, t1) == 0);
	assert(view_v.n == 4);
	assert(view_v.data == t1->data);
	tensor_fill(t2, 0.0);
	p = tensor_at(t2, idx);
	assert(p != NULL);
	*p = 7.5;
	assert(tensor_as_mat(&view_m, t2) == 0);
	assert(view_m.rows == 2 && view_m.cols == 3);
	assert(fabs(mat_get(&view_m, 1, 2) - 7.5) < 1e-9);
	tensor_free(&t1);
	tensor_free(&t2);
}

static void test_mat_mul_known(void)
{
	mat *a;
	mat *b;
	mat *c;
	vec *x;
	vec *y;

	a = mat_create(2, 2);
	b = mat_create(2, 2);
	c = mat_create(2, 2);
	x = vec_create(2);
	y = vec_create(2);
	assert(a && b && c && x && y);
	mat_set(a, 0, 0, 1.0);
	mat_set(a, 0, 1, 2.0);
	mat_set(a, 1, 0, 3.0);
	mat_set(a, 1, 1, 4.0);
	x->data[0] = 1.0;
	x->data[1] = 0.0;
	assert(mat_vec_mul(y, a, x) == 0);
	assert(fabs(y->data[0] - 1.0) < 1e-9);
	assert(fabs(y->data[1] - 3.0) < 1e-9);
	mat_set(b, 0, 0, 0.0);
	mat_set(b, 0, 1, 1.0);
	mat_set(b, 1, 0, 1.0);
	mat_set(b, 1, 1, 0.0);
	assert(mat_mul(c, a, b) == 0);
	assert(fabs(mat_get(c, 0, 0) - 2.0) < 1e-9);
	assert(fabs(mat_get(c, 0, 1) - 1.0) < 1e-9);
	assert(fabs(mat_get(c, 1, 0) - 4.0) < 1e-9);
	assert(fabs(mat_get(c, 1, 1) - 3.0) < 1e-9);
	mat_free(&a);
	mat_free(&b);
	mat_free(&c);
	vec_free(&x);
	vec_free(&y);
}

static void test_vec_dot_axpy(void)
{
	vec *a;
	vec *b;
	vec *y;

	a = vec_create(3);
	b = vec_create(3);
	y = vec_create(3);
	assert(a && b && y);
	a->data[0] = 1.0;
	a->data[1] = 2.0;
	a->data[2] = 3.0;
	b->data[0] = 4.0;
	b->data[1] = 5.0;
	b->data[2] = 6.0;
	assert(fabs(vec_dot(a, b) - 32.0) < 1e-9);
	y->data[0] = 3.0;
	y->data[1] = 0.0;
	y->data[2] = 1.0;
	vec_axpy(y, 2.0, a);
	assert(fabs(y->data[0] - 5.0) < 1e-9);
	assert(fabs(y->data[1] - 4.0) < 1e-9);
	assert(fabs(y->data[2] - 7.0) < 1e-9);
	vec_free(&a);
	vec_free(&b);
	vec_free(&y);
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

static void test_serialize_roundtrip(void)
{
	network *net;
	network *loaded;
	const int sizes[] = {4, 1};
	const char *path = "tests/tmp_xor.ann";
	double out_a;
	double out_b;
	int s;

	srand(1);
	net = network_create_sizes(XOR_INPUTS, sizes, 2);
	network_train(net, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS, 2000, 0.5);
	assert(network_save(net, path) == 0);
	loaded = network_load(path);
	assert(loaded != NULL);
	assert(loaded->n_inputs == net->n_inputs);
	assert(network_output_size(loaded) == network_output_size(net));

	for (s = 0; s < XOR_SAMPLES; s++) {
		network_set_inputs(net, xor_inputs + s * XOR_INPUTS, XOR_INPUTS);
		network_forward(net);
		out_a = net->out->neuron->output;
		network_set_inputs(loaded, xor_inputs + s * XOR_INPUTS, XOR_INPUTS);
		network_forward(loaded);
		out_b = loaded->out->neuron->output;
		assert(fabs(out_a - out_b) < 1e-9);
	}
	remove(path);
	network_free(&net);
	network_free(&loaded);
}

static void test_xor_adam(void)
{
	network *net;
	adam *opt;
	const int sizes[] = {4, 1};
	double mse;
	int s;
	int correct;
	double out;

	srand(1);
	net = network_create_sizes(XOR_INPUTS, sizes, 2);
	opt = adam_create(net);
	assert(opt != NULL);
	network_train_adam(net, opt, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS, 2000, 0.05);
	mse = network_mse_dataset(net, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS);
	assert(mse < 0.05);

	correct = 0;
	for (s = 0; s < XOR_SAMPLES; s++) {
		network_set_inputs(net, xor_inputs + s * XOR_INPUTS, XOR_INPUTS);
		network_forward(net);
		out = net->out->neuron->output;
		if ((out >= 0.5) == (xor_targets[s] >= 0.5))
			correct++;
	}
	assert(correct == XOR_SAMPLES);
	adam_free(&opt);
	network_free(&net);
}

static void test_xor_minibatches(void)
{
	network *net;
	const int sizes[] = {4, 1};
	double mse;
	int s;
	int correct;
	double out;

	srand(1);
	net = network_create_sizes(XOR_INPUTS, sizes, 2);
	network_train_mb(net, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS, 5000, 0.5, 2);
	mse = network_mse_dataset(net, xor_inputs, xor_targets,
		XOR_SAMPLES, XOR_OUTPUTS);
	assert(mse < 0.02);

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

static void test_tokenize_roundtrip(void)
{
	const char *text = "Hello";
	int ids[16];
	char out[16];
	int n;

	n = tokenize_encode(text, ids, 16);
	assert(n == 5);
	assert(ids[0] == VOCAB_SPECIALS + (unsigned char)'H');
	assert(ids[4] == VOCAB_SPECIALS + (unsigned char)'o');
	assert(tokenize_decode(ids, n, out, 16) == 5);
	assert(strcmp(out, text) == 0);
}






static void test_attn_mha(void)
{
	attn *a;
	mat *x;
	mat *out;

	a = attn_create();
	assert(a);
	x = mat_create(3, LM_D_MODEL);
	out = mat_create(3, LM_D_MODEL);
	assert(x && out);
	mat_fill(x, 0.05);
	assert(attn_mha(out, a, x, 0) == 0);
	attn_free(&a);
	mat_free(&x); mat_free(&out);
}

static void test_attn_self(void)
{
	mat *q;
	mat *k;
	mat *v;
	mat *out;
	int i;

	q = mat_create(2, 4);
	k = mat_create(2, 4);
	v = mat_create(2, 4);
	out = mat_create(2, 4);
	assert(q && k && v && out);
	mat_fill(q, 0.25);
	mat_fill(k, 0.25);
	mat_fill(v, 0.0);
	for (i = 0; i < 4; i++) {
		mat_set(v, 0, i, 2.0);
		mat_set(v, 1, i, 4.0);
	}
	assert(attn_self(out, q, k, v, 0) == 0);
	/* equal Q/K rows => uniform weights => mean of V rows */
	assert(fabs(mat_get(out, 0, 0) - 3.0) < 1e-9);
	assert(fabs(mat_get(out, 1, 0) - 3.0) < 1e-9);
	mat_free(&q); mat_free(&k); mat_free(&v); mat_free(&out);
}

static void test_attn_qkv(void)
{
	attn *a;
	mat *x;
	mat *q;
	mat *k;
	mat *v;

	a = attn_create();
	assert(a);
	x = mat_create(3, LM_D_MODEL);
	q = mat_create(3, LM_D_MODEL);
	k = mat_create(3, LM_D_MODEL);
	v = mat_create(3, LM_D_MODEL);
	assert(x && q && k && v);
	mat_fill(x, 0.1);
	assert(attn_qkv_project(q, k, v, a, x) == 0);
	attn_free(&a);
	mat_free(&x); mat_free(&q); mat_free(&k); mat_free(&v);
}

static void test_posenc_changes(void)
{
	mat *m;
	double before;

	m = mat_create(2, LM_D_MODEL);
	assert(m);
	mat_fill(m, 0.0);
	posenc_sinusoidal_add(m);
	before = mat_get(m, 1, 0);
	assert(fabs(before) > 0.0 || fabs(mat_get(m, 1, 1)) > 0.0);
	mat_free(&m);
}

static void test_embed_lookup(void)
{
	embed *e;
	mat *out;
	int ids[2];

	e = embed_create(8);
	assert(e != NULL);
	out = mat_create(2, LM_D_MODEL);
	assert(out != NULL);
	ids[0] = 1;
	ids[1] = 2;
	assert(embed_lookup(out, e, ids, 2) == 0);
	assert(fabs(mat_get(out, 0, 0) - mat_get(e->table, 1, 0)) < 1e-12);
	embed_free(&e);
	mat_free(&out);
}

static void test_vocab_byte(void)
{
	vocab *v;

	v = vocab_byte_create();
	assert(v != NULL);
	assert(vocab_size(v) == LM_BYTE_VOCAB + VOCAB_SPECIALS);
	assert(strcmp(vocab_token(v, VOCAB_BOS), "<bos>") == 0);
	assert(vocab_id(v, "<eos>") == VOCAB_EOS);
	assert(vocab_id(v, "A") == VOCAB_SPECIALS + (unsigned char)'A');
	vocab_free(&v);
	assert(v == NULL);
}








static void test_ffn_relu_path(void)
{
	ffn *f;
	mat *x;

	f = ffn_create();
	assert(f);
	x = mat_create(2, LM_D_MODEL);
	assert(x);
	mat_fill(x, 0.1);
	assert(ffn_forward(x, f) == 0);
	ffn_free(&f);
	mat_free(&x);
}


static void test_layernorm_zero_mean(void)
{
	layernorm *ln;
	mat *x;
	double mean;
	int j;

	ln = layernorm_create();
	assert(ln);
	x = mat_create(1, LM_D_MODEL);
	assert(x);
	for (j = 0; j < LM_D_MODEL; j++)
		mat_set(x, 0, j, (double)j);
	assert(layernorm_forward(x, ln) == 0);
	mean = 0.0;
	for (j = 0; j < LM_D_MODEL; j++)
		mean += mat_get(x, 0, j);
	mean /= (double)LM_D_MODEL;
	assert(fabs(mean) < 1e-9);
	layernorm_free(&ln);
	mat_free(&x);
}


static void test_resid_add(void)
{
	mat *a;
	mat *b;

	a = mat_create(2, 2);
	b = mat_create(2, 2);
	assert(a && b);
	mat_fill(a, 1.0);
	mat_fill(b, 2.0);
	assert(resid_add(a, b) == 0);
	assert(fabs(mat_get(a, 0, 0) - 3.0) < 1e-12);
	mat_free(&a); mat_free(&b);
}


static void test_attn_causal(void)
{
	mat *q;
	mat *k;
	mat *v;
	mat *out;
	int i;

	q = mat_create(2, 4);
	k = mat_create(2, 4);
	v = mat_create(2, 4);
	out = mat_create(2, 4);
	assert(q && k && v && out);
	mat_fill(q, 0.25);
	mat_fill(k, 0.25);
	mat_fill(v, 0.0);
	for (i = 0; i < 4; i++) {
		mat_set(v, 0, i, 2.0);
		mat_set(v, 1, i, 8.0);
	}
	assert(attn_self(out, q, k, v, 1) == 0);
	assert(fabs(mat_get(out, 0, 0) - 2.0) < 1e-9);
	mat_free(&q); mat_free(&k); mat_free(&v); mat_free(&out);
}


static void test_transformer_block(void)
{
	transformer *tr;
	mat *x;

	tr = transformer_create();
	assert(tr);
	x = mat_create(4, LM_D_MODEL);
	assert(x);
	mat_fill(x, 0.02);
	assert(transformer_forward(x, tr) == 0);
	transformer_free(&tr);
	mat_free(&x);
}


static void test_lm_ntp_loss_drops(void)
{
	lm *model;
	int ids[LM_MAX_CTX];
	int n;
	double before;
	double after;

	srand(2);
	model = lm_create();
	assert(model);
	n = tokenize_encode("hello", ids, LM_MAX_CTX);
	assert(n >= 2);
	before = lm_nll(model, ids, n);
	after = lm_train_steps(model, "hello", 15, 0.01);
	assert(before > 0.0);
	assert(after > 0.0);
	assert(after < before);
	lm_free(&model);
}





static void test_lm_train_corpus(void)
{
	lm *model;
	lm_adam *opt;
	text_dataset *ds;
	double loss;

	srand(11);
	model = lm_create();
	ds = text_dataset_load("data/toy_tiny.txt", 8);
	opt = lm_adam_create(model);
	assert(model && ds && opt);
	loss = lm_train_corpus(model, ds, opt, 1, 2, 0.01, 0);
	assert(loss > 0.0);
	lm_adam_free(&opt);
	text_dataset_free(&ds);
	lm_free(&model);
}

static void test_lm_serialize_roundtrip(void)
{
	lm *a;
	lm *b;
	int ids[LM_MAX_CTX];
	int n;
	double nll_a;
	double nll_b;

	srand(12);
	a = lm_create();
	assert(a);
	n = tokenize_encode("hello", ids, LM_MAX_CTX);
	lm_train_steps(a, "hello", 5, 0.01);
	nll_a = lm_nll(a, ids, n);
	assert(lm_save(a, "objects/test_lm.bin") == 0);
	b = lm_load("objects/test_lm.bin");
	assert(b);
	nll_b = lm_nll(b, ids, n);
	assert(fabs(nll_a - nll_b) < 1e-9);
	lm_free(&a);
	lm_free(&b);
}

static void test_lm_checkpoint_resume(void)
{
	lm *a;
	lm *b;
	lm_adam *oa;
	lm_adam *ob;
	text_dataset *ds;
	double loss1;
	double loss2;
	int epoch;
	int step;

	srand(13);
	a = lm_create();
	ds = text_dataset_load("data/toy_tiny.txt", 8);
	oa = lm_adam_create(a);
	assert(a && ds && oa);
	loss1 = lm_train_corpus(a, ds, oa, 1, 2, 0.01, 0);
	assert(lm_save_checkpoint(a, oa, 1, 0, "objects/test_ckpt.bin") == 0);
	b = lm_load_checkpoint("objects/test_ckpt.bin", &ob, &epoch, &step);
	assert(b && ob && epoch == 1);
	loss2 = lm_train_corpus(b, ds, ob, 1, 2, 0.01, 0);
	assert(loss2 > 0.0 && loss1 > 0.0);
	lm_adam_free(&oa);
	lm_adam_free(&ob);
	text_dataset_free(&ds);
	lm_free(&a);
	lm_free(&b);
}

static void test_lm_perplexity(void)
{
	lm *model;
	text_dataset *ds;
	int ids[LM_MAX_CTX];
	int n;
	double ppl;

	srand(14);
	model = lm_create();
	ds = text_dataset_load("data/toy_val.txt", 4);
	assert(model && ds);
	assert(text_dataset_window(ds, 0, ids, &n) == 0);
	ppl = lm_perplexity(model, ids, n);
	assert(ppl >= 1.0);
	lm_free(&model);
	text_dataset_free(&ds);
}

static void test_lm_generate_from_checkpoint(void)
{
	lm *model;
	lm_adam *opt;
	text_dataset *ds;
	lm *loaded;
	char out[128];
	int n;

	srand(15);
	model = lm_create();
	ds = text_dataset_load("data/toy_tiny.txt", 8);
	opt = lm_adam_create(model);
	assert(model && ds && opt);
	lm_train_corpus(model, ds, opt, 1, 2, 0.01, 0);
	assert(lm_save_checkpoint(model, opt, 1, 0, "objects/gen_ckpt.bin") == 0);
	loaded = lm_load_checkpoint("objects/gen_ckpt.bin", NULL, NULL, NULL);
	assert(loaded);
	n = lm_generate(loaded, "he", out, (int)sizeof(out), 4, 0.0);
	assert(n > 0);
	lm_adam_free(&opt);
	text_dataset_free(&ds);
	lm_free(&model);
	lm_free(&loaded);
}

static void test_text_dataset(void)
{
	text_dataset *ds;
	int ids[LM_MAX_CTX];
	int n;
	int nw;

	ds = text_dataset_load("data/toy_tiny.txt", 8);
	assert(ds);
	nw = text_dataset_n_windows(ds);
	assert(nw > 0);
	assert(text_dataset_window(ds, 0, ids, &n) == 0);
	assert(n == 8);
	text_dataset_free(&ds);
}

static void test_lm_backprop_head(void)
{
	lm *model;
	lm_cache *cache;
	mat *dhidden;
	int ids[4];
	int n;
	double g_analytic;
	double g_numeric;
	double eps;
	double loss_p;
	double loss_m;
	double saved;

	srand(7);
	model = lm_create();
	assert(model);
	n = tokenize_encode("hi", ids, LM_MAX_CTX);
	assert(n >= 2);
	cache = lm_cache_create(n);
	dhidden = mat_create(n, LM_D_MODEL);
	assert(cache && dhidden);
	lm_zero_grad(model);
	assert(lm_forward_cache(model, ids, n, cache) == 0);
	assert(lm_loss_backward(model, cache, ids, n, dhidden) == 0);
	assert(lm_backward(model, dhidden, cache) == 0);
	g_analytic = mat_get(model->dw_head, 0, ids[1]);
	eps = 1e-5;
	saved = mat_get(model->w_head, 0, ids[1]);
	mat_set(model->w_head, 0, ids[1], saved + eps);
	loss_p = lm_nll(model, ids, n);
	mat_set(model->w_head, 0, ids[1], saved - eps);
	loss_m = lm_nll(model, ids, n);
	mat_set(model->w_head, 0, ids[1], saved);
	g_numeric = (loss_p - loss_m) / (2.0 * eps);
	/* mean NLL vs sum of CE: scale by (n-1) for comparable grad */
	g_numeric *= (double)(n - 1);
	assert(fabs(g_analytic - g_numeric) < 1e-3 || fabs(g_analytic) < 1e-8);
	lm_cache_free(&cache);
	mat_free(&dhidden);
	lm_free(&model);
}

static void test_lm_generate_smoke(void)
{
	lm *model;
	char out[128];
	int n;

	srand(3);
	model = lm_create();
	assert(model);
	lm_train_steps(model, "ab", 10, 0.01);
	n = lm_generate(model, "a", out, (int)sizeof(out), 4, 0.0);
	assert(n > 0);
	assert(out[0] != '\0');
	lm_free(&model);
}

static void test_gpu_stub(void)
{
	assert(gpu_available() == 0);
	assert(strcmp(gpu_backend_name(), "cpu") == 0);
	gpu_set_sync_mode(GPU_SYNC_BATCH);
	assert(gpu_sync_mode() == GPU_SYNC_BATCH);
	gpu_set_sync_mode(GPU_SYNC_EAGER);
	assert(gpu_sync_mode() == GPU_SYNC_EAGER);
}

static void test_gpu_gemm_bwd_stub(void)
{
	mat *a;
	mat *b;
	mat *c;
	mat *da;
	mat *db;

	a = mat_create(2, 2);
	b = mat_create(2, 2);
	c = mat_create(2, 2);
	da = mat_create(2, 2);
	db = mat_create(2, 2);
	assert(a && b && c && da && db);
	assert(gpu_mat_mul_backward(da, db, c, a, b) != 0);
	assert(gpu_mat_mul_bt(c, a, b) != 0);
	assert(gpu_mat_outer_add(da, a, b) != 0);
	assert(gpu_adam_step(a->data, b->data, c->data, da->data, 4,
	    0.01, 0.9, 0.999, 1e-8, 1) != 0);
	assert(gpu_kv_attn_decode(a->data, b->data, c->data, da->data, db->data,
	    a->data, b->data, c->data, 1, 2, 2, 1, 2) != 0);
	mat_free(&a);
	mat_free(&b);
	mat_free(&c);
	mat_free(&da);
	mat_free(&db);
}

static void test_mat_device_pin_cpu(void)
{
	mat *m;

	m = mat_create(2, 3);
	assert(m);
	assert(m->d_data == NULL);
	mat_try_pin(m);
	/* Stub: pin must not allocate. */
	assert(m->d_data == NULL);
	assert(mat_device_alloc(m) != 0);
	mat_free(&m);
}

static void test_lm_generate_kv(void)
{
	lm *model;
	char out[128];
	int n;

	srand(31);
	model = lm_create();
	assert(model);
	n = lm_generate_kv(model, "hi", out, (int)sizeof(out), 4, 0.0);
	assert(n > 0);
	assert(out[0] != '\0');
	lm_free(&model);
}

static void test_qmat_roundtrip(void)
{
	mat *m;
	qmat *q;
	vec *y;
	double x[2];
	double ref0;
	double ref1;

	m = mat_create(2, 2);
	assert(m);
	mat_set(m, 0, 0, 1.0);
	mat_set(m, 0, 1, -2.0);
	mat_set(m, 1, 0, 0.5);
	mat_set(m, 1, 1, 0.25);
	q = qmat_from_mat(m);
	y = vec_create(2);
	assert(q && y);
	x[0] = 1.0;
	x[1] = 2.0;
	ref0 = 1.0 * 1.0 + 2.0 * 0.5;
	ref1 = 1.0 * -2.0 + 2.0 * 0.25;
	assert(qmat_gemv_t(y, q, x) == 0);
	assert(fabs(y->data[0] - ref0) < 0.05);
	assert(fabs(y->data[1] - ref1) < 0.05);
	qmat_free(&q);
	vec_free(&y);
	mat_free(&m);
}

static void test_ann_dot_simd(void)
{
	double a[4] = {1.0, 2.0, 3.0, 4.0};
	double b[4] = {5.0, 6.0, 7.0, 8.0};
	double x[16];
	double y[16];
	double ref;
	int i;

	assert(fabs(ann_dot(a, b, 4) - 70.0) < 1e-12);
	ref = 0.0;
	for (i = 0; i < 16; i++) {
		x[i] = (double)(i + 1);
		y[i] = (double)(16 - i);
		ref += x[i] * y[i];
	}
	assert(fabs(ann_dot(x, y, 16) - ref) < 1e-9);
	assert(ann_simd_backend() != NULL);
}

static void test_mat_mul_bt_known(void)
{
	mat *a;
	mat *b;
	mat *c;

	a = mat_create(2, 3);
	b = mat_create(2, 3);
	c = mat_create(2, 2);
	assert(a && b && c);
	/* a rows: [1,0,0] and [0,1,0]; b same -> I_2 */
	mat_set(a, 0, 0, 1.0);
	mat_set(a, 1, 1, 1.0);
	mat_set(b, 0, 0, 1.0);
	mat_set(b, 1, 1, 1.0);
	assert(mat_mul_bt(c, a, b) == 0);
	assert(fabs(mat_get(c, 0, 0) - 1.0) < 1e-12);
	assert(fabs(mat_get(c, 0, 1) - 0.0) < 1e-12);
	assert(fabs(mat_get(c, 1, 0) - 0.0) < 1e-12);
	assert(fabs(mat_get(c, 1, 1) - 1.0) < 1e-12);
	assert(mat_row(c, 0) == c->data);
	mat_free(&a);
	mat_free(&b);
	mat_free(&c);
}

static void test_arena_bump(void)
{
	arena *a;
	double *p;
	double *q;

	a = arena_create(1024);
	assert(a);
	p = arena_alloc(a, 8 * sizeof(double));
	q = arena_alloc(a, 4 * sizeof(double));
	assert(p && q);
	assert((char *)q > (char *)p);
	arena_reset(a);
	assert(a->used == 0);
	arena_free(&a);
	assert(a == NULL);
}

static void test_lm_cache_reuse(void)
{
	lm *model;
	lm_cache *c;
	transformer_cache *trc0;
	int ids[8];
	int i;

	srand(21);
	model = lm_create();
	c = lm_cache_create(8);
	assert(model && c);
	for (i = 0; i < 8; i++)
		ids[i] = i + 1;
	assert(lm_forward_cache(model, ids, 8, c) == 0);
	trc0 = c->trc;
	assert(trc0 != NULL);
	assert(lm_forward_cache(model, ids, 8, c) == 0);
	assert(c->trc == trc0);
	lm_cache_free(&c);
	lm_free(&model);
}


int main(void)
{
	test_mse_known();
	test_forward_shape();
	test_create_free_cycle();
	test_create_rejects_empty();
	test_vec_dot_axpy();
	test_mat_mul_known();
	test_tensor_views();
	test_mat_mul_blocked_matches_naive();
	test_softmax();
	test_cross_entropy();
	test_xor_trains();
	test_xor_minibatches();
	test_xor_adam();
	test_serialize_roundtrip();
	test_tokenize_roundtrip();
	test_vocab_byte();
	test_embed_lookup();
	test_posenc_changes();
	test_attn_qkv();
	test_attn_self();
	test_attn_mha();
	test_ffn_relu_path();
	test_layernorm_zero_mean();
	test_resid_add();
	test_attn_causal();
	test_transformer_block();
	test_lm_ntp_loss_drops();
	test_lm_generate_smoke();
	test_lm_backprop_head();
	test_text_dataset();
	test_lm_train_corpus();
	test_lm_serialize_roundtrip();
	test_lm_checkpoint_resume();
	test_lm_perplexity();
	test_lm_generate_from_checkpoint();
	test_arena_bump();
	test_lm_cache_reuse();
	test_mat_mul_bt_known();
	test_ann_dot_simd();
	test_qmat_roundtrip();
	test_lm_generate_kv();
	test_gpu_stub();
	test_mat_device_pin_cpu();
	test_gpu_gemm_bwd_stub();
	{
		lm *m;
		lm_adam *o;
		double ms;

		m = lm_create();
		o = m ? lm_adam_create(m) : NULL;
		assert(m && o);
		ms = lm_bench_train_ms(m, o, 8, 2);
		assert(ms >= 0.0);
		lm_adam_free(&o);
		lm_free(&m);
	}
	puts("ok");
	return 0;
}
