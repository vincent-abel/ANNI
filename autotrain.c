/*
**	Opinionated Target-oriented train entry (AT1–AT6 MVP + E11 JSON).
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "bpe.h"
#include "dataset_stream.h"
#include "generate.h"
#include "gpu.h"
#include "lm.h"
#include "lm_adam.h"
#include "lm_config.h"
#include "lm_serialize.h"
#include "lm_train.h"

static int file_exists(const char *p)
{
	struct stat st;

	return p && stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

static void write_plan_json(const char *out, const char *data, int B, int T,
	double lr, int epochs, size_t fr, size_t tot)
{
	char path[640];
	FILE *fp;

	snprintf(path, sizeof(path), "%s.autotrain.json", out);
	fp = fopen(path, "w");
	if (!fp)
		return;
	fprintf(fp,
	    "{\n"
	    "  \"out\": \"%s\",\n"
	    "  \"data\": \"%s\",\n"
	    "  \"tokenizer\": \"bpe\",\n"
	    "  \"stream\": true,\n"
	    "  \"val_frac\": 0.01,\n"
	    "  \"batch\": \"auto\",\n"
	    "  \"B\": %d,\n"
	    "  \"window\": %d,\n"
	    "  \"lr\": %.6g,\n"
	    "  \"epochs\": %d,\n"
	    "  \"vram_free_mb\": %.1f,\n"
	    "  \"vram_total_mb\": %.1f\n"
	    "}\n",
	    out, data ? data : "", B, T, lr, epochs,
	    (double)fr / 1048576.0, (double)tot / 1048576.0);
	fclose(fp);
	printf("wrote %s\n", path);
}

static int arg_has_sweep(int argc, char **argv)
{
	int i;

	for (i = 1; i < argc; i++)
		if (strcmp(argv[i], "--sweep") == 0)
			return 1;
	return 0;
}

static void usage(const char *argv0)
{
	fprintf(stderr,
	    "Usage: %s [--out PATH] [--data PATH] [--epochs N] [--sweep] [--help]\n",
	    argv0);
}

int main(int argc, char **argv)
{
	const char *out;
	const char *data;
	int epochs;
	int i;
	int window;
	int B;
	double lr;
	size_t fr;
	size_t tot;
	lm *model;
	lm_adam *opt;
	bpe *tok;
	text_split sp;
	text_stream *tr;
	text_stream *va;
	double loss;
	char smoke[256];
	char vocab_path[640];

	out = "model_auto.bin";
	data = NULL;
	epochs = 1;
	lr = 0.001;
	window = LM_MAX_CTX < 64 ? LM_MAX_CTX : 64;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
			usage(argv[0]);
			return 0;
		}
		if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
			out = argv[++i];
		else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
			data = argv[++i];
		else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
			epochs = atoi(argv[++i]);
	}
	if (epochs < 1)
		epochs = 1;

	/* AT3: ensure data */
	if (!data)
		data = "data/wiki_simple.txt";
	if (!file_exists(data)) {
		if (file_exists("scripts/fetch_wiki.sh")) {
			printf("data missing; running scripts/fetch_wiki.sh\n");
			if (system("scripts/fetch_wiki.sh") != 0) {
				fprintf(stderr,
				    "autotrain: offline — provide --data PATH\n");
				return 1;
			}
		} else {
			fprintf(stderr,
			    "autotrain: missing %s (pass --data or fetch wiki)\n",
			    data);
			return 1;
		}
	}

	fr = tot = 0;
	(void)gpu_mem_info(&fr, &tot);
	B = LM_BATCH_AUTO;
	printf("autotrain plan:\n");
	printf("  data=%s out=%s\n", data, out);
	printf("  tokenizer=bpe stream=1 val_frac=0.01 batch=auto window=%d lr=%g epochs=%d\n",
	    window, lr, epochs);
	if (tot)
		printf("  VRAM free %.0f / total %.0f MB backend=%s\n",
		    (double)fr / 1048576.0, (double)tot / 1048576.0,
		    gpu_backend_name());

	/* AT4: resume if out exists */
	model = NULL;
	opt = NULL;
	tok = NULL;
	if (file_exists(out)) {
		int ep0;
		int st0;

		model = lm_load_checkpoint(out, &opt, &ep0, &st0);
		if (model)
			printf("resuming %s (epoch=%d step=%d)\n", out, ep0, st0);
	}
	if (!model) {
		snprintf(vocab_path, sizeof(vocab_path), "%s.vocab", out);
		if (file_exists(vocab_path))
			tok = bpe_load(vocab_path);
		if (!tok) {
			printf("training BPE vocab on %s...\n", data);
			tok = bpe_train_from_file(data, 16384, 0);
			if (!tok) {
				fprintf(stderr, "autotrain: BPE train failed\n");
				return 1;
			}
			(void)bpe_save(tok, vocab_path);
		}
		model = lm_create_bpe(tok);
		if (!model) {
			fprintf(stderr, "autotrain: lm_create_bpe failed\n");
			bpe_free(&tok);
			return 1;
		}
		opt = lm_adam_create(model);
		if (!opt) {
			lm_free(&model);
			return 1;
		}
	}

	if (text_split_init(&sp, data, TEXT_SPLIT_DOC, 0.01, 1) != 0) {
		fprintf(stderr, "autotrain: split init failed\n");
		return 1;
	}
	tr = text_stream_open(data, window, model->bpe, &sp, TEXT_SIDE_TRAIN,
	    0, 1024, 1);
	va = text_stream_open(data, window, model->bpe, &sp, TEXT_SIDE_VAL,
	    0, 0, 1);
	if (!tr) {
		fprintf(stderr, "autotrain: stream open failed\n");
		return 1;
	}

	lm_train_set_vram_frac(0.85);
	write_plan_json(out, data, B, window, lr, epochs, fr, tot); /* E11/AT5 */
	/* AT7–AT9: optional short lr sweep; keep best via E15/E16 hooks. */
	if (arg_has_sweep(argc, argv)) {
		const double lrs[] = { 3e-4, 1e-3, 3e-3 };
		double best = 1e300;
		double cand;
		int bi = 0;
		char trial[640];

		lm_train_set_early_stop(1);
		lm_train_set_diverge_delta(2.0);
		for (i = 0; i < 3; i++) {
			snprintf(trial, sizeof(trial), "%s.sweep%d", out, i);
			cand = lm_train_stream(model, tr, opt, 1, B, lrs[i], 1,
			    trial, 0, 1, 16, va);
			printf("sweep lr=%g metric=%.4f\n", lrs[i], cand);
			if (cand >= 0.0 && cand < best) {
				best = cand;
				bi = i;
			}
			text_stream_rewind(tr);
			if (va)
				text_stream_rewind(va);
		}
		printf("AT7 best sweep idx %d lr=%g\n", bi, lrs[bi]);
		lr = lrs[bi];
	}
	lm_train_set_early_stop(2);
	lm_train_set_diverge_delta(5.0);
	loss = lm_train_stream(model, tr, opt, epochs, B, lr, 1, out, 0,
	    1, 32, va);
	printf("final stream train metric %.4f\n", loss);

	if (lm_generate(model, "The ", smoke, (int)sizeof(smoke), 32, 0.8, 40,
	    0.9) == 0)
		printf("smoke: %s\n", smoke);
	else
		printf("smoke generate failed\n");

	(void)lm_save_checkpoint(model, opt, epochs, opt->t, out);
	text_stream_free(&tr);
	text_stream_free(&va);
	lm_adam_free(&opt);
	lm_free(&model);
	printf("autotrain done → %s\n", out);
	return 0;
}
