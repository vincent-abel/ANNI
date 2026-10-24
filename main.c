#include "main.h"
#include "gpu.h"
#ifdef _WIN32
#include "ann_posix.h"
#else
#include <unistd.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#endif

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

static int run_lm_train_bench_gpu(int argc, char **argv)
{
	lm *model;
	lm_adam *opt;
	int n;
	int steps;
	double ms;

	n = atoi(arg_value(argc, argv, "--tokens", "32"));
	steps = atoi(arg_value(argc, argv, "--steps", "20"));
	if (n < 2)
		n = 32;
	if (steps <= 0)
		steps = 20;
	srand(1);
	model = lm_create();
	opt = model ? lm_adam_create(model) : NULL;
	if (!model || !opt) {
		lm_free(&model);
		lm_adam_free(&opt);
		fprintf(stderr, "lm train --bench-gpu: init failed\n");
		return 1;
	}
	ms = lm_bench_train_ms(model, opt, n, steps);
	printf("LM train GPU bench backend=%s available=%d tokens=%d steps=%d mean_ms=%.3f\n",
	    gpu_backend_name(), gpu_available(), n, steps, ms);
	if (gpu_available())
		printf("Corroborate util: nvidia-smi dmon -s u -c 10  (while this runs)\n");
	else
		printf("No CUDA device in this binary (CPU stub). Rebuild with: make cuda\n");
	lm_adam_free(&opt);
	lm_free(&model);
	return ms < 0.0 ? 1 : 0;
}

static int run_lm_train_core(int argc, char **argv, int use_tower)
{
	lm *model;
	lm_adam *opt;
	lm *probe;
	text_dataset *ds;
	const char *data;
	const char *out;
	const char *resume;
	const char *gen_prompt;
	double loss;
	double lr;
	double win_nll;
	int epochs;
	int batch;
	int grad_accum;
	double vram_frac;
	const char *batch_arg;
	int window;
	int seed;
	int debug;
	int force;
	tower_train_cfg tw;
	int tower_on;
	int eval_every;
	int eval_windows;
	int nw;
	int prev_epoch;
	int prev_step;
	int total_epochs;
	char msg[256];
	char line[32];
	char batch_str[16];
	const char *vocab_arg;
	const char *tokenizer;
	const char *split_arg;
	char vocab_buf[600];
	int vocab_size_arg;
	long bpe_max_bytes;
	double val_frac;
	int split_seed;
	int split_mode;
	int use_split;
	int stream;
	int shuffle_buf;
	int stride;
	int rc;
	bpe *tok;
	bpe *chk;
	text_split sp;
	text_dataset *val_ds;
	const char *val_data;
	text_stream *tstream;
	text_stream *vstream;
	double val_nll;

	if (arg_flag(argc, argv, NULL, "--bench-gpu"))
		return run_lm_train_bench_gpu(argc, argv);

	data = arg_value(argc, argv, "--data", "data/toy.txt");
	resume = arg_value(argc, argv, "--resume", NULL);
	out = arg_value(argc, argv, "--out", resume ? resume : "model.bin");
	epochs = atoi(arg_value(argc, argv, "--epochs", "3"));
	/*
	** --batch N: N sequences packed per forward/backward (token-parallel);
	** --batch auto: size from free VRAM. Default 16 on a GPU build, else 4.
	*/
	batch_arg = arg_value(argc, argv, "--batch", NULL);
	if (!batch_arg)
		batch = gpu_available() ? 16 : 4;
	else if (arg_eq(batch_arg, "auto"))
		batch = LM_BATCH_AUTO;
	else
		batch = atoi(batch_arg);
	grad_accum = atoi(arg_value(argc, argv, "--grad-accum", "1"));
	vram_frac = atof(arg_value(argc, argv, "--vram-frac", "0.85"));
	lr = atof(arg_value(argc, argv, "--lr", "0.001"));	/* E1: safe FP32 default */
	window = atoi(arg_value(argc, argv, "--window", "64"));
	seed = atoi(arg_value(argc, argv, "--seed", "1"));
	eval_every = atoi(arg_value(argc, argv, "--eval-every", "1"));
	/* 0 = all windows (slow); default 32 matches mid-epoch subsample. */
	eval_windows = atoi(arg_value(argc, argv, "--eval-windows", "32"));
	debug = arg_flag(argc, argv, "-d", "--debug");
	force = arg_flag(argc, argv, "-y", "--force");
	gen_prompt = arg_value(argc, argv, "--prompt", "The ");
	vocab_arg = arg_value(argc, argv, "--vocab", NULL);
	tower_on = 0;
	if (use_tower) {
		if (!tower_available()) {
			fprintf(stderr,
			    "lm tower train: ncurses UI unavailable "
			    "(install libncurses-dev and rebuild, or use a TTY)\n");
			return 1;
		}
		/* Tower owns the UI; suppress stderr progress bars. */
		debug = 0;
	}
	tokenizer = arg_value(argc, argv, "--tokenizer", "byte");
	vocab_size_arg = atoi(arg_value(argc, argv, "--vocab-size",
	    "16384"));
	bpe_max_bytes = atol(arg_value(argc, argv, "--bpe-max-bytes", "0"));
	val_frac = atof(arg_value(argc, argv, "--val-frac", "0"));
	split_arg = arg_value(argc, argv, "--split", "doc");
	split_seed = atoi(arg_value(argc, argv, "--split-seed", "1"));
	stream = arg_flag(argc, argv, NULL, "--stream");
	val_data = arg_value(argc, argv, "--val-data", NULL);
	shuffle_buf = atoi(arg_value(argc, argv, "--shuffle-buf", "1024"));
	stride = atoi(arg_value(argc, argv, "--stride", "0"));
	if (epochs <= 0)
		epochs = 3;
	if (batch < 0 || (batch == 0 && batch_arg && !arg_eq(batch_arg, "auto"))) {
		fprintf(stderr, "lm train: --batch must be N >= 1 or auto\n");
		return 1;
	}
	if (grad_accum < 1)
		grad_accum = 1;
	if (vram_frac <= 0.0 || vram_frac > 1.0)
		vram_frac = 0.85;
	lm_train_set_grad_accum(grad_accum);
	lm_train_set_vram_frac(vram_frac);
	{
		int warm = atoi(arg_value(argc, argv, "--lr-warmup", "0"));
		int decay = atoi(arg_value(argc, argv, "--lr-decay-steps", "0"));
		const char *dk = arg_value(argc, argv, "--lr-decay", "none");
		double clip = atof(arg_value(argc, argv, "--grad-clip", "0"));
		double wd = atof(arg_value(argc, argv, "--weight-decay", "0"));
		int tps = atoi(arg_value(argc, argv, "--tokens-per-step", "0"));

		lm_train_set_lr_schedule(warm, decay, dk);
		lm_train_set_grad_clip(clip);
		lm_train_set_weight_decay(wd);
		lm_train_set_tokens_per_step(tps);
		lm_train_set_early_stop(atoi(arg_value(argc, argv,
		    "--early-stop-patience", "0")));
		lm_train_set_diverge_delta(atof(arg_value(argc, argv,
		    "--diverge-delta", "0")));
		if (arg_flag(argc, argv, NULL, "--varlen"))
			lm_train_set_varlen(1);
		if (arg_flag(argc, argv, NULL, "--cuda-graphs"))
			lm_train_set_cuda_graphs(1);
		{
			int bf16 = arg_flag(argc, argv, NULL, "--bf16");
			int fp16 = arg_flag(argc, argv, NULL, "--fp16");

			if (bf16 && fp16) {
				fprintf(stderr,
				    "lm train: use only one of --bf16 / --fp16\n");
				return 1;
			}
			if (bf16)
				lm_train_set_compute_prec(GPU_PREC_BF16);
			else if (fp16)
				lm_train_set_compute_prec(GPU_PREC_FP16);
			/*
			** --fp8: E4M3 forward + E5M2 dgrad GEMMs; wgrad stays
			** BF16; FP32 Adam masters. Implies --fp8-dgrad.
			*/
			if (arg_flag(argc, argv, NULL, "--fp8"))
				lm_train_set_fp8(1);
			/* --fp8-dgrad alone: E5M2 dX only (no forward FP8). */
			if (arg_flag(argc, argv, NULL, "--fp8-dgrad"))
				lm_train_set_fp8_dgrad(1);
		}
		if (arg_flag(argc, argv, NULL, "--flash-attn"))
			lm_train_set_flash_attn(1);
		if (arg_flag(argc, argv, NULL, "--act-checkpoint"))
			lm_train_set_act_checkpoint(1);
		if (arg_flag(argc, argv, NULL, "--adam-offload"))
			lm_train_set_adam_offload(1);
	}
	if (lr <= 0.0)
		lr = 0.001;
	if (window < 2)
		window = 64;
	if (window > LM_MAX_CTX)
		window = LM_MAX_CTX;
	if (seed < 0)
		seed = 1;
	if (eval_every < 0)
		eval_every = 1;
	if (eval_windows < 0)
		eval_windows = 0;
	if (shuffle_buf < 0)
		shuffle_buf = 0;
	if (val_frac < 0.0 || val_frac >= 1.0) {
		fprintf(stderr, "lm train: --val-frac must be in [0,1)\n");
		return 1;
	}
	if (arg_eq(split_arg, "doc")) {
		split_mode = TEXT_SPLIT_DOC;
	} else if (arg_eq(split_arg, "offset")) {
		split_mode = TEXT_SPLIT_OFFSET;
	} else {
		fprintf(stderr, "lm train: --split must be doc or offset\n");
		return 1;
	}
	if (!arg_eq(tokenizer, "byte") && !arg_eq(tokenizer, "bpe")) {
		fprintf(stderr, "lm train: --tokenizer must be byte or bpe\n");
		return 1;
	}
	if (!vocab_arg && arg_eq(tokenizer, "bpe") && !resume) {
		snprintf(vocab_buf, sizeof(vocab_buf), "%s.vocab", out);
		vocab_arg = vocab_buf;
	}
	use_split = val_frac > 0.0;
	prev_epoch = 0;
	prev_step = 0;
	model = NULL;
	opt = NULL;
	tok = NULL;
	val_ds = NULL;
	tstream = NULL;
	vstream = NULL;
	ds = NULL;
	rc = 1;

	/*
	** Existing --out that looks like a model: offer resume unless --force.
	** Non-TTY must pass --resume or --force (no hang).
	*/
	if (!resume && access(out, R_OK) == 0) {
		probe = lm_load_any(out);
		if (probe) {
			lm_free(&probe);
			if (force) {
				train_dbg(debug,
				    "--force: overwriting existing --out");
			} else if (!isatty(STDIN_FILENO)) {
				fprintf(stderr,
				    "lm train: %s looks like a checkpoint; "
				    "pass --resume %s or --force to overwrite\n",
				    out, out);
				return 1;
			} else {
				fprintf(stderr,
				    "%s looks like a model/checkpoint. "
				    "Resume training from it? [y/N] ", out);
				fflush(stderr);
				if (!fgets(line, (int)sizeof(line), stdin)) {
					fprintf(stderr,
					    "lm train: no confirmation; abort\n");
					return 1;
				}
				if (line[0] == 'y' || line[0] == 'Y')
					resume = out;
				else
					train_dbg(debug,
					    "overwrite confirmed (fresh model)");
			}
		}
	}

	if (batch == LM_BATCH_AUTO)
		snprintf(batch_str, sizeof(batch_str), "auto");
	else
		snprintf(batch_str, sizeof(batch_str), "%d", batch);
	train_dbg(debug, "lm train starting");
	snprintf(msg, sizeof(msg), "will ingest file: %s", data);
	train_dbg(debug, msg);
	snprintf(msg, sizeof(msg),
	    "params: epochs=%d batch=%s grad_accum=%d lr=%.4g window=%d seed=%d "
	    "eval_every=%d eval_windows=%d out=%s%s%s",
	    epochs, batch_str, grad_accum, lr, window, seed, eval_every,
	    eval_windows, out,
	    resume ? " resume=" : "",
	    resume ? resume : "");
	train_dbg(debug, msg);

	srand((unsigned int)seed);
	if (use_split && text_split_init(&sp, data, split_mode, val_frac,
	    (unsigned int)split_seed) != 0) {
		fprintf(stderr, "lm train: cannot read --data %s\n", data);
		return 1;
	}
	if (resume) {
		snprintf(msg, sizeof(msg), "resuming from checkpoint: %s", resume);
		train_dbg(debug, msg);
		model = lm_load_checkpoint(resume, &opt, &prev_epoch, &prev_step);
		if (!model) {
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
		/* The checkpoint decides the tokenizer; --vocab must agree. */
		if (model && vocab_arg && access(vocab_arg, R_OK) == 0) {
			chk = bpe_load(vocab_arg);
			if (!chk || !model->bpe || !bpe_equal(chk, model->bpe)) {
				fprintf(stderr, "lm train: --vocab %s does not match "
				    "the tokenizer in %s\n", vocab_arg, resume);
				bpe_free(&chk);
				lm_free(&model);
				lm_adam_free(&opt);
				return 1;
			}
			bpe_free(&chk);
		}
	} else if (vocab_arg) {
		tok = access(vocab_arg, R_OK) == 0 ? bpe_load(vocab_arg) : NULL;
		if (!tok && access(vocab_arg, F_OK) == 0) {
			fprintf(stderr, "lm train: cannot parse vocab %s\n",
			    vocab_arg);
			return 1;
		}
		if (!tok) {
			snprintf(msg, sizeof(msg), "training BPE (vocab %d) on %s%s",
			    vocab_size_arg, data,
			    use_split ? " (train split only)" : "");
			train_dbg(debug, msg);
			if (vocab_size_arg < BPE_BASE)
				vocab_size_arg = BPE_DEFAULT_VOCAB;
			tok = text_split_train_bpe(data, use_split ? &sp : NULL,
			    vocab_size_arg, bpe_max_bytes);
			if (!tok || bpe_save(tok, vocab_arg) != 0) {
				fprintf(stderr, "lm train: BPE training/save failed "
				    "(%s)\n", vocab_arg);
				bpe_free(&tok);
				return 1;
			}
			printf("BPE vocab %d tokens (%d merges) saved to %s\n",
			    tok->vocab_size, tok->n_merges, vocab_arg);
		}
		train_dbg(debug, "creating BPE model and Adam optimizer");
		model = lm_create_bpe(tok);
		if (!model)
			bpe_free(&tok);
		opt = model ? lm_adam_create(model) : NULL;
	} else {
		train_dbg(debug, "creating model and Adam optimizer");
		model = lm_create();
		opt = model ? lm_adam_create(model) : NULL;
	}
	train_dbg(debug, "loading dataset / tokenizing");
	if (model && opt && stream) {
		tstream = text_stream_open(data, window, model->bpe,
		    use_split ? &sp : NULL,
		    use_split ? TEXT_SIDE_TRAIN : TEXT_SIDE_ALL, stride,
		    shuffle_buf, (unsigned int)seed);
		if (use_split)
			vstream = text_stream_open(data, window, model->bpe, &sp,
			    TEXT_SIDE_VAL, stride, 0, (unsigned int)seed);
	} else if (model && opt && use_split) {
		if (text_dataset_load_split(data, window, model->bpe, &sp, &ds,
		    &val_ds) != 0)
			ds = NULL;
	} else if (model && opt) {
		ds = text_dataset_load_tok(data, window, model->bpe);
	}
	/* Non-stream: --stride N (e.g. =window) for one pass over ids. */
	if (ds && !stream && stride > 0)
		text_dataset_set_stride(ds, stride);
	/* --val-data: held-out file kept apart from --data (e.g. *_val_ids.bin). */
	if (ds && !stream && !use_split && val_data) {
		val_ds = text_dataset_load_tok(val_data, window, model->bpe);
		if (!val_ds) {
			fprintf(stderr, "lm train: cannot load --val-data %s\n",
			    val_data);
			text_dataset_free(&ds);
		} else
			text_dataset_set_stride(val_ds, window);
	}
	if (!model || !opt || (stream ? !tstream : !ds)) {
		lm_free(&model);
		text_dataset_free(&ds);
		text_dataset_free(&val_ds);
		text_stream_free(&tstream);
		text_stream_free(&vstream);
		lm_adam_free(&opt);
		fprintf(stderr, "lm train: failed to init%s\n",
		    resume ? " (bad --resume path?)" : "");
		return 1;
	}
	if (model->bpe) {
		printf("LM tokenizer: bpe vocab=%d merges=%d\n",
		    model->bpe->vocab_size, model->bpe->n_merges);
	}
	if (use_split) {
		if (stream)
			snprintf(msg, sizeof(msg),
			    "split: %s val_frac=%.4g (streaming, doc-disjoint)",
			    split_arg, val_frac);
		else
			snprintf(msg, sizeof(msg),
			    "split: %s val_frac=%.4g train windows=%d val windows=%d",
			    split_arg, val_frac, text_dataset_n_windows(ds),
			    val_ds ? text_dataset_n_windows(val_ds) : 0);
		train_dbg(debug, msg);
		if (!stream && !val_ds)
			fprintf(stderr, "lm train: warning: split left no "
			    "validation windows (docs shorter than --window?)\n");
	}
	if (use_tower) {
		memset(&tw, 0, sizeof(tw));
		tw.model = model;
		tw.opt = opt;
		tw.data = data;
		tw.out = out;
		tw.prompt = gen_prompt;
		tw.lr = lr;
		tw.epochs = epochs;
		tw.batch = batch;
		tw.window = window;
		tw.epoch_base = prev_epoch;
		if (tower_train_begin(&tw) != 0) {
			fprintf(stderr, "lm tower train: UI init failed\n");
			goto done;
		}
		tower_on = 1;
	}
	if (stream) {
		snprintf(msg, sizeof(msg),
		    "streaming: window=%d stride=%d shuffle_buf=%d",
		    window, text_stream_stride(tstream), shuffle_buf);
		train_dbg(debug, msg);
		total_epochs = prev_epoch + epochs;
		loss = lm_train_stream(model, tstream, opt, epochs, batch, lr,
		    debug, out, prev_epoch, eval_every, eval_windows, vstream);
	} else {
		nw = text_dataset_n_windows(ds);
		snprintf(msg, sizeof(msg),
		    "dataset ready: %ld token ids, %d windows (len %d)",
		    ds->n_ids, nw, window);
		train_dbg(debug, msg);
		if (nw <= 0) {
			fprintf(stderr,
			    "lm train: no windows (need >%d tokens after "
			    "tokenize; try a larger --data or smaller --window)\n",
			    window);
			if (tower_on)
				tower_train_end();
			lm_free(&model);
			text_dataset_free(&ds);
			text_dataset_free(&val_ds);
			lm_adam_free(&opt);
			return 1;
		}
		train_dbg(debug, "running mini-batch corpus training");
		total_epochs = prev_epoch + epochs;
		loss = lm_train_corpus_val(model, ds, opt, epochs, batch, lr,
		    debug, out, prev_epoch, eval_every, eval_windows, val_ds);
	}
	if (tower_on) {
		tower_train_end();
		tower_on = 0;
	}
	if (loss < 0.0) {
		fprintf(stderr, "lm train: failed (train or per-epoch save)\n");
		goto done;
	}
	printf("LM train NLL %.4f\n", loss);
	if (eval_every > 0) {
		if (vstream) {
			val_nll = lm_nll_stream(model, vstream, eval_windows);
			if (val_nll >= 0.0)
				printf("LM val window NLL %.4f\n", val_nll);
		} else if (val_ds) {
			win_nll = lm_nll_windows(model, val_ds, eval_windows);
			printf("LM val window NLL %.4f\n", win_nll);
		} else if (!stream) {
			win_nll = lm_nll_windows(model, ds, eval_windows);
			printf("LM train window NLL %.4f\n", win_nll);
		}
	}
	printf("LM saved %s (after each epoch, last epoch %d)\n",
	    out, total_epochs);
	snprintf(msg, sizeof(msg), "lm train finished (wrote %s)", out);
	train_dbg(debug, msg);
	rc = 0;
done:
	if (tower_on)
		tower_train_end();
	lm_free(&model);
	text_dataset_free(&ds);
	text_dataset_free(&val_ds);
	text_stream_free(&tstream);
	text_stream_free(&vstream);
	lm_adam_free(&opt);
	return rc;
}

static int run_lm_train(int argc, char **argv)
{
	return run_lm_train_core(argc, argv, 0);
}

static int run_lm_info(int argc, char **argv)
{
	lm *model;
	lm_adam *opt;
	lm_info info;
	text_dataset *ds;
	const char *path;
	const char *prompt;
	const char *data;
	int epoch;
	int step;
	int eval_windows;
	int window;
	int ids[LM_MAX_CTX];
	int n;

	path = arg_value(argc, argv, "--model", "model.bin");
	prompt = arg_value(argc, argv, "--prompt", "hello");
	data = arg_value(argc, argv, "--data", NULL);
	eval_windows = atoi(arg_value(argc, argv, "--eval-windows", "0"));
	window = atoi(arg_value(argc, argv, "--window", "64"));
	if (window < 2)
		window = 64;
	if (window > LM_MAX_CTX)
		window = LM_MAX_CTX;

	opt = NULL;
	epoch = 0;
	step = 0;
	model = lm_load_checkpoint(path, &opt, &epoch, &step);
	if (!model) {
		/* Legacy LMCK or bad file: still print peek intel. */
		if (lm_info_peek(path, &info) == 0
		    && info.magic_kind == LM_INFO_MAGIC_LMCK) {
			fprintf(stderr,
			    "lm info: legacy double checkpoint (LMCK); "
			    "architecture from header only\n");
			lm_info_print(&info, stdout);
			return 0;
		}
		model = lm_load_any(path);
	}
	if (!model) {
		fprintf(stderr, "lm info: cannot load %s\n", path);
		return 1;
	}
	lm_info_from_model(&info, path, model, opt, epoch, step);
	n = lm_encode(model, prompt, ids, LM_MAX_CTX);
	if (n >= 2) {
		info.prompt_nll = lm_nll(model, ids, n);
		info.prompt_ppl = lm_perplexity(model, ids, n);
	}
	ds = NULL;
	if (data) {
		ds = text_dataset_load_tok(data, window, model->bpe);
		if (ds)
			info.data_win_nll = lm_nll_windows(model, ds,
			    eval_windows);
	}
	lm_info_print(&info, stdout);
	text_dataset_free(&ds);
	lm_adam_free(&opt);
	lm_free(&model);
	return 0;
}

static int run_lm_tower(int argc, char **argv)
{
	tower_inspect_cfg icfg;

	if (argc >= 4 && arg_eq(argv[3], "train"))
		return run_lm_train_core(argc, argv, 1);
	if (argc >= 4 && arg_eq(argv[3], "inspect")) {
		if (!tower_available()) {
			fprintf(stderr,
			    "lm tower inspect: ncurses UI unavailable; "
			    "use ./ann lm info --model … instead\n");
			return 1;
		}
		memset(&icfg, 0, sizeof(icfg));
		icfg.model_path = arg_value(argc, argv, "--model", "model.bin");
		icfg.prompt = arg_value(argc, argv, "--prompt", "The ");
		icfg.data = arg_value(argc, argv, "--data", NULL);
		icfg.eval_windows = atoi(arg_value(argc, argv,
		    "--eval-windows", "0"));
		icfg.window = atoi(arg_value(argc, argv, "--window", "64"));
		if (tower_inspect(&icfg) != 0) {
			fprintf(stderr, "lm tower inspect: failed to load %s\n",
			    icfg.model_path);
			return 1;
		}
		return 0;
	}
	fprintf(stderr,
	    "Usage: ./anni lm tower train [lm train flags…]\n"
	    "       ./anni lm tower inspect --model PATH [--prompt TEXT]\n"
	    "                 [--data PATH] [--eval-windows M] [--window N]\n");
	return 1;
}

/*
** --vocab PATH for generate/load: the checkpoint's embedded tokenizer wins;
** the file must match it, or (for a vocab-less model of equal size) is attached.
*/
static int apply_vocab_arg(lm *model, int argc, char **argv, const char *who)
{
	const char *path;
	bpe *tok;

	path = arg_value(argc, argv, "--vocab", NULL);
	if (!path)
		return 0;
	tok = bpe_load(path);
	if (!tok) {
		fprintf(stderr, "%s: cannot load vocab %s\n", who, path);
		return -1;
	}
	if (model->bpe) {
		if (!bpe_equal(tok, model->bpe)) {
			fprintf(stderr, "%s: --vocab %s differs from the "
			    "tokenizer embedded in the model\n", who, path);
			bpe_free(&tok);
			return -1;
		}
		bpe_free(&tok);
		return 0;
	}
	if (lm_attach_bpe(model, tok) != 0) {
		fprintf(stderr, "%s: --vocab %s has %d tokens but the model "
		    "has %d\n", who, path, tok->vocab_size, model->vocab_size);
		bpe_free(&tok);
		return -1;
	}
	return 0;
}

struct gen_view {
	int		nnew;
	int		vocab;
	int		tty;
	double		t0;
	double		t_first;	/* wall time of the first new piece */
	int		cols;		/* terminal size */
	int		lines;
	int		row;		/* cursor row below the meter, 0-based */
	int		col;
};

static double gen_wall(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static void gen_term_size(struct gen_view *v)
{
	v->cols = 80;
	v->lines = 24;
#ifdef TIOCGWINSZ
	{
		struct winsize ws;

		if (ioctl(fileno(stderr), TIOCGWINSZ, &ws) == 0) {
			if (ws.ws_col > 0)
				v->cols = ws.ws_col;
			if (ws.ws_row > 0)
				v->lines = ws.ws_row;
		}
	}
#endif
}

/* Follow the cursor through text written below the meter (with autowrap). */
static void gen_track(struct gen_view *v, const char *s, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		unsigned char c = (unsigned char)s[i];

		if (c == '\n') {
			v->row++;
			v->col = 0;
		} else if (c == '\r')
			v->col = 0;
		else if (c == '\t') {
			if (v->col >= v->cols) {
				v->row++;
				v->col = 0;
			}
			v->col += 8 - v->col % 8;
			if (v->col > v->cols)
				v->col = v->cols;
		} else if (c >= 0x20 && c != 0x7f && (c & 0xc0) != 0x80) {
			if (v->col >= v->cols) {
				v->row++;
				v->col = 0;
			}
			v->col++;
		}
	}
}

/* Forward GEMMs: 2 flops per stored weight (blocks + vocab head). */
static double gen_tflops(double tok_sec, int vocab)
{
	double params;

	params = (double)LM_N_BLOCKS * (4.0 * LM_D_MODEL * LM_D_MODEL
	    + 2.0 * LM_D_MODEL * LM_D_FF)
	    + (double)LM_D_MODEL * (double)vocab;
	return 2.0 * params * tok_sec / 1e12;
}

static void gen_meter(struct gen_view *v, int final)
{
	double dt;
	double ts;
	double tf;
	double steady;
	char buf[96];

	dt = gen_wall() - v->t0;
	if (v->nnew <= 0 || dt <= 0.0) {
		ts = 0.0;
		tf = 0.0;
	} else {
		ts = (double)v->nnew / dt;
		tf = gen_tflops(ts, v->vocab);
	}
	/* Steady excludes load, prefill and the first token. */
	steady = 0.0;
	if (v->nnew > 1 && gen_wall() > v->t_first)
		steady = (double)(v->nnew - 1) / (gen_wall() - v->t_first);
	if (v->tty) {
		/* Once the meter scrolls off the top, cursor-up would clamp
		** onto a text line; print the summary below instead. */
		if (v->row + 1 > v->lines - 1) {
			if (final)
				fprintf(stderr, "%.0f t/s  %.3f TF  steady %.1f "
				    "t/s\n", ts, tf, steady);
			fflush(stderr);
			return;
		}
		/* At the last column the wrap is pending; save/restore drops
		** it on some terminals, so wait for the next piece. */
		if (v->col >= v->cols)
			return;
		if (final)
			snprintf(buf, sizeof(buf), "%7.0f t/s  %7.3f TF  steady "
			    "%.1f t/s", ts, tf, steady);
		else
			snprintf(buf, sizeof(buf), "%7.0f t/s  %7.3f TF", ts, tf);
		fprintf(stderr, "\0337\033[%dA\r%.*s\033[K\0338", v->row + 1,
		    v->cols - 1, buf);
		fflush(stderr);
		return;
	}
	if (!final)
		return;
	fprintf(stderr, "%.0f t/s  %.3f TF  steady %.1f t/s\n", ts, tf, steady);
	fflush(stderr);
}

static void gen_emit(const char *s, int n, void *ud)
{
	struct gen_view *v;

	v = ud;
	if (s && n > 0) {
		fwrite(s, 1, (size_t)n, stdout);
		fflush(stdout);
		gen_track(v, s, n);
	}
	if (v->nnew == 0)
		v->t_first = gen_wall();
	v->nnew++;
	gen_meter(v, 0);
}

static int run_lm_generate(int argc, char **argv)
{
	lm *model;
	struct gen_view view;
	const char *path;
	const char *prompt;
	int nnew;
	int top_k;
	int sync_was;
	double temperature;
	double top_p;

	path = arg_value(argc, argv, "--model", "model.bin");
	prompt = arg_value(argc, argv, "--prompt", "hello");
	nnew = atoi(arg_value(argc, argv, "--tokens", "16"));
	temperature = atof(arg_value(argc, argv, "--temperature", "1.0"));
	top_k = atoi(arg_value(argc, argv, "--top-k", "0"));
	top_p = atof(arg_value(argc, argv, "--top-p", "1.0"));
	if (nnew < 0)
		nnew = 16;
	if (top_k < 0)
		top_k = 0;

	memset(&view, 0, sizeof(view));
	view.tty = isatty(fileno(stdout)) && isatty(fileno(stderr));
	setvbuf(stdout, NULL, _IONBF, 0);
	/* Prompt is already known. Show it before the checkpoint load. */
	if (view.tty) {
		gen_term_size(&view);
		fprintf(stderr, "%7.0f t/s  %7.3f TF\n", 0.0, 0.0);
		fflush(stderr);
	}
	fputs(prompt, stdout);
	fflush(stdout);
	gen_track(&view, prompt, (int)strlen(prompt));

	model = lm_load(path);
	if (!model)
		model = lm_load_checkpoint(path, NULL, NULL, NULL);
	if (!model) {
		fprintf(stderr, "\nlm generate: cannot load %s\n", path);
		return 1;
	}
	if (apply_vocab_arg(model, argc, argv, "lm generate") != 0) {
		lm_free(&model);
		return 1;
	}
	view.vocab = model->vocab_size;
	view.t0 = gen_wall();
	sync_was = gpu_sync_mode();
	gpu_set_sync_mode(GPU_SYNC_BATCH);
	if (lm_generate_stream(model, prompt, nnew, temperature, top_k, top_p,
		gen_emit, &view) < 0) {
		gpu_set_sync_mode(sync_was);
		fprintf(stderr, "\nlm generate: failed\n");
		lm_free(&model);
		return 1;
	}
	gpu_set_sync_mode(sync_was);
	fputc('\n', stdout);
	fflush(stdout);
	gen_track(&view, "\n", 1);
	gen_meter(&view, 1);
	lm_free(&model);
	return 0;
}

static int run_anni_chat_stdin(chat_session *sess, const char *path)
{
	char line[1024];
	char reply[8192];

	printf("AnnI chat (%s) — empty line or Ctrl-D to exit\n", path);
	for (;;) {
		printf("User: ");
		fflush(stdout);
		if (!fgets(line, (int)sizeof(line), stdin)) {
			printf("\n");
			break;
		}
		line[strcspn(line, "\r\n")] = '\0';
		if (line[0] == '\0')
			break;
		if (chat_session_turn(sess, line, reply, (int)sizeof(reply))
		    != 0) {
			fprintf(stderr, "chat: generate failed\n");
			return 1;
		}
		printf("Assistant: %s\n", reply);
	}
	return 0;
}

/* Top-level ./anni chat and alias ./anni lm chat. */
static int run_anni_chat(int argc, char **argv)
{
	lm *model;
	const char *path;
	chat_session sess;
	chat_ui_cfg ui;
	int nnew;
	int top_k;
	double temperature;
	double top_p;
	int rc;

	path = arg_value(argc, argv, "--model", "wiki_bpe.bin");
	nnew = atoi(arg_value(argc, argv, "--tokens", "64"));
	temperature = atof(arg_value(argc, argv, "--temperature", "0.2"));
	top_k = atoi(arg_value(argc, argv, "--top-k", "0"));
	top_p = atof(arg_value(argc, argv, "--top-p", "0.9"));

	model = lm_load(path);
	if (!model)
		model = lm_load_checkpoint(path, NULL, NULL, NULL);
	if (!model) {
		fprintf(stderr, "chat: cannot load %s\n", path);
		return 1;
	}
	if (apply_vocab_arg(model, argc, argv, "chat") != 0) {
		lm_free(&model);
		return 1;
	}
	if (chat_session_init(&sess, model, nnew, temperature, top_k, top_p)
	    != 0) {
		lm_free(&model);
		return 1;
	}

	rc = 1;
	if (tower_available()) {
		memset(&ui, 0, sizeof(ui));
		ui.model = model;
		ui.model_path = path;
		ui.tokens = sess.nnew;
		ui.temperature = temperature;
		ui.top_k = top_k;
		ui.top_p = top_p;
		chat_session_free(&sess);
		rc = chat_ui_run(&ui) != 0 ? 1 : 0;
	} else {
		rc = run_anni_chat_stdin(&sess, path);
		chat_session_free(&sess);
	}
	lm_free(&model);
	return rc;
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
	if (apply_vocab_arg(model, argc, argv, "lm load") != 0) {
		lm_free(&model);
		return 1;
	}
	n = lm_encode(model, arg_value(argc, argv, "--prompt", "hello"), ids,
	    LM_MAX_CTX);
	if (n < 2) {
		fprintf(stderr, "lm load: prompt needs at least 2 tokens\n");
		lm_free(&model);
		return 1;
	}
	nll = lm_nll(model, ids, n);
	ppl = lm_perplexity(model, ids, n);
	printf("LM loaded %s NLL %.4f PPL %.4f\n", path, nll, ppl);
	lm_free(&model);
	return 0;
}

static int run_lm_bench(int argc, char **argv)
{
	lm *model;
	lm_adam *opt;
	int n;
	int iters;
	double ms;

	n = atoi(arg_value(argc, argv, "--tokens", "32"));
	iters = atoi(arg_value(argc, argv, "--iters", "20"));
	if (n <= 0)
		n = 32;
	if (iters <= 0)
		iters = 20;
	srand(1);
	model = lm_create();
	if (!model)
		return 1;
	if (arg_flag(argc, argv, NULL, "--train")) {
		opt = lm_adam_create(model);
		if (!opt) {
			lm_free(&model);
			return 1;
		}
		ms = lm_bench_train_ms(model, opt, n, iters);
		lm_adam_free(&opt);
		lm_free(&model);
		if (ms < 0.0) {
			fprintf(stderr, "lm bench --train: failed\n");
			return 1;
		}
		printf("LM bench train backend=%s available=%d tokens=%d steps=%d mean_ms=%.3f\n",
		    gpu_backend_name(), gpu_available(), n, iters, ms);
		return 0;
	}
	ms = lm_bench_forward_ms(model, n, iters);
	lm_free(&model);
	if (ms < 0.0) {
		fprintf(stderr, "lm bench: failed\n");
		return 1;
	}
	printf("LM bench tokens=%d iters=%d mean_ms=%.3f backend=%s\n",
	    n, iters, ms, gpu_backend_name());
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
	if (lm_generate(model, phrase, out, (int)sizeof(out), 8, 0.0, 0, 1.0) < 0) {
		lm_free(&model);
		return 1;
	}
	printf("LM gen: %s\n", out);
	lm_free(&model);
	return 0;
}

static int run_lm_merge(int argc, char **argv)
{
	const char *out;
	lm *models[32];
	lm *merged;
	const char *paths[32];
	int n;
	int i;
	int rc;

	out = arg_value(argc, argv, "--out", "model_merged.bin");
	n = 0;
	for (i = 3; i < argc; i++) {
		if (argv[i][0] == '-') {
			if (arg_eq(argv[i], "--out") && i + 1 < argc)
				i++;
			continue;
		}
		if (n >= 32) {
			fprintf(stderr, "lm merge: too many models (max 32)\n");
			return 1;
		}
		paths[n++] = argv[i];
	}
	if (n < 2) {
		fprintf(stderr,
		    "Usage: ./ann lm merge --out OUT.bin a.bin b.bin [c.bin ...]\n");
		return 1;
	}
	memset(models, 0, sizeof(models));
	for (i = 0; i < n; i++) {
		models[i] = lm_load_any(paths[i]);
		if (!models[i]) {
			fprintf(stderr, "lm merge: cannot load %s\n", paths[i]);
			while (--i >= 0)
				lm_free(&models[i]);
			return 1;
		}
	}
	merged = NULL;
	rc = lm_merge_average(models, n, &merged);
	for (i = 0; i < n; i++)
		lm_free(&models[i]);
	if (rc != 0 || !merged) {
		fprintf(stderr, "lm merge: average failed\n");
		lm_free(&merged);
		return 1;
	}
	if (lm_save(merged, out) != 0) {
		fprintf(stderr, "lm merge: save %s failed\n", out);
		lm_free(&merged);
		return 1;
	}
	printf("LM merged %d models -> %s (weights only, fresh Adam on next train)\n",
	    n, out);
	lm_free(&merged);
	return 0;
}

static int run_lm(int argc, char **argv)
{
	if (argc >= 3 && arg_eq(argv[2], "train"))
		return run_lm_train(argc, argv);
	if (argc >= 3 && arg_eq(argv[2], "tower"))
		return run_lm_tower(argc, argv);
	if (argc >= 3 && arg_eq(argv[2], "info"))
		return run_lm_info(argc, argv);
	if (argc >= 3 && arg_eq(argv[2], "generate"))
		return run_lm_generate(argc, argv);
	if (argc >= 3 && arg_eq(argv[2], "chat"))
		return run_anni_chat(argc, argv);
	if (argc >= 3 && arg_eq(argv[2], "load"))
		return run_lm_load(argc, argv);
	if (argc >= 3 && arg_eq(argv[2], "bench"))
		return run_lm_bench(argc, argv);
	if (argc >= 3 && arg_eq(argv[2], "merge"))
		return run_lm_merge(argc, argv);
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
	if (argc >= 2 && strcmp(argv[1], "chat") == 0)
		return run_anni_chat(argc, argv);
	if (argc >= 2 && strcmp(argv[1], "lm") == 0)
		return run_lm(argc, argv);

	srand((unsigned int)time(NULL));

	if (argc < 4) {
		printf("Usage: ./anni <inputs> <neurons-per-layer> <layers>\n");
		printf("       ./anni xor\n");
		printf("       ./anni chat --model wiki_bpe.bin\n");
		printf("                 [--tokens N] [--temperature T] [--top-k K] [--top-p P]\n");
		printf("                 [--vocab PATH]\n");
		printf("       ./anni lm [phrase]\n");
		printf("       ./anni lm train [--data PATH] [--epochs N] [--out PATH]\n");
		printf("                 [--resume PATH] [--val-data PATH] [--lr RATE] [--window N]\n");
		printf("                 [--batch N|auto] [--grad-accum K] [--vram-frac F]\n");
		printf("                 [--seed N] [--eval-every N]\n");
		printf("                 [--eval-windows M] [-y|--force] [-d|--debug]\n");
		printf("                 [--vocab PATH] [--tokenizer byte|bpe] [--vocab-size N]\n");
		printf("                 [--bpe-max-bytes N] [--val-frac F] [--split doc|offset]\n");
		printf("                 [--split-seed N] [--stream] [--shuffle-buf N] [--stride N]\n");
		printf("       ./anni lm train --bench-gpu [--tokens N] [--steps N]\n");
		printf("       ./anni lm tower train [same flags as lm train] [--prompt TEXT]\n");
		printf("       ./anni lm tower inspect --model PATH [--prompt TEXT]\n");
		printf("                 [--data PATH] [--eval-windows M] [--window N]\n");
		printf("       ./anni lm info --model PATH [--prompt TEXT] [--data PATH]\n");
		printf("       ./anni lm merge --out OUT.bin a.bin b.bin [c.bin ...]\n");
		printf("       ./anni lm generate --model model.bin --prompt hello\n");
		printf("                 [--tokens N] [--temperature T] [--top-k K] [--top-p P]\n");
		printf("                 [--vocab PATH]\n");
		printf("       ./anni lm chat --model wiki_bpe.bin   (alias of ./anni chat)\n");
		printf("       ./anni lm load --model model.bin [--vocab PATH] [--prompt TEXT]\n");
		printf("       ./anni lm bench [--tokens N] [--iters N] [--train]\n");
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
