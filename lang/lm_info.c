/*
**	Checkpoint / model intel
*/
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "gpu.h"
#include "lm_config.h"
#include "lm_info.h"
#include "lm_serialize.h"
#include "real.h"

static int read_i32_fp(FILE *fp, int *v)
{
	return fread(v, sizeof(int), 1, fp) == 1 ? 0 : -1;
}

int lm_info_peek(const char *path, lm_info *out)
{
	FILE *fp;
	struct stat st;
	int magic;
	int real_size;
	int d_model;
	int n_heads;
	int d_ff;
	int max_ctx;
	int n_blocks;
	int vsize;
	int merges;

	if (!path || !out)
		return -1;
	memset(out, 0, sizeof(*out));
	out->prompt_nll = -1.0;
	out->prompt_ppl = -1.0;
	out->data_win_nll = -1.0;
	snprintf(out->path, sizeof(out->path), "%s", path);
	if (stat(path, &st) == 0)
		out->file_bytes = (long)st.st_size;
	fp = fopen(path, "rb");
	if (!fp)
		return -1;
	if (read_i32_fp(fp, &magic) != 0) {
		fclose(fp);
		return -1;
	}
	if (magic == (int)LM_MAGIC_F64) {
		out->magic_kind = LM_INFO_MAGIC_LMCK;
		out->real_size = 8;
		if (read_i32_fp(fp, &d_model) == 0
		    && read_i32_fp(fp, &n_heads) == 0
		    && read_i32_fp(fp, &d_ff) == 0
		    && read_i32_fp(fp, &max_ctx) == 0
		    && read_i32_fp(fp, &n_blocks) == 0
		    && read_i32_fp(fp, &vsize) == 0) {
			int block;

			out->d_model = d_model;
			out->n_heads = n_heads;
			out->d_ff = d_ff;
			out->max_ctx = max_ctx;
			out->n_blocks = n_blocks;
			out->vocab_size = vsize;
			block = 4 * d_model * d_model + d_model * d_ff + d_ff
			    + d_ff * d_model + d_model + 4 * d_model;
			out->params = vsize * d_model + n_blocks * block
			    + d_model * vsize + vsize;
			out->weight_bytes = (size_t)out->params
			    * (size_t)out->real_size;
		}
		fclose(fp);
		return 0;
	}
	if (magic == (int)LM_MAGIC)
		out->magic_kind = LM_INFO_MAGIC_LMCF;
	else if (magic == (int)LM_MAGIC_BPE)
		out->magic_kind = LM_INFO_MAGIC_LMCV;
	else {
		out->magic_kind = LM_INFO_MAGIC_OTHER;
		fclose(fp);
		return 0;
	}
	if (read_i32_fp(fp, &real_size) != 0
	    || read_i32_fp(fp, &d_model) != 0
	    || read_i32_fp(fp, &n_heads) != 0
	    || read_i32_fp(fp, &d_ff) != 0
	    || read_i32_fp(fp, &max_ctx) != 0
	    || read_i32_fp(fp, &n_blocks) != 0
	    || read_i32_fp(fp, &vsize) != 0) {
		fclose(fp);
		return -1;
	}
	out->real_size = real_size;
	out->d_model = d_model;
	out->n_heads = n_heads;
	out->d_ff = d_ff;
	out->max_ctx = max_ctx;
	out->n_blocks = n_blocks;
	out->vocab_size = vsize;
	merges = 0;
	if (magic == (int)LM_MAGIC_BPE) {
		if (read_i32_fp(fp, &merges) != 0) {
			fclose(fp);
			return -1;
		}
		out->is_bpe = 1;
		out->n_merges = merges;
	}
	fclose(fp);
	return 0;
}

int lm_info_from_model(lm_info *out, const char *path, const lm *model,
	const lm_adam *opt, int epoch, int step)
{
	size_t fr;
	size_t tot;
	int d;
	int V;
	int ff;
	int L;
	int block;

	if (!out || !model)
		return -1;
	if (path)
		lm_info_peek(path, out);
	else {
		memset(out, 0, sizeof(*out));
		out->prompt_nll = -1.0;
		out->prompt_ppl = -1.0;
		out->data_win_nll = -1.0;
	}
	out->d_model = LM_D_MODEL;
	out->n_heads = LM_N_HEADS;
	out->d_ff = LM_D_FF;
	out->max_ctx = LM_MAX_CTX;
	out->n_blocks = LM_N_BLOCKS;
	out->vocab_size = model->vocab_size;
	out->real_size = (int)sizeof(ann_real);
	if (model->bpe) {
		out->is_bpe = 1;
		out->n_merges = model->bpe->n_merges;
		if (out->magic_kind == LM_INFO_MAGIC_NONE)
			out->magic_kind = LM_INFO_MAGIC_LMCV;
	} else if (out->magic_kind == LM_INFO_MAGIC_NONE) {
		out->magic_kind = LM_INFO_MAGIC_LMCF;
	}
	out->params = lm_param_count(model);
	/* Cross-check against the closed formula. */
	d = LM_D_MODEL;
	V = model->vocab_size;
	ff = LM_D_FF;
	L = model->tr ? model->tr->n_blocks : LM_N_BLOCKS;
	block = 4 * d * d + d * ff + ff + ff * d + d + 4 * d;
	if (out->params <= 0)
		out->params = V * d + L * block + d * V + V;
	out->weight_bytes = (size_t)out->params * sizeof(ann_real);
	if (opt) {
		out->has_checkpoint = 1;
		out->epoch = epoch;
		out->step = step;
		out->adam_t = opt->t;
		out->adam_slots = opt->n;
	}
	out->gpu_name = gpu_backend_name();
	fr = tot = 0;
	if (gpu_mem_info(&fr, &tot) == 0) {
		out->gpu_free = fr;
		out->gpu_total = tot;
	}
	return 0;
}

static const char *magic_name(int kind)
{
	switch (kind) {
	case LM_INFO_MAGIC_LMCF:
		return "LMCF (float)";
	case LM_INFO_MAGIC_LMCV:
		return "LMCV (float+bpe)";
	case LM_INFO_MAGIC_LMCK:
		return "LMCK (legacy double)";
	case LM_INFO_MAGIC_OTHER:
		return "unknown";
	default:
		return "n/a";
	}
}

void lm_info_print(const lm_info *info, FILE *fp)
{
	if (!info || !fp)
		return;
	fprintf(fp, "path:          %s\n", info->path[0] ? info->path : "(memory)");
	if (info->file_bytes > 0)
		fprintf(fp, "file size:     %.2f MB (%ld bytes)\n",
		    (double)info->file_bytes / 1e6, info->file_bytes);
	fprintf(fp, "format:        %s\n", magic_name(info->magic_kind));
	fprintf(fp, "ann_real:      %d bytes\n", info->real_size);
	fprintf(fp, "architecture:  d=%d heads=%d ff=%d ctx=%d blocks=%d\n",
	    info->d_model, info->n_heads, info->d_ff, info->max_ctx,
	    info->n_blocks);
	if (info->is_bpe)
		fprintf(fp, "tokenizer:     bpe vocab=%d merges=%d\n",
		    info->vocab_size, info->n_merges);
	else
		fprintf(fp, "tokenizer:     byte vocab=%d\n", info->vocab_size);
	fprintf(fp, "parameters:     %d (%.3f M)\n", info->params,
	    (double)info->params / 1e6);
	fprintf(fp, "weight bytes:  %.2f MB\n",
	    (double)info->weight_bytes / 1e6);
	if (info->has_checkpoint)
		fprintf(fp, "checkpoint:    epoch=%d step=%d adam_t=%d slots=%d\n",
		    info->epoch, info->step, info->adam_t, info->adam_slots);
	else
		fprintf(fp, "checkpoint:    weights only (no Adam state)\n");
	fprintf(fp, "gpu:           %s", info->gpu_name ? info->gpu_name : "?");
	if (info->gpu_total > 0)
		fprintf(fp, "  VRAM %.0f / %.0f MB free/total",
		    (double)info->gpu_free / (1024.0 * 1024.0),
		    (double)info->gpu_total / (1024.0 * 1024.0));
	fputc('\n', fp);
	if (info->prompt_nll >= 0.0)
		fprintf(fp, "prompt NLL:    %.4f  PPL %.4f\n",
		    info->prompt_nll, info->prompt_ppl);
	if (info->data_win_nll >= 0.0)
		fprintf(fp, "data win NLL:  %.4f\n", info->data_win_nll);
}
