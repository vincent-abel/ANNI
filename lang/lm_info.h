/*
**	Checkpoint / model intel (architecture, params, tokenizer, meta)
*/
#ifndef LM_INFO_H__
#define LM_INFO_H__

#include "lm.h"
#include "lm_adam.h"
#include <stddef.h>

#define LM_INFO_MAGIC_NONE	0
#define LM_INFO_MAGIC_LMCF	1
#define LM_INFO_MAGIC_LMCV	2
#define LM_INFO_MAGIC_LMCK	3
#define LM_INFO_MAGIC_OTHER	4

typedef struct	lm_info {
	char		path[512];
	long		file_bytes;
	int		magic_kind;	/* LM_INFO_MAGIC_* */
	int		real_size;
	int		d_model;
	int		n_heads;
	int		d_ff;
	int		max_ctx;
	int		n_blocks;
	int		vocab_size;
	int		n_merges;	/* 0 if byte tokenizer */
	int		is_bpe;
	int		params;
	size_t		weight_bytes;
	int		has_checkpoint;
	int		epoch;
	int		step;
	int		adam_t;
	int		adam_slots;
	const char	*gpu_name;
	size_t		gpu_free;
	size_t		gpu_total;
	double		prompt_nll;
	double		prompt_ppl;
	double		data_win_nll;
} lm_info;

/*
** Peek file header without constructing a full model. Returns 0 on success.
** LMCK (legacy double) sets magic_kind and returns 0 with params=0.
*/
int	lm_info_peek(const char *path, lm_info *out);
/*
** Fill intel from a loaded model (+ optional checkpoint meta and metrics).
** prompt_nll/ppl and data_win_nll left <0 if not computed by the caller.
*/
int	lm_info_from_model(lm_info *out, const char *path, const lm *model,
		const lm_adam *opt, int epoch, int step);
void	lm_info_print(const lm_info *info, FILE *fp);

#endif
