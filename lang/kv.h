/*
**	KV cache for incremental LM decode
*/
#ifndef KV_H__
#define KV_H__

#include "lm.h"
#include "mat.h"

typedef struct	lm_kv {
	mat	**k;	/* n_blocks mats: max_T x D_MODEL */
	mat	**v;
	int	n_blocks;
	int	max_T;
	int	len;	/* tokens currently cached */
} lm_kv;

lm_kv	*lm_kv_create(const lm *model, int max_T);
void	lm_kv_free(lm_kv **kv);
void	lm_kv_reset(lm_kv *kv);
/* Prefill prompt (full forward) then append tokens via decode steps. */
int	lm_generate_kv(lm *model, const char *prompt, char *out, int out_cap,
		int max_new, double temperature);

#endif
