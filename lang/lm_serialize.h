/*
**	LM weight (+ optional Adam) binary serialization
*/
#ifndef LM_SERIALIZE_H__
#define LM_SERIALIZE_H__

#include "lm.h"
#include "lm_adam.h"

#define LM_MAGIC	0x46434D4CUL	/* LMCF: float (ann_real) checkpoints */
#define LM_MAGIC_F64	0x4B434D4CUL	/* LMCK: legacy double (rejected) */
#define LM_MAGIC_BPE	0x56434D4CUL	/* LMCV: float + embedded BPE merge table */

int	lm_save(const lm *model, const char *path);
lm	*lm_load(const char *path);
int	lm_save_checkpoint(const lm *model, const lm_adam *opt,
		int epoch, int step, const char *path);
lm	*lm_load_checkpoint(const char *path, lm_adam **opt_out,
		int *epoch_out, int *step_out);
/*
** Average weights of n models into *out (fresh lm). Same architecture required.
** Adam state is not merged. Returns 0 on success.
*/
int	lm_merge_average(lm *const *models, int n, lm **out);
/* Weights-only or checkpoint (Adam discarded). */
lm	*lm_load_any(const char *path);

#endif
