/*
**	LM weight (+ optional Adam) binary serialization
*/
#ifndef LM_SERIALIZE_H__
#define LM_SERIALIZE_H__

#include "lm.h"
#include "lm_adam.h"

#define LM_MAGIC	0x4B434D4CUL	/* LMCK */

int	lm_save(const lm *model, const char *path);
lm	*lm_load(const char *path);
int	lm_save_checkpoint(const lm *model, const lm_adam *opt,
		int epoch, int step, const char *path);
lm	*lm_load_checkpoint(const char *path, lm_adam **opt_out,
		int *epoch_out, int *step_out);

#endif
