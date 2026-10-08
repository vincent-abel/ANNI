/*
**	Mini-batch corpus training for the LM
*/
#ifndef LM_TRAIN_H__
#define LM_TRAIN_H__

#include "dataset_text.h"
#include "lm.h"
#include "lm_adam.h"

double	lm_train_corpus(lm *model, text_dataset *ds, lm_adam *opt,
		int epochs, int batch_size, double lr, int debug);
/* Mean NLL over every training window (ds->window bytes), not the long ctx score. */
double	lm_nll_windows(const lm *model, const text_dataset *ds);

#endif
