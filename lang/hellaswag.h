#ifndef HELLASWAG_H
#define HELLASWAG_H

#include "lm.h"

/*
** HellaSwag by log-likelihood: each question has a context and four endings,
** and the model picks the ending it finds the most probable.
** acc:       highest sum of log-probabilities over the ending's tokens
** acc_norm:  the same divided by the ending's bytes, the usual headline
** nll_token: mean loss per token of the right endings
** Chance is 0.25 for acc and acc_norm.
*/
typedef struct	hellaswag_score {
	int	questions;
	double	acc;
	double	acc_norm;
	double	nll_token;
}		hellaswag_score;

/* path: lines of answer, context and four endings, separated by tabs (see
** scripts/hellaswag_data.py). limit > 0 keeps the first questions. */
int	lm_hellaswag(lm *model, const char *path, int limit,
		hellaswag_score *score);

#endif
