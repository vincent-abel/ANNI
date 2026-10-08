/*
**	Language model: embed + transformer + next-token head
*/
#ifndef LM_H__
#define LM_H__

#include "block.h"
#include "embed.h"
#include "mat.h"
#include "vocab.h"

typedef struct	lm {
	vocab		*vocab;
	embed		*embed;
	transformer	*tr;
	mat		*w_head;
	vec		*b_head;
	int		vocab_size;
} lm;

lm	*lm_create(void);
void	lm_free(lm **model);
int	lm_forward(mat *hidden, const lm *model, const int *ids, int n);
int	lm_logits_at(vec *logits, const lm *model, const mat *hidden, int t);
double	lm_nll(const lm *model, const int *ids, int n);
double	lm_train_steps(lm *model, const char *text, int steps, double lr);

#endif
