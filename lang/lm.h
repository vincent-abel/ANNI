/*
**	Language model: embed + transformer + next-token head + backward
*/
#ifndef LM_H__
#define LM_H__

#include "block.h"
#include "embed.h"
#include "mat.h"
#include "quant.h"
#include "vocab.h"

typedef struct	lm {
	vocab		*vocab;
	embed		*embed;
	transformer	*tr;
	mat		*w_head;
	vec		*b_head;
	mat		*dw_head;
	vec		*db_head;
	qmat		*w_head_q;	/* optional int8 head for inference */
	int		vocab_size;
} lm;

typedef struct	lm_cache {
	int			*ids;
	int			n;
	mat			*hidden;
	transformer_cache	*trc;
} lm_cache;

lm	*lm_create(void);
void	lm_free(lm **model);
void	lm_zero_grad(lm *model);
int	lm_forward(mat *hidden, const lm *model, const int *ids, int n);
int	lm_forward_cache(lm *model, const int *ids, int n, lm_cache *c);
int	lm_logits_at(vec *logits, const lm *model, const mat *hidden, int t);
int	lm_quantize_head(lm *model);
int	lm_loss_backward(lm *model, const lm_cache *c, const int *ids, int n,
		mat *dhidden);
int	lm_backward(lm *model, const mat *dhidden, const lm_cache *c);
double	lm_nll(const lm *model, const int *ids, int n);
double	lm_perplexity(const lm *model, const int *ids, int n);
double	lm_train_steps(lm *model, const char *text, int steps, double lr);

lm_cache	*lm_cache_create(int n);
void		lm_cache_free(lm_cache **c);

#endif
