/*
**	Token embedding table + backward
*/
#ifndef EMBED_H__
#define EMBED_H__

#include "lm_config.h"
#include "mat.h"
#include "vocab.h"

/*
** Tied: table and dtable are the LM head's D x vocab mats, borrowed, and token
** id is column id. Rows are multiplied by scale on the way in (sqrt(D) when
** tied, so a table sized for the head still counts next to the sinusoids).
*/
typedef struct	embed {
	mat		*table;
	mat		*dtable;
	int		tied;
	ann_real	scale;
} embed;

embed	*embed_create(int vocab_size);
embed	*embed_create_tied(mat *w_head, mat *dw_head);
void	embed_free(embed **e);
void	embed_zero_grad(embed *e);
int	embed_lookup(mat *out, const embed *e, const int *ids, int n);
int	embed_lookup_row(ann_real *out, const embed *e, int id);
int	embed_backward(embed *e, const mat *dout, const int *ids, int n);

#endif
