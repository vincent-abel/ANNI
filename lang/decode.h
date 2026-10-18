/*
**	Device-resident incremental decode (KV cache on the GPU)
*/
#ifndef DECODE_H__
#define DECODE_H__

#include "gpu.h"
#include "lm.h"
#include "vec.h"

typedef struct	lm_decoder {
	lm	*model;
	gpu_dec	*dec;
	int	len;	/* tokens already in the K/V cache */
} lm_decoder;

/* NULL when there is no GPU or an upload fails (use the full forward). */
lm_decoder	*lm_decoder_create(lm *model);
void		lm_decoder_free(lm_decoder **d);
/*
** Full forward over the prompt once, K/V of every layer into the device
** cache, logits of the last prompt row into logits.
*/
int	lm_decoder_prefill(lm_decoder *d, const int *ids, int n, vec *logits);
/* Feed id at position d->len; logits for the next token. */
int	lm_decoder_step(lm_decoder *d, int id, vec *logits);

#endif
