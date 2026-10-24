/*
**	Language model: embed + transformer + next-token head + backward
*/
#ifndef LM_H__
#define LM_H__

#include "block.h"
#include "bpe.h"
#include "embed.h"
#include "mat.h"
#include "quant.h"
#include "vocab.h"

typedef struct	lm {
	vocab		*vocab;
	bpe		*bpe;		/* NULL = byte tokenizer (default) */
	embed		*embed;
	transformer	*tr;
	mat		*w_head;
	vec		*b_head;
	mat		*dw_head;
	vec		*db_head;
	qmat		*w_head_q;	/* optional int8 head for inference */
	int		vocab_size;
} lm;

/*
** B sequences of T tokens packed back to back: n = B*T rows. Classic
** (single-window) caches have B = 1, T = n. Packed caches own fat (B*T)
** hidden / LN / FFN buffers plus B per-sequence attention caches.
*/
typedef struct	lm_cache {
	int			*ids;
	int			n;		/* B * T */
	int			B;
	int			T;
	mat			*hidden;
	transformer_cache	*trc;
} lm_cache;

lm	*lm_create(void);
/* Byte-tokenizer model with a vocab_size-row head (bench a BPE-sized head). */
lm	*lm_create_vocab(int vocab_size);
/* Subword model: vocab_size = tok->vocab_size. Takes ownership of tok on success. */
lm	*lm_create_bpe(bpe *tok);
/* Attach/replace the tokenizer of an existing model (sizes must match). */
int	lm_attach_bpe(lm *model, bpe *tok);
/* Tokenize with the model's tokenizer (byte or BPE). Same contract as tokenize_*. */
int	lm_encode(const lm *model, const char *text, int *ids, int max_ids);
int	lm_decode(const lm *model, const int *ids, int n_ids, char *out,
		int out_cap);
void	lm_free(lm **model);
void	lm_zero_grad(lm *model);
/*
** GPU residency (no-ops on CPU builds). Weights have persistent device twins
** pinned at create. Training: lm_weights_to_device() uploads them (state
** "synced") so GEMMs read the twin with no per-op H2D; call it once at the
** start of a run (A1: Adam then updates the twins in place, no re-upload).
** lm_grads_ensure_host() pulls device-authoritative dW to host (only the
** host Adam fallback needs it). lm_weights_host_dirty() must be called
** before any host code writes weights again (end of training, load, ...).
*/
void	lm_weights_to_device(lm *model);
void	lm_weights_host_dirty(lm *model);
void	lm_grads_ensure_host(lm *model);
/*
** A1: Adam keeps weights device-authoritative between steps (no per-step
** D2H/H2D). Call lm_weights_ensure_host() before host code reads weight
** ->data directly (save/generate/quantize; end of training does this).
** lm_wgrads_zero_device() zeroes dW host+twin so no zero upload is needed.
*/
void	lm_weights_ensure_host(lm *model);
void	lm_wgrads_zero_device(lm *model);
int	lm_forward(mat *hidden, const lm *model, const int *ids, int n);
int	lm_forward_cache(lm *model, const int *ids, int n, lm_cache *c);
int	lm_logits_at(vec *logits, const lm *model, const mat *hidden, int t);
int	lm_quantize_head(lm *model);
/* mean_ce may be NULL; otherwise receives mean softmax CE over predicting rows. */
int	lm_loss_backward(lm *model, const lm_cache *c, const int *ids, int n,
		mat *dhidden, double *mean_ce);
int	lm_backward(lm *model, const mat *dhidden, const lm_cache *c);
double	lm_nll(const lm *model, const int *ids, int n);
double	lm_perplexity(const lm *model, const int *ids, int n);
double	lm_train_steps(lm *model, const char *text, int steps, double lr);

lm_cache	*lm_cache_create(int n);
void		lm_cache_free(lm_cache **c);

/*
** Token-parallel packed training (B sequences of T tokens in one step).
** ids is B*T ints, row b*T+t = token t of sequence b. Attention is causal per
** sequence; every other layer sees one (B*T) x D GEMM. The head/CE runs one
** GEMM over the B*(T-1) predicting rows (position t predicts ids[b*T+t+1]).
** Grads are summed over all B*(T-1) rows (scale by 1/B in the optimizer).
** B == 1 matches the classic single-window path.
*/
lm_cache	*lm_cache_create_packed(int B, int T);
int	lm_forward_cache_packed(lm *model, const int *ids, int B, int T,
		lm_cache *c);
/* dhidden is (B*T) x D; the last row of each sequence gets zero gradient.
** lens optional (E8): lens[b] in [2,T] real tokens; pad rows get zero grad.
** mean_ce may be NULL; otherwise mean CE over B*(T-1) predicting rows. */
int	lm_loss_backward_packed(lm *model, const lm_cache *c, const int *ids,
		int B, int T, mat *dhidden, double *mean_ce);
int	lm_loss_backward_packed_lens(lm *model, const lm_cache *c,
		const int *ids, int B, int T, const int *lens, mat *dhidden,
		double *mean_ce);
int	lm_backward_packed(lm *model, const mat *dhidden, const lm_cache *c);

/*
** Estimated working-set bytes of one packed step (activation caches, backward
** temporaries, per-sequence attention scores, logits/delta of the head).
*/
size_t	lm_packed_bytes(int B, int T, int vocab_size);
/*
** Largest B in [1, max_b] with lm_packed_bytes(B, T, vocab) < frac * free_bytes
** (at least 1). free_bytes == 0 (unknown / no GPU) returns fallback.
*/
int	lm_batch_auto(int T, int vocab_size, size_t free_bytes, double frac,
		int max_b, int fallback);

#endif
