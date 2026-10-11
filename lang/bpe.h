/*
**	Subword byte-pair-encoding (BPE) tokenizer
**
**	Ids: VOCAB_SPECIALS specials, then 256 raw bytes (same as the byte
**	tokenizer), then one id per learned merge:
**		id(merge i) = VOCAB_SPECIALS + 256 + i
**	vocab_size = VOCAB_SPECIALS + 256 + n_merges.
*/
#ifndef BPE_H__
#define BPE_H__

#include <stddef.h>
#include <stdio.h>
#include "vocab.h"

#define BPE_DEFAULT_VOCAB	16384
#define BPE_MAX_WORD		64	/* longest pre-token (bytes) */
#define BPE_BASE		(VOCAB_SPECIALS + LM_BYTE_VOCAB)

typedef struct	bpe {
	int			n_merges;
	int			vocab_size;
	int			*left;		/* n_merges */
	int			*right;
	unsigned char		**bytes;	/* vocab_size expansions */
	int			*blen;
	unsigned long long	*hkey;		/* (left,right) -> merge rank */
	int			*hval;
	int			hcap;
} bpe;

typedef struct	bpe_trainer bpe_trainer;

/* Build a tokenizer from an ordered merge list (validates ids). */
bpe	*bpe_create(const int *left, const int *right, int n_merges);
bpe	*bpe_clone(const bpe *b);
void	bpe_free(bpe **b);
int	bpe_equal(const bpe *a, const bpe *b);

/*
** Training. The trainer counts pre-tokens ("words") from any number of text
** pieces, then learns merges until vocab_size is reached (or no pair is left).
*/
bpe_trainer	*bpe_trainer_create(void);
void		bpe_trainer_free(bpe_trainer **t);
int		bpe_trainer_add(bpe_trainer *t, const char *text, size_t len);
bpe		*bpe_trainer_finish(bpe_trainer *t, int vocab_size);
/* One-shot helpers. max_bytes<=0 reads the whole file. */
bpe	*bpe_train_text(const char *text, size_t len, int vocab_size);
bpe	*bpe_train_from_file(const char *path, int vocab_size, long max_bytes);

/* Encode: NUL-terminated or explicit length. Returns #ids (truncated to max_ids). */
int	bpe_encode(const bpe *b, const char *text, int *ids, int max_ids);
int	bpe_encode_n(const bpe *b, const char *text, size_t len, int *ids,
		int max_ids);
/* Decode: specials are dropped. Returns #bytes written (NUL-terminated). */
int	bpe_decode(const bpe *b, const int *ids, int n_ids, char *out,
		int out_cap);

/* Text vocab file: "BPEV 1 <n_merges>\n" then one "<left> <right>" per line. */
int	bpe_save(const bpe *b, const char *path);
bpe	*bpe_load(const char *path);
/* Binary form embedded in LM checkpoints: i32 n_merges, then (i32,i32) pairs. */
int	bpe_write_fp(FILE *fp, const bpe *b);
bpe	*bpe_read_fp(FILE *fp);

/* Human-readable vocab table (id -> token string) for the lm->vocab field. */
vocab	*bpe_vocab_create(const bpe *b);

#endif
