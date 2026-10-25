/*
**	UTF-8 text file → token id windows for LM training
*/
#ifndef DATASET_TEXT_H__
#define DATASET_TEXT_H__

#include "bpe.h"
#include "lm_config.h"

/*
** starts == NULL: every sliding window [i, i+window) is valid (default).
** starts != NULL: only the listed window start offsets (n_starts of them) are
** valid, so windows can be kept inside documents (see dataset_stream.h).
*/
typedef struct	text_dataset {
	int	*ids;
	long	n_ids;	/* long: pretokenized mixes exceed 2^31 ids */
	int	window;
	int	stride;	/* start step; 1 = every offset, window = non-overlap */
	int	*starts;
	int	n_starts;
	void	*map;	/* ids mapped read-only from an id file, or NULL */
	int	masked;	/* *_mids.bin: ids < 0 are -(id+1), no-loss tokens */
	size_t	map_len;
} text_dataset;

/* Byte tokenizer. */
text_dataset	*text_dataset_load(const char *path, int window);
/* tok == NULL: byte tokenizer; otherwise BPE ids. */
text_dataset	*text_dataset_load_tok(const char *path, int window,
			const bpe *tok);
/*
** Pretokenized int32 LE id file (e.g. data/wiki_ids.bin). Through
** text_dataset_load_tok, a *_mids.bin file loads with masked = 1.
*/
text_dataset	*text_dataset_load_ids(const char *path, int window);
void		text_dataset_free(text_dataset **ds);
/* stride<=0 → 1; stride>=window → non-overlapping chunks (wiki epoch). */
void		text_dataset_set_stride(text_dataset *ds, int stride);
int		text_dataset_n_windows(const text_dataset *ds);
int		text_dataset_window(const text_dataset *ds, int index,
			int *ids_out, int *n_out);
/*
** Like text_dataset_window; loss_out (window bytes, may be NULL) gets 1 for
** tokens that count in the loss, 0 for masked ones (chat user turns).
*/
int		text_dataset_window_mask(const text_dataset *ds, int index,
			int *ids_out, unsigned char *loss_out, int *n_out);
/* First min(n_ids, max_ids) decoded ids; returns the count. */
long		text_dataset_head(const text_dataset *ds, int *ids_out,
			long max_ids);

#endif
