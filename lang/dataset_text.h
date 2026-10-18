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
	int	n_ids;
	int	window;
	int	stride;	/* start step; 1 = every offset, window = non-overlap */
	int	*starts;
	int	n_starts;
} text_dataset;

/* Byte tokenizer. */
text_dataset	*text_dataset_load(const char *path, int window);
/* tok == NULL: byte tokenizer; otherwise BPE ids. */
text_dataset	*text_dataset_load_tok(const char *path, int window,
			const bpe *tok);
/* Pretokenized int32 LE id file (e.g. data/wiki_ids.bin). */
text_dataset	*text_dataset_load_ids(const char *path, int window);
void		text_dataset_free(text_dataset **ds);
/* stride<=0 → 1; stride>=window → non-overlapping chunks (wiki epoch). */
void		text_dataset_set_stride(text_dataset *ds, int stride);
int		text_dataset_n_windows(const text_dataset *ds);
int		text_dataset_window(const text_dataset *ds, int index,
			int *ids_out, int *n_out);

#endif
