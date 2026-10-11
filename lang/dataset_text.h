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
	int	*starts;
	int	n_starts;
} text_dataset;

/* Byte tokenizer. */
text_dataset	*text_dataset_load(const char *path, int window);
/* tok == NULL: byte tokenizer; otherwise BPE ids. */
text_dataset	*text_dataset_load_tok(const char *path, int window,
			const bpe *tok);
void		text_dataset_free(text_dataset **ds);
int		text_dataset_n_windows(const text_dataset *ds);
int		text_dataset_window(const text_dataset *ds, int index,
			int *ids_out, int *n_out);

#endif
