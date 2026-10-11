/*
**	Held-out document split + streaming/packing window iterator
**
**	A "document" is a run of non-blank lines; blank lines separate documents.
**	Splits are decided per document (never per window), so train and val
**	windows can never overlap.
*/
#ifndef DATASET_STREAM_H__
#define DATASET_STREAM_H__

#include <stddef.h>
#include "bpe.h"
#include "dataset_text.h"

#define TEXT_SPLIT_NONE		0	/* everything is train */
#define TEXT_SPLIT_DOC		1	/* val iff hash(document, seed) falls in val_frac */
#define TEXT_SPLIT_OFFSET	2	/* val = documents starting in the last val_frac bytes */

#define TEXT_SIDE_ALL		0
#define TEXT_SIDE_TRAIN		1
#define TEXT_SIDE_VAL		2

typedef struct	text_split {
	int		mode;
	double		val_frac;
	unsigned int	seed;
	long		file_size;
} text_split;

/* mode: TEXT_SPLIT_*; val_frac in [0,1). Returns 0 on success. */
int	text_split_init(text_split *sp, const char *path, int mode,
		double val_frac, unsigned int seed);
/* 1 if the document belongs to validation. start_off = byte offset in file. */
int	text_split_is_val(const text_split *sp, const char *doc, size_t len,
		long start_off);

/* Blank-line separated document reader (bounded memory per document). */
typedef struct text_docs text_docs;
text_docs	*text_docs_open(const char *path);
void		text_docs_free(text_docs **d);
int		text_docs_rewind(text_docs *d);
/* 1 = got a document (valid until next call), 0 = EOF, -1 = error. */
int		text_docs_next(text_docs *d, const char **doc, size_t *len,
			long *start_off);

/*
** In-memory loader: tokenizes each document separately and keeps only windows
** that lie inside one document. tok NULL = byte tokenizer. *val is NULL when
** the split puts no window in validation. Returns 0 on success.
*/
int	text_dataset_load_split(const char *path, int window, const bpe *tok,
		const text_split *sp, text_dataset **train, text_dataset **val);
/* Train BPE merges on the train side of the split only (max_bytes<=0: all). */
bpe	*text_split_train_bpe(const char *path, const text_split *sp,
		int vocab_size, long max_bytes);

/*
** Streaming/packing iterator. Documents are read lazily, tokenized, joined with
** VOCAB_EOS between them, and cut into windows every `stride` ids (stride<=0
** means window-1: each next-token target is used exactly once). Memory is
** O(shuffle_buf * window + largest document), independent of corpus size.
** shuffle_buf>0 draws windows at random from a buffer of that many windows.
*/
typedef struct text_stream text_stream;
text_stream	*text_stream_open(const char *path, int window, const bpe *tok,
			const text_split *sp, int side, int stride,
			int shuffle_buf, unsigned int seed);
void		text_stream_free(text_stream **s);
/* 1 = window written to ids[window], 0 = end of epoch, -1 = error. */
int		text_stream_next(text_stream *s, int *ids, int *n);
/* Start a new epoch (reopens at the file start, reseeds the shuffle). */
int		text_stream_rewind(text_stream *s);
int		text_stream_window(const text_stream *s);
int		text_stream_stride(const text_stream *s);
long		text_stream_count(const text_stream *s);	/* this epoch */
long		text_stream_last_count(const text_stream *s);	/* previous epoch */

#endif
