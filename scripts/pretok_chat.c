/*
** Encode User:/Assistant: chat text with a BPE vocab into masked int32 ids.
** Usage: pretok_chat VOCAB TEXT OUT_mids.bin
**
** The user's words and the "\nAssistant:" that follows them are stored as
** -(id + 1): the model reads them but is not trained to predict them. The
** "User:" marker, assistant replies and newlines keep their plain ids, so the
** model learns to answer and to end a reply with "\nUser:". Each segment is
** encoded on its own; segments start at a space or a line break, where BPE
** words split anyway.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe.h"

static int line_starts(const char *text, size_t i, const char *tag)
{
	return (i == 0 || text[i - 1] == '\n')
	    && strncmp(text + i, tag, strlen(tag)) == 0;
}

/* Encode text[a, b) and append it, negated when masked. */
static int emit(const bpe *tok, const char *text, size_t a, size_t b,
	int masked, int *ids, long *n, long cap)
{
	int k;
	int i;

	if (b <= a)
		return 0;
	k = bpe_encode_n(tok, text + a, b - a, ids + *n, (int)(cap - *n));
	if (k < 0)
		return -1;
	if (masked)
		for (i = 0; i < k; i++)
			ids[*n + i] = -ids[*n + i] - 1;
	*n += k;
	return 0;
}

int main(int argc, char **argv)
{
	bpe *tok;
	FILE *fp;
	long sz;
	char *text;
	int *ids;
	long n;
	long masked;
	size_t i;
	size_t seg;
	int in_user;
	const size_t ulen = strlen("User:");
	const size_t alen = strlen("Assistant:");

	if (argc < 4) {
		fprintf(stderr, "usage: %s VOCAB TEXT OUT_mids.bin\n", argv[0]);
		return 2;
	}
	tok = bpe_load(argv[1]);
	fp = tok ? fopen(argv[2], "rb") : NULL;
	if (!fp) {
		fprintf(stderr, "pretok_chat: cannot open %s or %s\n", argv[1],
		    argv[2]);
		return 1;
	}
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	rewind(fp);
	text = malloc((size_t)sz + 1);
	ids = malloc((size_t)(sz + 1) * sizeof(int));
	if (!text || !ids || fread(text, 1, (size_t)sz, fp) != (size_t)sz) {
		fprintf(stderr, "pretok_chat: read failed\n");
		return 1;
	}
	fclose(fp);
	text[sz] = '\0';
	n = 0;
	seg = 0;
	in_user = 0;
	for (i = 0; i < (size_t)sz; i++) {
		if (!in_user && line_starts(text, i, "User:")) {
			/* "User:" stays a target; the user's words do not. */
			if (emit(tok, text, seg, i + ulen, 0, ids, &n, sz + 1) != 0)
				return 1;
			seg = i + ulen;
			in_user = 1;
		} else if (in_user && line_starts(text, i, "Assistant:")) {
			if (emit(tok, text, seg, i + alen, 1, ids, &n, sz + 1) != 0)
				return 1;
			seg = i + alen;
			in_user = 0;
		}
	}
	if (emit(tok, text, seg, (size_t)sz, in_user, ids, &n, sz + 1) != 0)
		return 1;
	masked = 0;
	for (i = 0; i < (size_t)n; i++)
		masked += ids[i] < 0;
	fp = fopen(argv[3], "wb");
	if (!fp || fwrite(ids, sizeof(int), (size_t)n, fp) != (size_t)n) {
		perror(argv[3]);
		return 1;
	}
	fclose(fp);
	printf("wrote %ld ids (%ld masked, %.1f%%) -> %s\n", n, masked,
	    n > 0 ? 100.0 * (double)masked / (double)n : 0.0, argv[3]);
	free(text);
	free(ids);
	bpe_free(&tok);
	return 0;
}
