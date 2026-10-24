/*
** Train a BPE vocab on a UTF-8 text file and save it.
** Usage: bpe_train TEXT.txt VOCAB_SIZE OUT.vocab [max_bytes]
*/
#include <stdio.h>
#include <stdlib.h>
#include "bpe.h"

int main(int argc, char **argv)
{
	bpe *tok;
	long max_bytes;

	if (argc < 4) {
		fprintf(stderr, "usage: %s TEXT VOCAB_SIZE OUT [max_bytes]\n",
		    argv[0]);
		return 2;
	}
	max_bytes = argc >= 5 ? atol(argv[4]) : 0;
	tok = bpe_train_from_file(argv[1], atoi(argv[2]), max_bytes);
	if (!tok) {
		fprintf(stderr, "bpe_train_from_file failed: %s\n", argv[1]);
		return 1;
	}
	if (bpe_save(tok, argv[3]) != 0) {
		fprintf(stderr, "bpe_save failed: %s\n", argv[3]);
		bpe_free(&tok);
		return 1;
	}
	printf("vocab %d tokens (%d merges) -> %s\n", tok->vocab_size,
	    tok->n_merges, argv[3]);
	bpe_free(&tok);
	return 0;
}
