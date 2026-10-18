/*
** Encode a UTF-8 file with a BPE vocab → packed int32 LE ids.
** Usage: pretok_ids VOCAB.txt TEXT.txt OUT.bin [max_bytes]
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe.h"

int main(int argc, char **argv)
{
	bpe *tok;
	FILE *fp;
	long sz;
	long cap;
	char *buf;
	int *ids;
	int n;
	size_t wrote;

	if (argc < 4) {
		fprintf(stderr, "usage: %s VOCAB TEXT OUT [max_bytes]\n", argv[0]);
		return 2;
	}
	tok = bpe_load(argv[1]);
	if (!tok) {
		fprintf(stderr, "bpe_load failed: %s\n", argv[1]);
		return 1;
	}
	fp = fopen(argv[2], "rb");
	if (!fp) {
		perror(argv[2]);
		bpe_free(&tok);
		return 1;
	}
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	rewind(fp);
	cap = sz;
	if (argc >= 5 && atol(argv[4]) > 0 && atol(argv[4]) < sz)
		cap = atol(argv[4]);
	buf = malloc((size_t)cap + 1);
	ids = malloc((size_t)cap * sizeof(int));
	if (!buf || !ids) {
		free(buf);
		free(ids);
		fclose(fp);
		bpe_free(&tok);
		return 1;
	}
	if (fread(buf, 1, (size_t)cap, fp) != (size_t)cap) {
		fclose(fp);
		free(buf);
		free(ids);
		bpe_free(&tok);
		return 1;
	}
	fclose(fp);
	buf[cap] = '\0';
	n = bpe_encode_n(tok, buf, (size_t)cap, ids, (int)cap);
	free(buf);
	bpe_free(&tok);
	if (n < 2) {
		free(ids);
		fprintf(stderr, "encode produced %d ids\n", n);
		return 1;
	}
	fp = fopen(argv[3], "wb");
	if (!fp) {
		perror(argv[3]);
		free(ids);
		return 1;
	}
	wrote = fwrite(ids, sizeof(int), (size_t)n, fp);
	fclose(fp);
	free(ids);
	if (wrote != (size_t)n) {
		fprintf(stderr, "short write\n");
		return 1;
	}
	fprintf(stderr, "wrote %d ids -> %s\n", n, argv[3]);
	return 0;
}
