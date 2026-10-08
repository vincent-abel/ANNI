/*
**	Byte / character tokenizer
*/
#include <stdio.h>
#include <string.h>
#include "tokenize.h"

int tokenize_encode(const char *text, int *ids, int max_ids)
{
	int i;
	int n;

	if (!text || !ids || max_ids <= 0)
		return -1;
	n = (int)strlen(text);
	if (n > max_ids)
		n = max_ids;
	for (i = 0; i < n; i++)
		ids[i] = (unsigned char)text[i];
	return n;
}

int tokenize_decode(const int *ids, int n_ids, char *out, int out_cap)
{
	int i;

	if (!ids || !out || out_cap <= 0 || n_ids < 0)
		return -1;
	if (n_ids >= out_cap)
		n_ids = out_cap - 1;
	for (i = 0; i < n_ids; i++) {
		if (ids[i] < 0 || ids[i] >= LM_BYTE_VOCAB)
			out[i] = '?';
		else
			out[i] = (char)ids[i];
	}
	out[n_ids] = '\0';
	return n_ids;
}
