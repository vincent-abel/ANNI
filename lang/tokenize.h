/*
**	Byte / character tokenizer
*/
#ifndef TOKENIZE_H__
#define TOKENIZE_H__

#include "lm_config.h"

int	tokenize_encode(const char *text, int *ids, int max_ids);
int	tokenize_decode(const int *ids, int n_ids, char *out, int out_cap);

#endif
