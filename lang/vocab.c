/*
**	Vocabulary
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vocab.h"

static char *xstrdup(const char *s)
{
	size_t n;
	char *p;

	if (!s)
		return NULL;
	n = strlen(s) + 1;
	p = malloc(n);
	if (!p)
		return NULL;
	memcpy(p, s, n);
	return p;
}

vocab *vocab_byte_create(void)
{
	vocab *v;
	int i;
	char buf[8];

	v = malloc(sizeof(vocab));
	if (!v)
		return NULL;
	v->size = LM_BYTE_VOCAB + VOCAB_SPECIALS;
	v->id_to_token = calloc((size_t)v->size, sizeof(char *));
	if (!v->id_to_token) {
		free(v);
		return NULL;
	}

	v->id_to_token[VOCAB_PAD] = xstrdup("<pad>");
	v->id_to_token[VOCAB_BOS] = xstrdup("<bos>");
	v->id_to_token[VOCAB_EOS] = xstrdup("<eos>");
	v->id_to_token[VOCAB_UNK] = xstrdup("<unk>");
	for (i = 0; i < LM_BYTE_VOCAB; i++) {
		if (i >= 32 && i < 127)
			snprintf(buf, sizeof(buf), "%c", i);
		else
			snprintf(buf, sizeof(buf), "<b%d>", i);
		v->id_to_token[VOCAB_SPECIALS + i] = xstrdup(buf);
	}
	for (i = 0; i < v->size; i++) {
		if (!v->id_to_token[i]) {
			vocab_free(&v);
			return NULL;
		}
	}
	return v;
}

/*
** Placeholder vocab of `size` ids (byte vocab first, then "<tN>"), so a model
** can have a BPE-sized head without a trained tokenizer (lm bench).
*/
vocab *vocab_sized_create(int size)
{
	vocab *v;
	char buf[16];
	int i;
	int nb;

	v = vocab_byte_create();
	if (!v || size <= v->size)
		return v;
	nb = v->size;
	{
		char **grown = realloc(v->id_to_token,
		    (size_t)size * sizeof(char *));

		if (!grown) {
			vocab_free(&v);
			return NULL;
		}
		v->id_to_token = grown;
	}
	for (i = nb; i < size; i++)
		v->id_to_token[i] = NULL;
	v->size = size;
	for (i = nb; i < size; i++) {
		snprintf(buf, sizeof(buf), "<t%d>", i);
		v->id_to_token[i] = xstrdup(buf);
		if (!v->id_to_token[i]) {
			vocab_free(&v);
			return NULL;
		}
	}
	return v;
}

void vocab_free(vocab **v)
{
	int i;

	if (!v || !*v)
		return;
	if ((*v)->id_to_token) {
		for (i = 0; i < (*v)->size; i++)
			free((*v)->id_to_token[i]);
		free((*v)->id_to_token);
	}
	free(*v);
	*v = NULL;
}

int vocab_size(const vocab *v)
{
	return v ? v->size : 0;
}

const char *vocab_token(const vocab *v, int id)
{
	if (!v || id < 0 || id >= v->size)
		return NULL;
	return v->id_to_token[id];
}

int vocab_id(const vocab *v, const char *token)
{
	int i;

	if (!v || !token)
		return VOCAB_UNK;
	for (i = 0; i < v->size; i++) {
		if (strcmp(v->id_to_token[i], token) == 0)
			return i;
	}
	return VOCAB_UNK;
}
