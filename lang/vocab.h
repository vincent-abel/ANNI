/*
**	Vocabulary
*/
#ifndef VOCAB_H__
#define VOCAB_H__

#include "lm_config.h"

#define VOCAB_PAD	0
#define VOCAB_BOS	1
#define VOCAB_EOS	2
#define VOCAB_UNK	3
#define VOCAB_SPECIALS	4

typedef struct	vocab {
	char	**id_to_token;
	int	size;
} vocab;

vocab	*vocab_byte_create(void);
/* Byte vocab padded with placeholder ids up to `size` (lm bench). */
vocab	*vocab_sized_create(int size);
void	vocab_free(vocab **v);
int	vocab_size(const vocab *v);
const char	*vocab_token(const vocab *v, int id);
int	vocab_id(const vocab *v, const char *token);

#endif
