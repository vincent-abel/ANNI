/*
**	Bump / arena allocator for contiguous scratch
*/
#ifndef ARENA_H__
#define ARENA_H__

#include <stddef.h>

typedef struct	arena {
	char	*buf;
	size_t	cap;
	size_t	used;
} arena;

arena	*arena_create(size_t cap);
void	arena_free(arena **a);
void	arena_reset(arena *a);
void	*arena_alloc(arena *a, size_t nbytes);
void	*arena_alloc_aligned(arena *a, size_t nbytes, size_t align);

#endif
