/*
**	Bump / arena allocator for contiguous scratch
*/
#include <stdlib.h>
#include <string.h>
#include "arena.h"

arena *arena_create(size_t cap)
{
	arena *a;

	if (cap == 0)
		return NULL;
	a = malloc(sizeof(arena));
	if (!a)
		return NULL;
	a->buf = calloc(1, cap);
	if (!a->buf) {
		free(a);
		return NULL;
	}
	a->cap = cap;
	a->used = 0;
	return a;
}

void arena_free(arena **a)
{
	if (!a || !*a)
		return;
	free((*a)->buf);
	free(*a);
	*a = NULL;
}

void arena_reset(arena *a)
{
	if (!a)
		return;
	a->used = 0;
}

void *arena_alloc(arena *a, size_t nbytes)
{
	return arena_alloc_aligned(a, nbytes, sizeof(double));
}

void *arena_alloc_aligned(arena *a, size_t nbytes, size_t align)
{
	size_t off;
	size_t pad;
	void *p;

	if (!a || !a->buf || nbytes == 0 || align == 0)
		return NULL;
	off = a->used;
	pad = (align - (off % align)) % align;
	if (off + pad + nbytes > a->cap)
		return NULL;
	off += pad;
	p = a->buf + off;
	a->used = off + nbytes;
	memset(p, 0, nbytes);
	return p;
}
