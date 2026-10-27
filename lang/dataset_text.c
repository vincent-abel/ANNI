/*
**	UTF-8 text file → token id windows for LM training
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include "dataset_text.h"
#include "tokenize.h"

text_dataset *text_dataset_load(const char *path, int window)
{
	return text_dataset_load_tok(path, window, NULL);
}

static int path_ends_with(const char *path, const char *suf)
{
	size_t np;
	size_t ns;

	if (!path || !suf)
		return 0;
	np = strlen(path);
	ns = strlen(suf);
	return np >= ns && strcmp(path + np - ns, suf) == 0;
}

#ifndef _WIN32
/*
** Map the id file instead of copying it: a multi-GB mix then lives once in
** the page cache, not twice (cache + heap). Returns NULL to use fread.
*/
static text_dataset *load_ids_mmap(const char *path, int window)
{
	int fd;
	struct stat st;
	void *map;
	text_dataset *ds;

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return NULL;
	if (fstat(fd, &st) != 0 || st.st_size < 2 * (off_t)sizeof(int)
	    || (st.st_size % (off_t)sizeof(int)) != 0) {
		close(fd);
		return NULL;
	}
	map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (map == MAP_FAILED)
		return NULL;
	ds = calloc(1, sizeof(text_dataset));
	if (!ds) {
		munmap(map, (size_t)st.st_size);
		return NULL;
	}
	ds->ids = (int *)map;
	ds->map = map;
	ds->map_len = (size_t)st.st_size;
	ds->n_ids = (long)((size_t)st.st_size / sizeof(int));
	ds->window = window;
	ds->stride = 1;
	return ds;
}
#endif

text_dataset *text_dataset_load_ids(const char *path, int window)
{
	FILE *fp;
	long sz;
	text_dataset *ds;
	size_t n;

	if (!path || window < 2 || window > LM_MAX_CTX)
		return NULL;
#ifndef _WIN32
	ds = load_ids_mmap(path, window);
	if (ds)
		return ds;
#endif
	fp = fopen(path, "rb");
	if (!fp)
		return NULL;
	if (fseek(fp, 0, SEEK_END) != 0) {
		fclose(fp);
		return NULL;
	}
	sz = ftell(fp);
	if (sz <= 0 || (sz % (long)sizeof(int)) != 0) {
		fclose(fp);
		return NULL;
	}
	rewind(fp);
	n = (size_t)sz / sizeof(int);
	ds = calloc(1, sizeof(text_dataset));
	if (!ds) {
		fclose(fp);
		return NULL;
	}
	ds->window = window;
	ds->stride = 1;
	ds->starts = NULL;
	ds->n_starts = 0;
	ds->ids = malloc(n * sizeof(int));
	if (!ds->ids) {
		free(ds);
		fclose(fp);
		return NULL;
	}
	if (fread(ds->ids, sizeof(int), n, fp) != n) {
		fclose(fp);
		text_dataset_free(&ds);
		return NULL;
	}
	fclose(fp);
	if (n < 2) {
		text_dataset_free(&ds);
		return NULL;
	}
	ds->n_ids = (long)n;
	return ds;
}

text_dataset *text_dataset_load_tok(const char *path, int window, const bpe *tok)
{
	FILE *fp;
	long sz;
	char *buf;
	text_dataset *ds;
	int n;

	if (!path || window < 2 || window > LM_MAX_CTX)
		return NULL;
	/* Pretokenized BPE id dumps skip encode on the train thread. */
	if (path_ends_with(path, "_mids.bin")) {
		ds = text_dataset_load_ids(path, window);
		if (ds)
			ds->masked = 1;
		return ds;
	}
	if (path_ends_with(path, "_ids.bin") || path_ends_with(path, "ids.bin")
	    || path_ends_with(path, "wiki_ids.bin"))
		return text_dataset_load_ids(path, window);
	fp = fopen(path, "rb");
	if (!fp)
		return NULL;
	if (fseek(fp, 0, SEEK_END) != 0) {
		fclose(fp);
		return NULL;
	}
	sz = ftell(fp);
	if (sz <= 0) {
		fclose(fp);
		return NULL;
	}
	rewind(fp);
	buf = malloc((size_t)sz + 1);
	if (!buf) {
		fclose(fp);
		return NULL;
	}
	if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
		free(buf);
		fclose(fp);
		return NULL;
	}
	fclose(fp);
	buf[sz] = '\0';

	ds = calloc(1, sizeof(text_dataset));
	if (!ds) {
		free(buf);
		return NULL;
	}
	ds->window = window;
	ds->stride = 1;
	ds->starts = NULL;
	ds->n_starts = 0;
	ds->ids = malloc((size_t)sz * sizeof(int));
	if (!ds->ids) {
		free(buf);
		free(ds);
		return NULL;
	}
	if (tok)
		n = bpe_encode_n(tok, buf, (size_t)sz, ds->ids, (int)sz);
	else
		n = tokenize_encode(buf, ds->ids, (int)sz);
	free(buf);
	if (n < 2) {
		text_dataset_free(&ds);
		return NULL;
	}
	ds->n_ids = n;
	return ds;
}

void text_dataset_free(text_dataset **ds)
{
	if (!ds || !*ds)
		return;
#ifndef _WIN32
	if ((*ds)->map)
		munmap((*ds)->map, (*ds)->map_len);
	else
#endif
		free((*ds)->ids);
	free((*ds)->starts);
	free(*ds);
	*ds = NULL;
}

void text_dataset_set_stride(text_dataset *ds, int stride)
{
	if (!ds)
		return;
	if (stride <= 0)
		ds->stride = 1;
	else if (stride > ds->window)
		ds->stride = ds->window;
	else
		ds->stride = stride;
}

int text_dataset_n_windows(const text_dataset *ds)
{
	long stride;
	long span;

	if (ds && ds->starts)
		return ds->n_starts;
	if (!ds || ds->n_ids < ds->window)
		return 0;
	stride = ds->stride > 0 ? ds->stride : 1;
	span = ds->n_ids - ds->window;
	/* Window indices are int: clamp; use --stride window on huge id files. */
	if (span / stride >= INT_MAX)
		return INT_MAX;
	return (int)(span / stride + 1);
}

int text_dataset_window(const text_dataset *ds, int index, int *ids_out, int *n_out)
{
	return text_dataset_window_mask(ds, index, ids_out, NULL, n_out);
}

int text_dataset_window_mask(const text_dataset *ds, int index, int *ids_out,
	unsigned char *loss_out, int *n_out)
{
	int i;
	int v;
	long start;
	long stride;

	if (!ds || !ids_out || !n_out || index < 0)
		return -1;
	if (index >= text_dataset_n_windows(ds))
		return -1;
	if (ds->starts)
		start = ds->starts[index];
	else {
		stride = ds->stride > 0 ? ds->stride : 1;
		start = (long)index * stride;
	}
	for (i = 0; i < ds->window; i++) {
		v = ds->ids[start + i];
		/* Masked files store no-loss tokens as -(id + 1). */
		if (ds->masked && v < 0)
			ids_out[i] = -v - 1;
		else
			ids_out[i] = v;
		if (loss_out)
			loss_out[i] = !(ds->masked && v < 0);
	}
	*n_out = ds->window;
	return 0;
}

long text_dataset_head(const text_dataset *ds, int *ids_out, long max_ids)
{
	long n;
	long i;
	int v;

	if (!ds || !ids_out || max_ids <= 0)
		return 0;
	n = ds->n_ids < max_ids ? ds->n_ids : max_ids;
	for (i = 0; i < n; i++) {
		v = ds->ids[i];
		ids_out[i] = ds->masked && v < 0 ? -v - 1 : v;
	}
	return n;
}
