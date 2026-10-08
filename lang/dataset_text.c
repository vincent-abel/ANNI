/*
**	UTF-8 text file → token id windows for LM training
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataset_text.h"
#include "tokenize.h"

text_dataset *text_dataset_load(const char *path, int window)
{
	FILE *fp;
	long sz;
	char *buf;
	text_dataset *ds;
	int n;

	if (!path || window < 2 || window > LM_MAX_CTX)
		return NULL;
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

	ds = malloc(sizeof(text_dataset));
	if (!ds) {
		free(buf);
		return NULL;
	}
	ds->window = window;
	ds->ids = malloc((size_t)sz * sizeof(int));
	if (!ds->ids) {
		free(buf);
		free(ds);
		return NULL;
	}
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
	free((*ds)->ids);
	free(*ds);
	*ds = NULL;
}

int text_dataset_n_windows(const text_dataset *ds)
{
	if (!ds || ds->n_ids < ds->window)
		return 0;
	return ds->n_ids - ds->window + 1;
}

int text_dataset_window(const text_dataset *ds, int index, int *ids_out, int *n_out)
{
	int i;

	if (!ds || !ids_out || !n_out || index < 0)
		return -1;
	if (index >= text_dataset_n_windows(ds))
		return -1;
	for (i = 0; i < ds->window; i++)
		ids_out[i] = ds->ids[index + i];
	*n_out = ds->window;
	return 0;
}
