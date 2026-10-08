/*
**	UTF-8 text file → token id windows for LM training
*/
#ifndef DATASET_TEXT_H__
#define DATASET_TEXT_H__

#include "lm_config.h"

typedef struct	text_dataset {
	int	*ids;
	int	n_ids;
	int	window;
} text_dataset;

text_dataset	*text_dataset_load(const char *path, int window);
void		text_dataset_free(text_dataset **ds);
int		text_dataset_n_windows(const text_dataset *ds);
int		text_dataset_window(const text_dataset *ds, int index,
			int *ids_out, int *n_out);

#endif
