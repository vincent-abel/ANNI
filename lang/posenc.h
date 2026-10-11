/*
**	Sinusoidal positional encoding (Transformer)
*/
#ifndef POSENC_H__
#define POSENC_H__

#include "mat.h"
#include "lm_config.h"

void	posenc_sinusoidal_add(mat *seq);
void	posenc_add_row(ann_real *row, int t);

#endif
