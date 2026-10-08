/*
**	Sinusoidal positional encoding (Vaswani et al.)
*/
#include <math.h>
#include "mat.h"
#include "lm_config.h"
#include "posenc.h"

void posenc_sinusoidal_add(mat *seq)
{
	int t;
	int i;
	int T;
	int d;
	double log_base;
	double angle;

	if (!seq || !seq->data || seq->rows <= 0 || seq->cols <= 0)
		return;
	T = seq->rows;
	d = seq->cols;
	log_base = -log(10000.0) / (double)d;
	for (t = 0; t < T; t++) {
		for (i = 0; 2 * i < d; i++) {
			angle = (double)t * exp((2.0 * (double)i) * log_base);
			mat_set(seq, t, 2 * i,
			    mat_get(seq, t, 2 * i) + sin(angle));
			if (2 * i + 1 < d)
				mat_set(seq, t, 2 * i + 1,
				    mat_get(seq, t, 2 * i + 1) + cos(angle));
		}
	}
}
