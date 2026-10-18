/*
**	Sinusoidal positional encoding (Vaswani et al.)
*/
#include <math.h>
#include "gpu.h"
#include "mat.h"
#include "lm_config.h"
#include "posenc.h"

void posenc_add_row(ann_real *row, int t)
{
	int i;
	double log_base;
	double angle;

	if (!row || t < 0)
		return;
	log_base = -log(10000.0) / (double)LM_D_MODEL;
	for (i = 0; 2 * i < LM_D_MODEL; i++) {
		angle = (double)t * exp((2.0 * (double)i) * log_base);
		row[2 * i] += sin(angle);
		if (2 * i + 1 < LM_D_MODEL)
			row[2 * i + 1] += cos(angle);
	}
}

void posenc_sinusoidal_add(mat *seq)
{
	int t;
	ann_real *hp;

	if (!seq || seq->rows <= 0 || seq->cols <= 0)
		return;
	/* Prefer device kernel when twin is live (matches packed train). */
	hp = mat_compute_ptr(seq);
	if (hp && seq->d_data && seq->on_device != MAT_DEV_HOST
	    && gpu_posenc_add_packed(hp, 1, seq->rows, seq->cols) == 0) {
		mat_mark_device(seq);
		return;
	}
	if (mat_ensure_host(seq) != 0 || !seq->data)
		return;
	for (t = 0; t < seq->rows; t++)
		posenc_add_row(&seq->data[t * seq->cols], t);
	mat_host_dirty(seq);
}
