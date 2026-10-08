/*
**	Autoregressive text generation
*/
#ifndef GENERATE_H__
#define GENERATE_H__

#include "lm.h"

int	lm_generate(lm *model, const char *prompt, char *out, int out_cap,
		int max_new, double temperature);

#endif
