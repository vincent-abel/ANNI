/*
**	Autoregressive text generation
*/
#ifndef GENERATE_H__
#define GENERATE_H__

#include "lm.h"
#include "vec.h"

/* temperature<=0 → greedy. top_k<=0 → off. top_p<=0 or >=1 → off. */
int	lm_sample_id(const vec *logits, double temperature, int top_k,
		double top_p);
int	lm_generate(lm *model, const char *prompt, char *out, int out_cap,
		int max_new, double temperature, int top_k, double top_p);
/* Stream pieces via on_text; used by the TTY meter and chat. */
int	lm_generate_stream(lm *model, const char *prompt, int max_new,
		double temperature, int top_k, double top_p,
		void (*on_text)(const char *s, int n, void *ud), void *ud);
/* E13: run generate independently for each of n prompts (batch over sequences). */
int	lm_generate_batch(lm *model, const char *const *prompts, char **outs,
		int *out_caps, int n, int max_new, double temperature,
		int top_k, double top_p);

#endif
