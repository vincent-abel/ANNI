/*
**	Plain-text chat session (User: / Assistant: prefixes)
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chat_session.h"
#include "generate.h"
#include "lm_config.h"

static void chat_trim_hist(lm *model, char *hist, size_t *hist_len, int budget)
{
	int ids[LM_MAX_CTX];
	int n;
	char *nl;

	if (!model || !hist || !hist_len || budget < 8)
		return;
	for (;;) {
		n = lm_encode(model, hist, ids, LM_MAX_CTX);
		if (n <= budget || *hist_len == 0)
			return;
		nl = strchr(hist, '\n');
		if (nl && nl[1] != '\0') {
			memmove(hist, nl + 1,
			    *hist_len - (size_t)(nl + 1 - hist) + 1);
			*hist_len = strlen(hist);
		} else {
			hist[0] = '\0';
			*hist_len = 0;
			return;
		}
	}
}

static int hist_grow(chat_session *s, size_t need)
{
	char *grown;
	size_t ncap;

	if (need <= s->hist_cap)
		return 0;
	ncap = s->hist_cap ? s->hist_cap : 16384;
	while (ncap < need)
		ncap *= 2;
	grown = realloc(s->hist, ncap);
	if (!grown)
		return -1;
	s->hist = grown;
	s->hist_cap = ncap;
	return 0;
}

int chat_session_init(chat_session *s, lm *model, int nnew,
	double temperature, int top_k, double top_p)
{
	if (!s || !model)
		return -1;
	memset(s, 0, sizeof(*s));
	s->model = model;
	if (nnew < 1)
		nnew = 64;
	if (nnew > LM_MAX_CTX - 8)
		nnew = LM_MAX_CTX - 8;
	s->nnew = nnew;
	s->temperature = temperature;
	s->top_k = top_k < 0 ? 0 : top_k;
	s->top_p = top_p;
	s->hist_cap = 16384;
	s->hist = malloc(s->hist_cap);
	if (!s->hist)
		return -1;
	s->hist[0] = '\0';
	s->hist_len = 0;
	return 0;
}

void chat_session_free(chat_session *s)
{
	if (!s)
		return;
	free(s->hist);
	s->hist = NULL;
	s->hist_len = 0;
	s->hist_cap = 0;
	s->model = NULL;
}

int chat_session_ctx_used(const chat_session *s)
{
	int ids[LM_MAX_CTX];

	if (!s || !s->model || !s->hist || s->hist_len == 0)
		return 0;
	return lm_encode(s->model, s->hist, ids, LM_MAX_CTX);
}

int chat_session_turn(chat_session *s, const char *user_line,
	char *reply, int reply_cap)
{
	char prompt[16384];
	char out[16384];
	const char *r;
	size_t need;
	size_t plen;
	int budget;

	if (!s || !s->model || !s->hist || !user_line || !reply || reply_cap < 2)
		return -1;
	reply[0] = '\0';

	need = s->hist_len + strlen("User: ") + strlen(user_line)
	    + strlen("\nAssistant:") + 1;
	if (hist_grow(s, need) != 0)
		return -1;
	if (s->hist_len > 0 && s->hist[s->hist_len - 1] != '\n') {
		s->hist[s->hist_len++] = '\n';
		s->hist[s->hist_len] = '\0';
	}
	s->hist_len += (size_t)snprintf(s->hist + s->hist_len,
	    s->hist_cap - s->hist_len, "User: %s\nAssistant:", user_line);

	budget = LM_MAX_CTX - s->nnew;
	if (budget < 8)
		budget = 8;
	chat_trim_hist(s->model, s->hist, &s->hist_len, budget);
	if (s->hist_len >= sizeof(prompt))
		return -1;
	memcpy(prompt, s->hist, s->hist_len + 1);
	plen = s->hist_len;

	if (lm_generate(s->model, prompt, out, (int)sizeof(out), s->nnew,
		s->temperature, s->top_k, s->top_p) < 0)
		return -1;
	if (strncmp(out, prompt, plen) == 0)
		r = out + plen;
	else
		r = out;
	while (*r == ' ' || *r == '\n')
		r++;
	snprintf(reply, (size_t)reply_cap, "%s", r);

	need = s->hist_len + strlen(reply) + 2;
	if (hist_grow(s, need) != 0)
		return -1;
	s->hist_len += (size_t)snprintf(s->hist + s->hist_len,
	    s->hist_cap - s->hist_len, "%s\n", reply);
	return 0;
}
