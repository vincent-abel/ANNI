/*
**	Plain-text User/Assistant chat history + generate
*/
#ifndef CHAT_SESSION_H__
#define CHAT_SESSION_H__

#include <stddef.h>
#include "lm.h"

typedef struct	chat_session {
	lm		*model;
	char		*hist;
	size_t		hist_len;
	size_t		hist_cap;
	int		nnew;
	double		temperature;
	int		top_k;
	double		top_p;
} chat_session;

/* Owns hist only; does not free model. Returns 0 on success. */
int	chat_session_init(chat_session *s, lm *model, int nnew,
		double temperature, int top_k, double top_p);
void	chat_session_free(chat_session *s);

/* Token count of current history (0 if empty/error). */
int	chat_session_ctx_used(const chat_session *s);

/*
** Append user turn, trim, generate. Writes assistant reply (NUL-terminated)
** into reply[reply_cap]. Returns 0 on success, -1 on failure.
*/
int	chat_session_turn(chat_session *s, const char *user_line,
		char *reply, int reply_cap);

#endif
