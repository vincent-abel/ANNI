/*
**	Held-out document split + streaming/packing window iterator
*/
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include "ann_posix.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "dataset_stream.h"
#include "tokenize.h"
#include "vocab.h"

#define DOC_MAX_BYTES	(8L * 1024 * 1024)	/* force-cut runaway "documents" */
#define LINE_BUF	4096

/* ------------------------------------------------------------------ */
/* Split policy                                                        */
/* ------------------------------------------------------------------ */

int text_split_init(text_split *sp, const char *path, int mode,
	double val_frac, unsigned int seed)
{
	FILE *fp;

	if (!sp || !path || mode < TEXT_SPLIT_NONE || mode > TEXT_SPLIT_OFFSET
	    || val_frac < 0.0 || val_frac >= 1.0)
		return -1;
	sp->mode = mode;
	sp->val_frac = val_frac;
	sp->seed = seed;
	sp->file_size = 0;
	fp = fopen(path, "rb");
	if (!fp)
		return -1;
	if (fseek(fp, 0, SEEK_END) == 0)
		sp->file_size = ftell(fp);
	fclose(fp);
	if (sp->file_size <= 0)
		return -1;
	return 0;
}

static unsigned long long doc_hash(const char *doc, size_t len,
	unsigned int seed)
{
	unsigned long long h;
	size_t i;

	h = 14695981039346656037ULL ^ (unsigned long long)seed;
	for (i = 0; i < len; i++) {
		h ^= (unsigned char)doc[i];
		h *= 1099511628211ULL;
	}
	h ^= h >> 33;
	h *= 0xff51afd7ed558ccdULL;
	h ^= h >> 33;
	return h;
}

int text_split_is_val(const text_split *sp, const char *doc, size_t len,
	long start_off)
{
	long cut;

	if (!sp || sp->val_frac <= 0.0)
		return 0;
	if (sp->mode == TEXT_SPLIT_DOC)
		return (long)(doc_hash(doc, len, sp->seed) % 10000ULL)
		    < (long)(sp->val_frac * 10000.0 + 0.5);
	if (sp->mode == TEXT_SPLIT_OFFSET) {
		cut = (long)((double)sp->file_size * (1.0 - sp->val_frac));
		return start_off >= cut;
	}
	return 0;
}

static int side_wants(const text_split *sp, int side, const char *doc,
	size_t len, long start_off)
{
	int v;

	if (side == TEXT_SIDE_ALL)
		return 1;
	v = text_split_is_val(sp, doc, len, start_off);
	return side == TEXT_SIDE_VAL ? v : !v;
}

/* ------------------------------------------------------------------ */
/* Document reader                                                     */
/* ------------------------------------------------------------------ */

struct text_docs {
	FILE	*fp;
	char	*buf;
	size_t	cap;
	long	pos;
	int	at_line_start;
	/* E14: mmap whole file when possible (map!=MAP_FAILED). */
	char	*map;
	size_t	map_len;
	size_t	map_off;
};

text_docs *text_docs_open(const char *path)
{
	text_docs *d;

	if (!path)
		return NULL;
	d = calloc(1, sizeof(text_docs));
	if (!d)
		return NULL;
	d->map = MAP_FAILED;
	d->cap = 4096;
	d->buf = malloc(d->cap);
	d->at_line_start = 1;
#ifndef _WIN32
	{
		int fd;
		struct stat st;

		/* E14: mmap whole file when possible (Windows uses fread path). */
		fd = open(path, O_RDONLY);
		if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 0) {
			d->map = mmap(NULL, (size_t)st.st_size, PROT_READ,
			    MAP_PRIVATE, fd, 0);
			if (d->map != MAP_FAILED) {
				d->map_len = (size_t)st.st_size;
				close(fd);
				if (!d->buf) {
					text_docs_free(&d);
					return NULL;
				}
				return d;
			}
			close(fd);
		}
	}
#endif
	d->fp = fopen(path, "rb");
	if (!d->fp || !d->buf) {
		text_docs_free(&d);
		return NULL;
	}
	return d;
}

void text_docs_free(text_docs **d)
{
	if (!d || !*d)
		return;
#ifndef _WIN32
	if ((*d)->map != MAP_FAILED && (*d)->map)
		munmap((*d)->map, (*d)->map_len);
#endif
	if ((*d)->fp)
		fclose((*d)->fp);
	free((*d)->buf);
	free(*d);
	*d = NULL;
}

int text_docs_rewind(text_docs *d)
{
	if (!d)
		return -1;
	if (d->map != MAP_FAILED) {
		d->map_off = 0;
		d->pos = 0;
		d->at_line_start = 1;
		return 0;
	}
	if (!d->fp || fseek(d->fp, 0, SEEK_SET) != 0)
		return -1;
	d->pos = 0;
	d->at_line_start = 1;
	return 0;
}

static int line_is_blank(const char *s, size_t l)
{
	size_t i;

	if (l == 0 || s[l - 1] != '\n')
		return 0;
	for (i = 0; i < l; i++) {
		if (s[i] != ' ' && s[i] != '\t' && s[i] != '\r' && s[i] != '\n')
			return 0;
	}
	return 1;
}

int text_docs_next(text_docs *d, const char **doc, size_t *len, long *start_off)
{
	char line[LINE_BUF];
	size_t l;
	size_t n;
	long line_start;
	long start;
	int in_doc;
	int at_start;
	char *nb;

	if (!d || !doc || !len)
		return -1;
	/* E14 mmap: scan blank-line documents from mapped bytes. */
	if (d->map != MAP_FAILED) {
		const char *base = d->map;
		size_t off = d->map_off;
		size_t begin;

		while (off < d->map_len) {
			while (off < d->map_len
			    && (base[off] == '\n' || base[off] == '\r'))
				off++;
			if (off >= d->map_len)
				break;
			begin = off;
			while (off < d->map_len) {
				if (base[off] == '\n'
				    && off + 1 < d->map_len
				    && (base[off + 1] == '\n'
				    || (base[off + 1] == '\r'
				    && off + 2 < d->map_len
				    && base[off + 2] == '\n')))
					break;
				off++;
				if ((long)(off - begin) > DOC_MAX_BYTES)
					break;
			}
			n = off - begin;
			if (n > 0) {
				if (n + 1 > d->cap) {
					nb = realloc(d->buf, n + 1);
					if (!nb)
						return -1;
					d->buf = nb;
					d->cap = n + 1;
				}
				memcpy(d->buf, base + begin, n);
				d->buf[n] = '\0';
				while (off < d->map_len
				    && (base[off] == '\n' || base[off] == '\r'))
					off++;
				d->map_off = off;
				d->pos = (long)off;
				*doc = d->buf;
				*len = n;
				if (start_off)
					*start_off = (long)begin;
				return 1;
			}
			off++;
		}
		d->map_off = d->map_len;
		return 0;
	}
	n = 0;
	start = 0;
	in_doc = 0;
	while (d->fp && fgets(line, (int)sizeof(line), d->fp)) {
		l = strlen(line);
		line_start = d->pos;
		d->pos += (long)l;
		at_start = d->at_line_start;
		d->at_line_start = l > 0 && line[l - 1] == '\n';
		if (at_start && line_is_blank(line, l)) {
			if (in_doc)
				break;
			continue;
		}
		if (!in_doc) {
			in_doc = 1;
			start = line_start;
		}
		if (n + l + 1 > d->cap) {
			size_t nc = d->cap;

			while (n + l + 1 > nc)
				nc *= 2;
			nb = realloc(d->buf, nc);
			if (!nb)
				return -1;
			d->buf = nb;
			d->cap = nc;
		}
		memcpy(d->buf + n, line, l);
		n += l;
		if ((long)n >= DOC_MAX_BYTES)
			break;
	}
	if (!in_doc)
		return 0;
	while (n > 0 && (d->buf[n - 1] == '\n' || d->buf[n - 1] == '\r'
	    || d->buf[n - 1] == ' ' || d->buf[n - 1] == '\t'))
		n--;
	d->buf[n] = '\0';
	*doc = d->buf;
	*len = n;
	if (start_off)
		*start_off = start;
	return 1;
}

/* ------------------------------------------------------------------ */
/* In-memory split loader                                              */
/* ------------------------------------------------------------------ */

typedef struct	accum {
	int	*ids;
	long	n;
	long	cap;
	int	*starts;
	long	ns;
	long	scap;
} accum;

static int accum_add_doc(accum *a, const int *ids, int n, int window)
{
	long i;
	int *p;

	if (a->n + n > a->cap) {
		long nc = a->cap ? a->cap : 4096;

		while (nc < a->n + n)
			nc *= 2;
		p = realloc(a->ids, (size_t)nc * sizeof(int));
		if (!p)
			return -1;
		a->ids = p;
		a->cap = nc;
	}
	if (n >= window) {
		long cnt = n - window + 1;

		if (a->ns + cnt > a->scap) {
			long nc = a->scap ? a->scap : 4096;

			while (nc < a->ns + cnt)
				nc *= 2;
			p = realloc(a->starts, (size_t)nc * sizeof(int));
			if (!p)
				return -1;
			a->starts = p;
			a->scap = nc;
		}
		for (i = 0; i < cnt; i++)
			a->starts[a->ns + i] = (int)(a->n + i);
		a->ns += cnt;
	}
	memcpy(a->ids + a->n, ids, (size_t)n * sizeof(int));
	a->n += n;
	return 0;
}

static text_dataset *accum_finish(accum *a, int window)
{
	text_dataset *ds;

	if (a->ns <= 0 || a->n >= 0x7fffffffL) {
		free(a->ids);
		free(a->starts);
		return NULL;
	}
	ds = malloc(sizeof(text_dataset));
	if (!ds) {
		free(a->ids);
		free(a->starts);
		return NULL;
	}
	ds->ids = a->ids;
	ds->n_ids = (int)a->n;
	ds->window = window;
	ds->starts = a->starts;
	ds->n_starts = (int)a->ns;
	return ds;
}

int text_dataset_load_split(const char *path, int window, const bpe *tok,
	const text_split *sp, text_dataset **train, text_dataset **val)
{
	text_docs *d;
	accum tr;
	accum va;
	const char *doc;
	size_t len;
	long off;
	int *tmp;
	size_t tcap;
	int n;
	int rc;
	int is_val;

	if (!path || !train || !val || window < 2 || window > LM_MAX_CTX)
		return -1;
	*train = NULL;
	*val = NULL;
	d = text_docs_open(path);
	if (!d)
		return -1;
	memset(&tr, 0, sizeof(tr));
	memset(&va, 0, sizeof(va));
	tmp = NULL;
	tcap = 0;
	rc = -1;
	while ((n = text_docs_next(d, &doc, &len, &off)) == 1) {
		if (len == 0)
			continue;
		if (len + 1 > tcap) {
			int *p;

			tcap = len + 1;
			p = realloc(tmp, tcap * sizeof(int));
			if (!p)
				goto out;
			tmp = p;
		}
		if (tok)
			n = bpe_encode_n(tok, doc, len, tmp, (int)len);
		else
			n = tokenize_encode(doc, tmp, (int)len);
		if (n <= 0)
			continue;
		is_val = sp ? text_split_is_val(sp, doc, len, off) : 0;
		if (accum_add_doc(is_val ? &va : &tr, tmp, n, window) != 0)
			goto out;
	}
	if (n < 0)
		goto out;
	rc = 0;
out:
	free(tmp);
	text_docs_free(&d);
	if (rc != 0) {
		free(tr.ids);
		free(tr.starts);
		free(va.ids);
		free(va.starts);
		return -1;
	}
	*train = accum_finish(&tr, window);
	*val = accum_finish(&va, window);
	if (!*train) {
		text_dataset_free(val);
		return -1;
	}
	return 0;
}

bpe *text_split_train_bpe(const char *path, const text_split *sp,
	int vocab_size, long max_bytes)
{
	text_docs *d;
	bpe_trainer *t;
	bpe *b;
	const char *doc;
	size_t len;
	long off;
	long total;
	int n;

	d = text_docs_open(path);
	t = bpe_trainer_create();
	if (!d || !t) {
		text_docs_free(&d);
		bpe_trainer_free(&t);
		return NULL;
	}
	total = 0;
	while ((n = text_docs_next(d, &doc, &len, &off)) == 1) {
		if (sp && text_split_is_val(sp, doc, len, off))
			continue;
		/* Newline keeps the last word of a doc apart from the next. */
		if (bpe_trainer_add(t, doc, len) != 0
		    || bpe_trainer_add(t, "\n\n", 2) != 0)
			break;
		total += (long)len;
		if (max_bytes > 0 && total >= max_bytes)
			break;
	}
	b = n < 0 ? NULL : bpe_trainer_finish(t, vocab_size);
	bpe_trainer_free(&t);
	text_docs_free(&d);
	return b;
}

/* ------------------------------------------------------------------ */
/* Streaming / packing iterator                                        */
/* ------------------------------------------------------------------ */

struct text_stream {
	text_docs	*docs;
	const bpe	*tok;
	text_split	sp;
	int		side;
	int		window;
	int		stride;
	int		*buf;	/* rolling token ids */
	long		len;
	long		pos;
	long		cap;
	int		eof;
	int		*tmp;	/* per-document encode scratch */
	size_t		tcap;
	int		sb_cap;	/* shuffle buffer (windows) */
	int		sb_n;
	int		sb_done;
	int		*sb;
	unsigned int	seed;
	int		epoch;
	unsigned long long rng;
	long		count;
	long		last_count;
	int		varlen;	/* E8: emit padded short tail at EOF */
};

static unsigned long long rng_next(unsigned long long *s)
{
	unsigned long long x;

	x = *s;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	*s = x;
	return x;
}

static void stream_reseed(text_stream *s)
{
	s->rng = 0x9E3779B97F4A7C15ULL
	    ^ ((unsigned long long)s->seed * 2654435761ULL)
	    ^ ((unsigned long long)s->epoch * 0xD1B54A32D192ED03ULL);
	if (s->rng == 0)
		s->rng = 1;
	rng_next(&s->rng);
}

text_stream *text_stream_open(const char *path, int window, const bpe *tok,
	const text_split *sp, int side, int stride, int shuffle_buf,
	unsigned int seed)
{
	text_stream *s;

	if (!path || window < 2 || window > LM_MAX_CTX)
		return NULL;
	s = calloc(1, sizeof(text_stream));
	if (!s)
		return NULL;
	s->docs = text_docs_open(path);
	s->tok = tok;
	if (sp)
		s->sp = *sp;
	s->side = side;
	s->window = window;
	s->stride = (stride <= 0 || stride > window) ? window - 1 : stride;
	s->seed = seed;
	if (shuffle_buf > 0) {
		s->sb_cap = shuffle_buf;
		s->sb = malloc((size_t)shuffle_buf * (size_t)window * sizeof(int));
	}
	if (!s->docs || (shuffle_buf > 0 && !s->sb)) {
		text_stream_free(&s);
		return NULL;
	}
	stream_reseed(s);
	return s;
}

void text_stream_free(text_stream **s)
{
	if (!s || !*s)
		return;
	text_docs_free(&(*s)->docs);
	free((*s)->buf);
	free((*s)->tmp);
	free((*s)->sb);
	free(*s);
	*s = NULL;
}

/* Append the next selected document (+EOS) to the rolling buffer. */
static int stream_refill(text_stream *s)
{
	const char *doc;
	size_t len;
	long off;
	int r;
	int n;
	int *p;

	for (;;) {
		r = text_docs_next(s->docs, &doc, &len, &off);
		if (r < 0)
			return -1;
		if (r == 0) {
			s->eof = 1;
			return 0;
		}
		if (len == 0 || !side_wants(&s->sp, s->side, doc, len, off))
			continue;
		if (len + 2 > s->tcap) {
			s->tcap = len + 2;
			p = realloc(s->tmp, s->tcap * sizeof(int));
			if (!p)
				return -1;
			s->tmp = p;
		}
		if (s->tok)
			n = bpe_encode_n(s->tok, doc, len, s->tmp, (int)len);
		else
			n = tokenize_encode(doc, s->tmp, (int)len);
		if (n <= 0)
			continue;
		s->tmp[n++] = VOCAB_EOS;
		if (s->pos > 0) {	/* drop consumed ids */
			memmove(s->buf, s->buf + s->pos,
			    (size_t)(s->len - s->pos) * sizeof(int));
			s->len -= s->pos;
			s->pos = 0;
		}
		if (s->len + n > s->cap) {
			long nc = s->cap ? s->cap : 4096;

			while (nc < s->len + n)
				nc *= 2;
			p = realloc(s->buf, (size_t)nc * sizeof(int));
			if (!p)
				return -1;
			s->buf = p;
			s->cap = nc;
		}
		memcpy(s->buf + s->len, s->tmp, (size_t)n * sizeof(int));
		s->len += n;
		return 0;
	}
}

/* Writes a full window, or (varlen) a short padded tail. *n_out = real len. */
static int stream_raw_next(text_stream *s, int *out, int *n_out)
{
	long remain;
	int i;

	while (s->len - s->pos < s->window && !s->eof) {
		if (stream_refill(s) != 0)
			return -1;
	}
	remain = s->len - s->pos;
	if (remain >= s->window) {
		memcpy(out, s->buf + s->pos, (size_t)s->window * sizeof(int));
		s->pos += s->stride;
		*n_out = s->window;
		return 1;
	}
	/* E8: recover the short EOF tail instead of dropping it. */
	if (s->varlen && remain >= 2) {
		memcpy(out, s->buf + s->pos, (size_t)remain * sizeof(int));
		for (i = (int)remain; i < s->window; i++)
			out[i] = VOCAB_EOS;
		s->pos = s->len;
		*n_out = (int)remain;
		return 1;
	}
	return 0;
}

void text_stream_set_varlen(text_stream *s, int on)
{
	if (s)
		s->varlen = on ? 1 : 0;
}

int text_stream_next(text_stream *s, int *ids, int *n)
{
	int r;
	int j;
	int *slot;
	int nn;

	if (!s || !ids || !n)
		return -1;
	*n = s->window;
	if (s->sb_cap <= 0) {
		r = stream_raw_next(s, ids, &nn);
		if (r == 1) {
			*n = nn;
			s->count++;
		}
		return r;
	}
	while (s->sb_n < s->sb_cap && !s->sb_done) {
		r = stream_raw_next(s,
		    s->sb + (size_t)s->sb_n * (size_t)s->window, &nn);
		if (r < 0)
			return -1;
		if (r == 0)
			s->sb_done = 1;
		else if (nn == s->window)
			s->sb_n++;
		else {
			/* Short varlen tails skip the shuffle buffer. */
			memcpy(ids, s->sb + (size_t)s->sb_n * (size_t)s->window,
			    (size_t)s->window * sizeof(int));
			*n = nn;
			s->count++;
			return 1;
		}
	}
	if (s->sb_n == 0)
		return 0;
	j = (int)(rng_next(&s->rng) % (unsigned long long)s->sb_n);
	slot = s->sb + (size_t)j * (size_t)s->window;
	memcpy(ids, slot, (size_t)s->window * sizeof(int));
	*n = s->window;
	if (!s->sb_done) {
		r = stream_raw_next(s, slot, &nn);
		if (r < 0)
			return -1;
		if (r == 1 && nn == s->window) {
			s->count++;
			return 1;
		}
		if (r == 1 && nn < s->window) {
			/* Leave short tail for a later direct emit. */
			memcpy(s->sb + (size_t)s->sb_n * (size_t)s->window,
			    slot, (size_t)s->window * sizeof(int));
			/* restore slot from end after pop below via done */
		}
		s->sb_done = 1;
	}
	if (j != s->sb_n - 1)
		memcpy(slot, s->sb + (size_t)(s->sb_n - 1) * (size_t)s->window,
		    (size_t)s->window * sizeof(int));
	s->sb_n--;
	s->count++;
	return 1;
}

int text_stream_rewind(text_stream *s)
{
	if (!s || text_docs_rewind(s->docs) != 0)
		return -1;
	s->last_count = s->count;
	s->count = 0;
	s->len = 0;
	s->pos = 0;
	s->eof = 0;
	s->sb_n = 0;
	s->sb_done = 0;
	s->epoch++;
	stream_reseed(s);
	return 0;
}

long text_stream_count(const text_stream *s)
{
	return s ? s->count : 0;
}

long text_stream_last_count(const text_stream *s)
{
	return s ? s->last_count : 0;
}

long text_stream_est_count(const text_stream *s)
{
	long bytes;
	double tok_per_byte;
	double tokens;
	double frac;
	long n;

	if (!s)
		return 0;
	/* After one full epoch, last_count is exact for the same split. */
	if (s->last_count > 0)
		return s->last_count;
	bytes = s->sp.file_size;
	if (bytes <= 0)
		return 0;
	/* Byte ids ≈ 1 token/byte; BPE on English wiki is roughly ~0.25–0.35. */
	tok_per_byte = s->tok ? 0.30 : 1.0;
	tokens = (double)bytes * tok_per_byte;
	frac = 1.0;
	if (s->side == TEXT_SIDE_TRAIN && s->sp.val_frac > 0.0
	    && s->sp.val_frac < 1.0)
		frac = 1.0 - s->sp.val_frac;
	else if (s->side == TEXT_SIDE_VAL && s->sp.val_frac > 0.0
	    && s->sp.val_frac < 1.0)
		frac = s->sp.val_frac;
	tokens *= frac;
	if (s->stride <= 0)
		return 0;
	n = (long)(tokens / (double)s->stride);
	return n > 0 ? n : 1;
}

int text_stream_window(const text_stream *s)
{
	return s ? s->window : 0;
}

int text_stream_stride(const text_stream *s)
{
	return s ? s->stride : 0;
}
