/*
**	Subword byte-pair-encoding (BPE): train, encode, decode, save/load
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe.h"

#define BPE_EMPTY	(~0ULL)
#define BPE_MAX_MERGES	(1 << 22)

static unsigned long long pair_key(int a, int b)
{
	return ((unsigned long long)(unsigned int)a << 32)
	    | (unsigned long long)(unsigned int)b;
}

static unsigned int key_hash(unsigned long long k)
{
	k *= 0x9E3779B97F4A7C15ULL;
	return (unsigned int)(k >> 32);
}

/* ------------------------------------------------------------------ */
/* Pre-tokenization: split text into "words" that merges never cross.  */
/* A word is an optional leading space plus a run of one byte class.   */
/* ------------------------------------------------------------------ */

static int byte_class(unsigned char c)
{
	if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
	    || (c >= '0' && c <= '9') || c >= 128)
		return 0;
	return 1;
}

static size_t word_len(const unsigned char *s, size_t n)
{
	size_t i;
	int cls;

	i = 0;
	if (s[0] == ' ')
		i = 1;
	if (i < n && s[i] != ' ') {
		cls = byte_class(s[i]);
		i++;
		while (i < n && i < BPE_MAX_WORD && s[i] != ' '
		    && byte_class(s[i]) == cls)
			i++;
	}
	if (i > BPE_MAX_WORD)
		i = BPE_MAX_WORD;
	return i;
}

/* ------------------------------------------------------------------ */
/* Tokenizer object                                                    */
/* ------------------------------------------------------------------ */

void bpe_free(bpe **b)
{
	int i;

	if (!b || !*b)
		return;
	if ((*b)->bytes) {
		for (i = 0; i < (*b)->vocab_size; i++)
			free((*b)->bytes[i]);
	}
	free((*b)->bytes);
	free((*b)->blen);
	free((*b)->left);
	free((*b)->right);
	free((*b)->hkey);
	free((*b)->hval);
	free(*b);
	*b = NULL;
}

static int rank_lookup(const bpe *b, int l, int r)
{
	unsigned long long k;
	unsigned int h;

	k = pair_key(l, r);
	h = key_hash(k) & (unsigned int)(b->hcap - 1);
	while (b->hkey[h] != BPE_EMPTY) {
		if (b->hkey[h] == k)
			return b->hval[h];
		h = (h + 1) & (unsigned int)(b->hcap - 1);
	}
	return -1;
}

bpe *bpe_create(const int *left, const int *right, int n_merges)
{
	bpe *b;
	int i;
	int cap;
	unsigned int h;
	unsigned long long k;

	if (n_merges < 0 || n_merges > BPE_MAX_MERGES)
		return NULL;
	if (n_merges > 0 && (!left || !right))
		return NULL;
	b = calloc(1, sizeof(bpe));
	if (!b)
		return NULL;
	b->n_merges = n_merges;
	b->vocab_size = BPE_BASE + n_merges;
	cap = 16;
	while (cap < 2 * n_merges + 16)
		cap <<= 1;
	b->hcap = cap;
	b->left = malloc((size_t)(n_merges ? n_merges : 1) * sizeof(int));
	b->right = malloc((size_t)(n_merges ? n_merges : 1) * sizeof(int));
	b->bytes = calloc((size_t)b->vocab_size, sizeof(unsigned char *));
	b->blen = calloc((size_t)b->vocab_size, sizeof(int));
	b->hkey = malloc((size_t)cap * sizeof(unsigned long long));
	b->hval = malloc((size_t)cap * sizeof(int));
	if (!b->left || !b->right || !b->bytes || !b->blen || !b->hkey
	    || !b->hval) {
		bpe_free(&b);
		return NULL;
	}
	for (i = 0; i < cap; i++)
		b->hkey[i] = BPE_EMPTY;
	for (i = 0; i < LM_BYTE_VOCAB; i++) {
		b->bytes[VOCAB_SPECIALS + i] = malloc(1);
		if (!b->bytes[VOCAB_SPECIALS + i]) {
			bpe_free(&b);
			return NULL;
		}
		b->bytes[VOCAB_SPECIALS + i][0] = (unsigned char)i;
		b->blen[VOCAB_SPECIALS + i] = 1;
	}
	for (i = 0; i < n_merges; i++) {
		int l = left[i];
		int r = right[i];
		int id = BPE_BASE + i;

		if (l < VOCAB_SPECIALS || l >= id || r < VOCAB_SPECIALS || r >= id) {
			bpe_free(&b);
			return NULL;
		}
		b->left[i] = l;
		b->right[i] = r;
		k = pair_key(l, r);
		h = key_hash(k) & (unsigned int)(cap - 1);
		while (b->hkey[h] != BPE_EMPTY) {
			if (b->hkey[h] == k) {	/* duplicate merge */
				bpe_free(&b);
				return NULL;
			}
			h = (h + 1) & (unsigned int)(cap - 1);
		}
		b->hkey[h] = k;
		b->hval[h] = i;
		b->blen[id] = b->blen[l] + b->blen[r];
		b->bytes[id] = malloc((size_t)b->blen[id]);
		if (!b->bytes[id]) {
			bpe_free(&b);
			return NULL;
		}
		memcpy(b->bytes[id], b->bytes[l], (size_t)b->blen[l]);
		memcpy(b->bytes[id] + b->blen[l], b->bytes[r], (size_t)b->blen[r]);
	}
	return b;
}

bpe *bpe_clone(const bpe *b)
{
	if (!b)
		return NULL;
	return bpe_create(b->left, b->right, b->n_merges);
}

int bpe_equal(const bpe *a, const bpe *b)
{
	if (!a || !b || a->n_merges != b->n_merges)
		return 0;
	return a->n_merges == 0
	    || (memcmp(a->left, b->left, (size_t)a->n_merges * sizeof(int)) == 0
	    && memcmp(a->right, b->right, (size_t)a->n_merges * sizeof(int)) == 0);
}

/* ------------------------------------------------------------------ */
/* Encode / decode                                                     */
/* ------------------------------------------------------------------ */

static int encode_word(const bpe *b, const unsigned char *w, size_t n, int *out)
{
	int s[BPE_MAX_WORD];
	int len;
	int j;
	int o;
	int r;
	int best;
	int a;
	int c;
	int nid;

	len = (int)n;
	for (j = 0; j < len; j++)
		s[j] = VOCAB_SPECIALS + w[j];
	while (len > 1 && b->n_merges > 0) {
		best = -1;
		for (j = 0; j + 1 < len; j++) {
			r = rank_lookup(b, s[j], s[j + 1]);
			if (r >= 0 && (best < 0 || r < best))
				best = r;
		}
		if (best < 0)
			break;
		a = b->left[best];
		c = b->right[best];
		nid = BPE_BASE + best;
		o = 0;
		for (j = 0; j < len; ) {
			if (j + 1 < len && s[j] == a && s[j + 1] == c) {
				s[o++] = nid;
				j += 2;
			} else {
				s[o++] = s[j++];
			}
		}
		len = o;
	}
	for (j = 0; j < len; j++)
		out[j] = s[j];
	return len;
}

int bpe_encode_n(const bpe *b, const char *text, size_t len, int *ids,
	int max_ids)
{
	const unsigned char *p;
	size_t pos;
	size_t wl;
	int n;
	int k;
	int tmp[BPE_MAX_WORD];

	if (!b || !text || !ids || max_ids <= 0)
		return -1;
	p = (const unsigned char *)text;
	n = 0;
	pos = 0;
	while (pos < len && n < max_ids) {
		wl = word_len(p + pos, len - pos);
		k = encode_word(b, p + pos, wl, tmp);
		if (k > max_ids - n)
			k = max_ids - n;
		memcpy(ids + n, tmp, (size_t)k * sizeof(int));
		n += k;
		pos += wl;
	}
	return n;
}

int bpe_encode(const bpe *b, const char *text, int *ids, int max_ids)
{
	if (!text)
		return -1;
	return bpe_encode_n(b, text, strlen(text), ids, max_ids);
}

int bpe_decode(const bpe *b, const int *ids, int n_ids, char *out, int out_cap)
{
	int i;
	int id;
	int w;
	int l;

	if (!b || !ids || !out || out_cap <= 0 || n_ids < 0)
		return -1;
	w = 0;
	for (i = 0; i < n_ids; i++) {
		id = ids[i];
		if (id >= 0 && id < VOCAB_SPECIALS)
			continue;
		if (id < 0 || id >= b->vocab_size) {
			if (w < out_cap - 1)
				out[w++] = '?';
			continue;
		}
		l = b->blen[id];
		if (l > out_cap - 1 - w)
			l = out_cap - 1 - w;
		if (l <= 0)
			break;
		memcpy(out + w, b->bytes[id], (size_t)l);
		w += l;
	}
	out[w] = '\0';
	return w;
}

/* ------------------------------------------------------------------ */
/* Save / load                                                         */
/* ------------------------------------------------------------------ */

int bpe_save(const bpe *b, const char *path)
{
	FILE *fp;
	int i;

	if (!b || !path)
		return -1;
	fp = fopen(path, "w");
	if (!fp)
		return -1;
	fprintf(fp, "BPEV 1 %d\n", b->n_merges);
	for (i = 0; i < b->n_merges; i++)
		fprintf(fp, "%d %d\n", b->left[i], b->right[i]);
	if (fclose(fp) != 0)
		return -1;
	return 0;
}

bpe *bpe_load(const char *path)
{
	FILE *fp;
	char tag[8];
	int ver;
	int n;
	int i;
	int *l;
	int *r;
	bpe *b;

	if (!path)
		return NULL;
	fp = fopen(path, "r");
	if (!fp)
		return NULL;
	if (fscanf(fp, "%7s %d %d", tag, &ver, &n) != 3 || strcmp(tag, "BPEV") != 0
	    || ver != 1 || n < 0 || n > BPE_MAX_MERGES) {
		fclose(fp);
		return NULL;
	}
	l = malloc((size_t)(n ? n : 1) * sizeof(int));
	r = malloc((size_t)(n ? n : 1) * sizeof(int));
	if (!l || !r) {
		free(l);
		free(r);
		fclose(fp);
		return NULL;
	}
	for (i = 0; i < n; i++) {
		if (fscanf(fp, "%d %d", &l[i], &r[i]) != 2) {
			free(l);
			free(r);
			fclose(fp);
			return NULL;
		}
	}
	fclose(fp);
	b = bpe_create(l, r, n);
	free(l);
	free(r);
	return b;
}

int bpe_write_fp(FILE *fp, const bpe *b)
{
	int i;
	int pair[2];

	if (!fp || !b)
		return -1;
	if (fwrite(&b->n_merges, sizeof(int), 1, fp) != 1)
		return -1;
	for (i = 0; i < b->n_merges; i++) {
		pair[0] = b->left[i];
		pair[1] = b->right[i];
		if (fwrite(pair, sizeof(int), 2, fp) != 2)
			return -1;
	}
	return 0;
}

bpe *bpe_read_fp(FILE *fp)
{
	int n;
	int i;
	int pair[2];
	int *l;
	int *r;
	bpe *b;

	if (!fp || fread(&n, sizeof(int), 1, fp) != 1 || n < 0
	    || n > BPE_MAX_MERGES)
		return NULL;
	l = malloc((size_t)(n ? n : 1) * sizeof(int));
	r = malloc((size_t)(n ? n : 1) * sizeof(int));
	if (!l || !r) {
		free(l);
		free(r);
		return NULL;
	}
	for (i = 0; i < n; i++) {
		if (fread(pair, sizeof(int), 2, fp) != 2) {
			free(l);
			free(r);
			return NULL;
		}
		l[i] = pair[0];
		r[i] = pair[1];
	}
	b = bpe_create(l, r, n);
	free(l);
	free(r);
	return b;
}

vocab *bpe_vocab_create(const bpe *b)
{
	vocab *v;
	int i;
	int j;
	int o;
	char buf[BPE_MAX_WORD * 6 + 8];
	unsigned char c;

	if (!b)
		return NULL;
	v = malloc(sizeof(vocab));
	if (!v)
		return NULL;
	v->size = b->vocab_size;
	v->id_to_token = calloc((size_t)v->size, sizeof(char *));
	if (!v->id_to_token) {
		free(v);
		return NULL;
	}
	for (i = 0; i < v->size; i++) {
		if (i == VOCAB_PAD) {
			strcpy(buf, "<pad>");
		} else if (i == VOCAB_BOS) {
			strcpy(buf, "<bos>");
		} else if (i == VOCAB_EOS) {
			strcpy(buf, "<eos>");
		} else if (i == VOCAB_UNK) {
			strcpy(buf, "<unk>");
		} else {
			o = 0;
			for (j = 0; j < b->blen[i] && o < (int)sizeof(buf) - 8; j++) {
				c = b->bytes[i][j];
				if (c >= 32 && c < 127)
					buf[o++] = (char)c;
				else
					o += sprintf(buf + o, "<b%d>", (int)c);
			}
			buf[o] = '\0';
		}
		v->id_to_token[i] = malloc(strlen(buf) + 1);
		if (!v->id_to_token[i]) {
			vocab_free(&v);
			return NULL;
		}
		strcpy(v->id_to_token[i], buf);
	}
	return v;
}

/* ------------------------------------------------------------------ */
/* Training                                                            */
/* ------------------------------------------------------------------ */

struct bpe_trainer {
	unsigned char	**w;
	int		*wlen;
	long		*wcnt;
	int		nw;
	int		cap_w;
	int		*htab;		/* word index or -1 */
	int		hcap;
};

static unsigned int bytes_hash(const unsigned char *p, size_t n)
{
	unsigned int h;
	size_t i;

	h = 2166136261u;
	for (i = 0; i < n; i++) {
		h ^= p[i];
		h *= 16777619u;
	}
	return h;
}

bpe_trainer *bpe_trainer_create(void)
{
	bpe_trainer *t;
	int i;

	t = calloc(1, sizeof(bpe_trainer));
	if (!t)
		return NULL;
	t->hcap = 1024;
	t->htab = malloc((size_t)t->hcap * sizeof(int));
	if (!t->htab) {
		free(t);
		return NULL;
	}
	for (i = 0; i < t->hcap; i++)
		t->htab[i] = -1;
	return t;
}

void bpe_trainer_free(bpe_trainer **t)
{
	int i;

	if (!t || !*t)
		return;
	for (i = 0; i < (*t)->nw; i++)
		free((*t)->w[i]);
	free((*t)->w);
	free((*t)->wlen);
	free((*t)->wcnt);
	free((*t)->htab);
	free(*t);
	*t = NULL;
}

static int trainer_grow_table(bpe_trainer *t)
{
	int ncap;
	int *nt;
	int i;
	unsigned int h;

	ncap = t->hcap * 2;
	nt = malloc((size_t)ncap * sizeof(int));
	if (!nt)
		return -1;
	for (i = 0; i < ncap; i++)
		nt[i] = -1;
	for (i = 0; i < t->nw; i++) {
		h = bytes_hash(t->w[i], (size_t)t->wlen[i]) & (unsigned int)(ncap - 1);
		while (nt[h] >= 0)
			h = (h + 1) & (unsigned int)(ncap - 1);
		nt[h] = i;
	}
	free(t->htab);
	t->htab = nt;
	t->hcap = ncap;
	return 0;
}

static int trainer_add_word(bpe_trainer *t, const unsigned char *p, size_t n)
{
	unsigned int h;
	int i;
	int nc;
	unsigned char **nw;
	int *nl;
	long *ncnt;

	h = bytes_hash(p, n) & (unsigned int)(t->hcap - 1);
	while ((i = t->htab[h]) >= 0) {
		if ((size_t)t->wlen[i] == n && memcmp(t->w[i], p, n) == 0) {
			t->wcnt[i]++;
			return 0;
		}
		h = (h + 1) & (unsigned int)(t->hcap - 1);
	}
	if (t->nw == t->cap_w) {
		nc = t->cap_w ? t->cap_w * 2 : 1024;
		nw = realloc(t->w, (size_t)nc * sizeof(unsigned char *));
		if (!nw)
			return -1;
		t->w = nw;
		nl = realloc(t->wlen, (size_t)nc * sizeof(int));
		if (!nl)
			return -1;
		t->wlen = nl;
		ncnt = realloc(t->wcnt, (size_t)nc * sizeof(long));
		if (!ncnt)
			return -1;
		t->wcnt = ncnt;
		t->cap_w = nc;
	}
	t->w[t->nw] = malloc(n);
	if (!t->w[t->nw])
		return -1;
	memcpy(t->w[t->nw], p, n);
	t->wlen[t->nw] = (int)n;
	t->wcnt[t->nw] = 1;
	t->htab[h] = t->nw;
	t->nw++;
	if (t->nw * 2 > t->hcap)
		return trainer_grow_table(t);
	return 0;
}

int bpe_trainer_add(bpe_trainer *t, const char *text, size_t len)
{
	const unsigned char *p;
	size_t pos;
	size_t wl;

	if (!t || (!text && len > 0))
		return -1;
	p = (const unsigned char *)text;
	pos = 0;
	while (pos < len) {
		wl = word_len(p + pos, len - pos);
		if (trainer_add_word(t, p + pos, wl) != 0)
			return -1;
		pos += wl;
	}
	return 0;
}

/* Pair statistics: counts per distinct (a,b) + lazy max-heap. */
typedef struct	ptab {
	int		*slot;		/* hash slots -> entry idx or -1 */
	int		hcap;
	int		np;
	int		cap;
	int		*pa;
	int		*pb;
	long		*pc;
	unsigned char	*dirty;
	int		*dlist;
	int		nd;
} ptab;

typedef struct	hent {
	long	c;
	int	idx;
} hent;

typedef struct	pheap {
	hent	*e;
	int	n;
	int	cap;
} pheap;

static int pt_init(ptab *pt)
{
	int i;

	memset(pt, 0, sizeof(*pt));
	pt->hcap = 1 << 12;
	pt->slot = malloc((size_t)pt->hcap * sizeof(int));
	if (!pt->slot)
		return -1;
	for (i = 0; i < pt->hcap; i++)
		pt->slot[i] = -1;
	return 0;
}

static void pt_free(ptab *pt)
{
	free(pt->slot);
	free(pt->pa);
	free(pt->pb);
	free(pt->pc);
	free(pt->dirty);
	free(pt->dlist);
}

static int pt_rehash(ptab *pt)
{
	int ncap;
	int *ns;
	int i;
	unsigned int h;

	ncap = pt->hcap * 2;
	ns = malloc((size_t)ncap * sizeof(int));
	if (!ns)
		return -1;
	for (i = 0; i < ncap; i++)
		ns[i] = -1;
	for (i = 0; i < pt->np; i++) {
		h = key_hash(pair_key(pt->pa[i], pt->pb[i])) & (unsigned int)(ncap - 1);
		while (ns[h] >= 0)
			h = (h + 1) & (unsigned int)(ncap - 1);
		ns[h] = i;
	}
	free(pt->slot);
	pt->slot = ns;
	pt->hcap = ncap;
	return 0;
}

static int pt_add(ptab *pt, int a, int b, long delta)
{
	unsigned int h;
	int i;
	int nc;
	int *t1;
	long *t2;
	unsigned char *t3;

	h = key_hash(pair_key(a, b)) & (unsigned int)(pt->hcap - 1);
	while ((i = pt->slot[h]) >= 0) {
		if (pt->pa[i] == a && pt->pb[i] == b)
			goto found;
		h = (h + 1) & (unsigned int)(pt->hcap - 1);
	}
	if (delta <= 0)
		return 0;
	if (pt->np == pt->cap) {
		nc = pt->cap ? pt->cap * 2 : 4096;
		t1 = realloc(pt->pa, (size_t)nc * sizeof(int));
		if (!t1)
			return -1;
		pt->pa = t1;
		t1 = realloc(pt->pb, (size_t)nc * sizeof(int));
		if (!t1)
			return -1;
		pt->pb = t1;
		t2 = realloc(pt->pc, (size_t)nc * sizeof(long));
		if (!t2)
			return -1;
		pt->pc = t2;
		t3 = realloc(pt->dirty, (size_t)nc);
		if (!t3)
			return -1;
		pt->dirty = t3;
		t1 = realloc(pt->dlist, (size_t)nc * sizeof(int));
		if (!t1)
			return -1;
		pt->dlist = t1;
		pt->cap = nc;
	}
	i = pt->np++;
	pt->pa[i] = a;
	pt->pb[i] = b;
	pt->pc[i] = 0;
	pt->dirty[i] = 0;
	pt->slot[h] = i;
	if (pt->np * 2 > pt->hcap && pt_rehash(pt) != 0)
		return -1;
found:
	pt->pc[i] += delta;
	if (delta > 0 && !pt->dirty[i]) {
		pt->dirty[i] = 1;
		pt->dlist[pt->nd++] = i;
	}
	return 0;
}

/* a before b: higher count first, ties by smaller (left,right). */
static int hent_before(const ptab *pt, const hent *x, const hent *y)
{
	if (x->c != y->c)
		return x->c > y->c;
	if (pt->pa[x->idx] != pt->pa[y->idx])
		return pt->pa[x->idx] < pt->pa[y->idx];
	return pt->pb[x->idx] < pt->pb[y->idx];
}

static int heap_push(pheap *h, const ptab *pt, long c, int idx)
{
	hent *ne;
	hent x;
	int i;
	int p;

	if (h->n == h->cap) {
		ne = realloc(h->e, (size_t)(h->cap ? h->cap * 2 : 4096) * sizeof(hent));
		if (!ne)
			return -1;
		h->e = ne;
		h->cap = h->cap ? h->cap * 2 : 4096;
	}
	x.c = c;
	x.idx = idx;
	i = h->n++;
	while (i > 0) {
		p = (i - 1) / 2;
		if (!hent_before(pt, &x, &h->e[p]))
			break;
		h->e[i] = h->e[p];
		i = p;
	}
	h->e[i] = x;
	return 0;
}

static hent heap_pop(pheap *h, const ptab *pt)
{
	hent top;
	hent x;
	int i;
	int c;

	top = h->e[0];
	x = h->e[--h->n];
	i = 0;
	while ((c = 2 * i + 1) < h->n) {
		if (c + 1 < h->n && hent_before(pt, &h->e[c + 1], &h->e[c]))
			c++;
		if (!hent_before(pt, &h->e[c], &x))
			break;
		h->e[i] = h->e[c];
		i = c;
	}
	if (h->n > 0)
		h->e[i] = x;
	return top;
}

static int flush_dirty(ptab *pt, pheap *h)
{
	int k;
	int i;

	for (k = 0; k < pt->nd; k++) {
		i = pt->dlist[k];
		pt->dirty[i] = 0;
		if (pt->pc[i] > 0 && heap_push(h, pt, pt->pc[i], i) != 0)
			return -1;
	}
	pt->nd = 0;
	return 0;
}

/* Replace every (a,b) in word s (length *plen) by nid; update pair counts. */
static int merge_word(ptab *pt, int *s, int *plen, long cnt, int a, int b,
	int nid)
{
	int L;
	int j;
	int o;
	int last_end;

	L = *plen;
	last_end = -1;
	for (j = 0; j + 1 < L; ) {
		if (s[j] == a && s[j + 1] == b) {
			if (pt_add(pt, a, b, -cnt) != 0)
				return -1;
			if (j > 0 && j != last_end
			    && pt_add(pt, s[j - 1], a, -cnt) != 0)
				return -1;
			if (j + 2 < L && pt_add(pt, b, s[j + 2], -cnt) != 0)
				return -1;
			last_end = j + 2;
			j += 2;
		} else {
			j++;
		}
	}
	o = 0;
	for (j = 0; j < L; ) {
		if (j + 1 < L && s[j] == a && s[j + 1] == b) {
			s[o++] = nid;
			j += 2;
		} else {
			s[o++] = s[j++];
		}
	}
	*plen = o;
	for (j = 0; j < o; j++) {
		if (s[j] != nid)
			continue;
		if (j > 0 && pt_add(pt, s[j - 1], nid, cnt) != 0)
			return -1;
		if (j + 1 < o && s[j + 1] != nid
		    && pt_add(pt, nid, s[j + 1], cnt) != 0)
			return -1;
	}
	return 0;
}

bpe *bpe_trainer_finish(bpe_trainer *t, int vocab_size)
{
	ptab pt;
	pheap heap;
	int **sym;
	int *len;
	int *active;
	int nact;
	int target;
	int m;
	int i;
	int j;
	int k;
	int a;
	int b;
	int found;
	int *left;
	int *right;
	hent top;
	bpe *out;

	if (!t || vocab_size < BPE_BASE)
		return NULL;
	target = vocab_size - BPE_BASE;
	out = NULL;
	sym = calloc((size_t)(t->nw ? t->nw : 1), sizeof(int *));
	len = malloc((size_t)(t->nw ? t->nw : 1) * sizeof(int));
	active = malloc((size_t)(t->nw ? t->nw : 1) * sizeof(int));
	left = malloc((size_t)(target ? target : 1) * sizeof(int));
	right = malloc((size_t)(target ? target : 1) * sizeof(int));
	memset(&heap, 0, sizeof(heap));
	if (!sym || !len || !active || !left || !right || pt_init(&pt) != 0) {
		free(sym);
		free(len);
		free(active);
		free(left);
		free(right);
		return NULL;
	}
	nact = 0;
	for (i = 0; i < t->nw; i++) {
		len[i] = t->wlen[i];
		sym[i] = malloc((size_t)len[i] * sizeof(int));
		if (!sym[i])
			goto done;
		for (j = 0; j < len[i]; j++)
			sym[i][j] = VOCAB_SPECIALS + t->w[i][j];
		for (j = 0; j + 1 < len[i]; j++) {
			if (pt_add(&pt, sym[i][j], sym[i][j + 1], t->wcnt[i]) != 0)
				goto done;
		}
		if (len[i] >= 2)
			active[nact++] = i;
	}
	if (flush_dirty(&pt, &heap) != 0)
		goto done;

	for (m = 0; m < target; m++) {
		found = 0;
		top.c = 0;
		top.idx = -1;
		while (heap.n > 0) {
			top = heap_pop(&heap, &pt);
			if (pt.pc[top.idx] <= 0)
				continue;
			if (pt.pc[top.idx] != top.c) {
				if (heap_push(&heap, &pt, pt.pc[top.idx], top.idx) != 0)
					goto done;
				continue;
			}
			found = 1;
			break;
		}
		if (!found)
			break;
		a = pt.pa[top.idx];
		b = pt.pb[top.idx];
		left[m] = a;
		right[m] = b;
		k = 0;
		for (i = 0; i < nact; i++) {
			int w = active[i];
			int contains = 0;

			for (j = 0; j + 1 < len[w]; j++) {
				if (sym[w][j] == a && sym[w][j + 1] == b) {
					contains = 1;
					break;
				}
			}
			if (contains && merge_word(&pt, sym[w], &len[w], t->wcnt[w],
			    a, b, BPE_BASE + m) != 0)
				goto done;
			if (len[w] >= 2)
				active[k++] = w;
		}
		nact = k;
		pt.pc[top.idx] = 0;
		if (flush_dirty(&pt, &heap) != 0)
			goto done;
	}
	out = bpe_create(left, right, m);
done:
	for (i = 0; i < t->nw; i++)
		free(sym[i]);
	free(sym);
	free(len);
	free(active);
	free(left);
	free(right);
	free(heap.e);
	pt_free(&pt);
	return out;
}

bpe *bpe_train_text(const char *text, size_t len, int vocab_size)
{
	bpe_trainer *t;
	bpe *b;

	t = bpe_trainer_create();
	if (!t)
		return NULL;
	if (bpe_trainer_add(t, text, len) != 0) {
		bpe_trainer_free(&t);
		return NULL;
	}
	b = bpe_trainer_finish(t, vocab_size);
	bpe_trainer_free(&t);
	return b;
}

#define TRAIN_CHUNK	(1 << 20)

bpe *bpe_train_from_file(const char *path, int vocab_size, long max_bytes)
{
	FILE *fp;
	bpe_trainer *t;
	bpe *b;
	char *buf;
	size_t have;
	size_t got;
	size_t cut;
	size_t want;
	long total;

	if (!path)
		return NULL;
	fp = fopen(path, "rb");
	if (!fp)
		return NULL;
	t = bpe_trainer_create();
	buf = malloc(2 * (size_t)TRAIN_CHUNK + 1);
	if (!t || !buf) {
		free(buf);
		bpe_trainer_free(&t);
		fclose(fp);
		return NULL;
	}
	have = 0;
	total = 0;
	for (;;) {
		want = (size_t)TRAIN_CHUNK;
		if (max_bytes > 0 && (long)want > max_bytes - total)
			want = (size_t)(max_bytes - total);
		got = want ? fread(buf + have, 1, want, fp) : 0;
		total += (long)got;
		have += got;
		if (got == 0) {	/* EOF or byte budget spent: flush the tail */
			if (have > 0 && bpe_trainer_add(t, buf, have) != 0)
				goto fail;
			break;
		}
		/* Cut before the last space (or after the last newline). */
		for (cut = have; cut > 0; cut--) {
			if (buf[cut - 1] == '\n')
				break;
			if (buf[cut - 1] == ' ') {
				cut--;
				break;
			}
		}
		if (cut == 0)
			cut = have;
		if (bpe_trainer_add(t, buf, cut) != 0)
			goto fail;
		memmove(buf, buf + cut, have - cut);
		have -= cut;
	}
	b = bpe_trainer_finish(t, vocab_size);
	bpe_trainer_free(&t);
	free(buf);
	fclose(fp);
	return b;
fail:
	bpe_trainer_free(&t);
	free(buf);
	fclose(fp);
	return NULL;
}
