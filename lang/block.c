/*
**	Transformer decoder block (pre-norm causal MHA + FFN) + backward
*/
#include <stdlib.h>
#include <string.h>
#include "block.h"
#include "gradutil.h"
#include "resid.h"

static int g_act_checkpoint;
/* When set, drop/recompute every stride-th block (2 = keep ~half VRAM, half recompute). */
static int g_act_ckpt_stride = 2;

void block_set_act_checkpoint(int on)
{
	g_act_checkpoint = on ? 1 : 0;
}

int block_act_checkpoint_enabled(void)
{
	return g_act_checkpoint;
}

static int block_ckpt_layer(int i)
{
	if (!g_act_checkpoint)
		return 0;
	if (g_act_ckpt_stride <= 1)
		return 1;
	return (i % g_act_ckpt_stride) == (g_act_ckpt_stride - 1);
}

/* S11: free device twins of recomputable activations; keep x_in. */
static void block_cache_drop_device(block_cache *bc)
{
	if (!bc)
		return;
	mat_device_free(bc->ln1_out);
	mat_device_free(bc->attn_out);
	mat_device_free(bc->x_mid);
	mat_device_free(bc->ln2_out);
	mat_device_free(bc->ffn_out);
	if (bc->ffnc) {
		if (!bc->ffnc->x_in_alias)
			mat_device_free(bc->ffnc->x_in);
		if (bc->ffnc->pre_act)
			mat_device_free(bc->ffnc->pre_act);
		mat_device_free(bc->ffnc->hidden);
	}
	if (bc->attnpc) {
		if (!bc->attnpc->x_in_alias)
			mat_device_free(bc->attnpc->x_in);
		mat_device_free(bc->attnpc->q);
		mat_device_free(bc->attnpc->k);
		mat_device_free(bc->attnpc->v);
		mat_device_free(bc->attnpc->concat);
		/*
		** Keep w_all and lse. lse is the forward causal logsumexp;
		** the tiled backward reads that same softmax. Do not drop it.
		*/
	}
	if (bc->attnc) {
		mat_device_free(bc->attnc->x_in);
		mat_device_free(bc->attnc->q);
		mat_device_free(bc->attnc->k);
		mat_device_free(bc->attnc->v);
		mat_device_free(bc->attnc->concat);
	}
}

/*
** B4: no residual D2D copies (16 MB each at 4096 x 1024, 38 per microbatch).
** Forward: each block writes its output straight into the next block's
** cached x_in. Backward: the residual gradient is accumulated in place in
** one buffer ("old + new", the same FP32 adds as copy-then-add) and handed
** to the embedding scatter without a final copy. ANN_RESID_INPLACE=0
** restores the copies.
*/
static int resid_inplace_on(void)
{
	static int on = -1;

	if (on < 0) {
		const char *e = getenv("ANN_RESID_INPLACE");

		on = (e && e[0] == '0') ? 0 : 1;
	}
	return on;
}

static void copy_mat(mat *dst, const mat *src)
{
	if (resid_copy(dst, src) == 0)
		return;
	if (mat_ensure_host((mat *)src) != 0 || mat_ensure_host(dst) != 0)
		return;
	memcpy(dst->data, src->data,
	    (size_t)dst->rows * (size_t)dst->cols * sizeof(ann_real));
	mat_host_dirty(dst);
}

tblock *tblock_create(void)
{
	tblock *b;

	b = malloc(sizeof(tblock));
	if (!b)
		return NULL;
	b->ln1 = layernorm_create();
	b->attn = attn_create();
	b->ln2 = layernorm_create();
	b->ffn = ffn_create();
	if (!b->ln1 || !b->attn || !b->ln2 || !b->ffn) {
		tblock_free(&b);
		return NULL;
	}
	return b;
}

void tblock_free(tblock **b)
{
	if (!b || !*b)
		return;
	layernorm_free(&(*b)->ln1);
	attn_free(&(*b)->attn);
	layernorm_free(&(*b)->ln2);
	ffn_free(&(*b)->ffn);
	free(*b);
	*b = NULL;
}

void tblock_zero_grad(tblock *b)
{
	if (!b)
		return;
	layernorm_zero_grad(b->ln1);
	attn_zero_grad(b->attn);
	layernorm_zero_grad(b->ln2);
	ffn_zero_grad(b->ffn);
}

static block_cache *block_cache_alloc(int B, int T, int n_heads, int d_head)
{
	block_cache *c;
	int n;

	if (B <= 0 || T <= 0)
		return NULL;
	n = B * T;
	c = malloc(sizeof(block_cache));
	if (!c)
		return NULL;
	memset(c, 0, sizeof(*c));
	c->T = n;
	/* Device-only when CUDA is up — host mirrors of all layers blew WSL RAM. */
	c->x_in = mat_create_dev(n, LM_D_MODEL);
	c->ln1_out = mat_create_dev(n, LM_D_MODEL);
	/*
	** attn_out is read only by the B == 1 path (the packed attention folds
	** the residual into the Wo GEMM) and ffn_out by nothing at all; both
	** were 16 MiB per layer of never-touched VRAM (ANN_VRAM_LAZY=0 keeps them).
	*/
	if (B == 1 || !attn_vram_lazy())
		c->attn_out = mat_create_dev(n, LM_D_MODEL);
	c->x_mid = mat_create_dev(n, LM_D_MODEL);
	c->ln2_out = mat_create_dev(n, LM_D_MODEL);
	if (!attn_vram_lazy())
		c->ffn_out = mat_create_dev(n, LM_D_MODEL);
	c->ln1c = ln_cache_create(n);
	if (B == 1)
		c->attnc = attn_cache_create(T, n_heads, d_head);
	else
		c->attnpc = attn_pcache_create(B, T, n_heads, d_head);
	c->ln2c = ln_cache_create(n);
	c->ffnc = ffn_cache_create(n);
	if (!c->x_in || !c->ln1_out || !c->x_mid
	    || (!c->attn_out && (B == 1 || !attn_vram_lazy()))
	    || !c->ln2_out || (!c->ffn_out && !attn_vram_lazy()) || !c->ln1c
	    || (!c->attnc && !c->attnpc)
	    || !c->ln2c || !c->ffnc) {
		block_cache_free(&c);
		return NULL;
	}
	/* R1: pin activation caches so BATCH GEMM leaves results on device. */
	mat_try_pin(c->x_in);
	mat_try_pin(c->ln1_out);
	mat_try_pin(c->attn_out);
	mat_try_pin(c->x_mid);
	mat_try_pin(c->ln2_out);
	mat_try_pin(c->ffn_out);
	return c;
}

block_cache *block_cache_create(int T, const attn *a)
{
	if (T <= 0 || !a)
		return NULL;
	return block_cache_alloc(1, T, a->n_heads, a->d_head);
}

block_cache *block_cache_create_packed(int B, int T, int n_heads, int d_head)
{
	return block_cache_alloc(B, T, n_heads, d_head);
}

void block_cache_free(block_cache **c)
{
	if (!c || !*c)
		return;
	mat_free(&(*c)->x_in);
	mat_free(&(*c)->ln1_out);
	mat_free(&(*c)->attn_out);
	mat_free(&(*c)->x_mid);
	mat_free(&(*c)->ln2_out);
	mat_free(&(*c)->ffn_out);
	ln_cache_free(&(*c)->ln1c);
	attn_cache_free(&(*c)->attnc);
	attn_pcache_free(&(*c)->attnpc);
	ln_cache_free(&(*c)->ln2c);
	ffn_cache_free(&(*c)->ffnc);
	free(*c);
	*c = NULL;
}

static void block_cache_repin(block_cache *bc)
{
	if (!bc)
		return;
	mat_try_pin(bc->x_in);
	mat_try_pin(bc->ln1_out);
	mat_try_pin(bc->attn_out);
	mat_try_pin(bc->x_mid);
	mat_try_pin(bc->ln2_out);
	mat_try_pin(bc->ffn_out);
	if (bc->ffnc) {
		mat_try_pin(bc->ffnc->x_in);
		if (bc->ffnc->pre_act)
			mat_try_pin(bc->ffnc->pre_act);
		mat_try_pin(bc->ffnc->hidden);
	}
	if (bc->attnpc) {
		mat_try_pin(bc->attnpc->x_in);
		mat_try_pin(bc->attnpc->q);
		if (!attn_vram_lazy()) {
			mat_try_pin(bc->attnpc->k);
			mat_try_pin(bc->attnpc->v);
		}
		mat_try_pin(bc->attnpc->concat);
	}
	if (bc->ln1c)
		mat_try_pin(bc->ln1c->x_hat);
	if (bc->ln2c)
		mat_try_pin(bc->ln2c->x_hat);
}

/* The block on c->x_in (already filled), output into out (not c->x_in). */
static int tblock_forward_body(mat *out, tblock *b, block_cache *c)
{
	/* LN(in→out) and resid sum skip a full D2D copy each. */
	if (layernorm_forward_into(c->ln1_out, c->x_in, b->ln1, c->ln1c) != 0)
		return -1;
	if (c->attnpc) {
		/* x_mid = x_in + Attn(ln1) Wo: residual fused into the GEMM. */
		if (attn_mha_forward_packed_res(c->x_mid, b->attn, c->ln1_out,
		    c->attnpc, c->x_in) != 0)
			return -1;
	} else {
		if (attn_mha_forward_cache(c->attn_out, b->attn, c->ln1_out, 1,
		    c->attnc) != 0)
			return -1;
		if (resid_sum(c->x_mid, c->x_in, c->attn_out) != 0)
			return -1;
	}
	if (layernorm_forward_into(c->ln2_out, c->x_mid, b->ln2, c->ln2c) != 0)
		return -1;
	/* x = x_mid + FFN(ln2): residual fused into the W2 GEMM. */
	if (ffn_forward_cache_res(out, c->ln2_out, b->ffn, c->ffnc, c->x_mid)
	    != 0)
		return -1;
	return 0;
}

int tblock_forward_cache(mat *x, tblock *b, block_cache *c)
{
	if (!x || !b || !c)
		return -1;
	/* After act-checkpoint drop_device, reattach twins before GEMMs. */
	block_cache_repin(c);
	copy_mat(c->x_in, x);
	return tblock_forward_body(x, b, c);
}

int tblock_forward(mat *x, tblock *b)
{
	block_cache *c;
	int rc;

	if (!x || !b)
		return -1;
	c = block_cache_create(x->rows, b->attn);
	if (!c)
		return -1;
	rc = tblock_forward_cache(x, b, c);
	block_cache_free(&c);
	return rc;
}

/*
** acc = 0: dx = d(block input) from dout, through the d_mid scratch.
** acc = 1 (B4): dx is the accumulator: it receives dout (no copy when
** dx == dout, the buffer then being overwritten in place) and both residual
** adds land in it; dout must not be read by the caller afterwards.
*/
static int tblock_backward_impl(mat *dx, const mat *dout, tblock *b,
	const block_cache *c, int acc)
{
	/* One shared pool for all blocks (backward is serial). */
	static mat *s_d_ffn_in;
	static mat *s_d_mid;
	static mat *s_d_attn_out;
	static mat *s_tmp;
	static int s_T;
	mat *d_ffn_in;
	mat *d_mid;
	mat *d_attn_out;
	mat *tmp;

	if (!dx || !dout || !b || !c)
		return -1;

	if (s_T != c->T || !s_d_ffn_in || !s_d_mid || !s_d_attn_out || !s_tmp) {
		mat_free(&s_d_ffn_in);
		mat_free(&s_d_mid);
		mat_free(&s_d_attn_out);
		mat_free(&s_tmp);
		s_T = 0;
		s_d_ffn_in = mat_create_dev(c->T, LM_D_MODEL);
		s_d_mid = mat_create_dev(c->T, LM_D_MODEL);
		s_d_attn_out = mat_create_dev(c->T, LM_D_MODEL);
		s_tmp = mat_create_dev(c->T, LM_D_MODEL);
		if (!s_d_ffn_in || !s_d_mid || !s_d_attn_out || !s_tmp) {
			mat_free(&s_d_ffn_in);
			mat_free(&s_d_mid);
			mat_free(&s_d_attn_out);
			mat_free(&s_tmp);
			return -1;
		}
		mat_try_pin(s_d_ffn_in);
		mat_try_pin(s_d_mid);
		mat_try_pin(s_d_attn_out);
		mat_try_pin(s_tmp);
		s_T = c->T;
	}
	d_ffn_in = s_d_ffn_in;
	d_mid = s_d_mid;
	d_attn_out = s_d_attn_out;
	tmp = s_tmp;
	mat_try_pin(dx);

	/* y = x_mid + ffn_out; dy splits to both */
	if (acc) {
		d_mid = dx;
		if (dx != dout)
			copy_mat(dx, dout);
	} else
		copy_mat(d_mid, dout);
	/* In place, ffn_backward reads dout before the LN2 add rewrites it. */
	if (ffn_backward(d_ffn_in, dout, b->ffn, c->ffnc) != 0)
		return -1;
	if (layernorm_backward_add(d_mid, d_ffn_in, b->ln2, c->ln2c, tmp) != 0)
		return -1;

	/* x_mid = x_in + attn_out (in place: the attention reads d_mid first) */
	if (!acc)
		copy_mat(dx, d_mid);
	if (c->attnpc) {
		if (attn_mha_backward_packed(d_attn_out, d_mid, b->attn,
		    c->attnpc) != 0)
			return -1;
	} else if (attn_mha_backward(d_attn_out, d_mid, b->attn, c->attnc) != 0)
		return -1;
	if (layernorm_backward_add(dx, d_attn_out, b->ln1, c->ln1c, tmp) != 0)
		return -1;
	return 0;
}

int tblock_backward(mat *dx, const mat *dout, tblock *b, const block_cache *c)
{
	if (dx == dout)
		return -1;
	return tblock_backward_impl(dx, dout, b, c, resid_inplace_on());
}

transformer *transformer_create(void)
{
	transformer *t;
	int i;

	t = malloc(sizeof(transformer));
	if (!t)
		return NULL;
	t->n_blocks = LM_N_BLOCKS;
	t->blocks = calloc((size_t)LM_N_BLOCKS, sizeof(tblock));
	if (!t->blocks) {
		free(t);
		return NULL;
	}
	for (i = 0; i < LM_N_BLOCKS; i++) {
		t->blocks[i].ln1 = layernorm_create();
		t->blocks[i].attn = attn_create();
		t->blocks[i].ln2 = layernorm_create();
		t->blocks[i].ffn = ffn_create();
		if (!t->blocks[i].ln1 || !t->blocks[i].attn
		    || !t->blocks[i].ln2 || !t->blocks[i].ffn) {
			transformer_free(&t);
			return NULL;
		}
	}
	return t;
}

void transformer_free(transformer **t)
{
	int i;

	if (!t || !*t)
		return;
	if ((*t)->blocks) {
		for (i = 0; i < (*t)->n_blocks; i++) {
			layernorm_free(&(*t)->blocks[i].ln1);
			attn_free(&(*t)->blocks[i].attn);
			layernorm_free(&(*t)->blocks[i].ln2);
			ffn_free(&(*t)->blocks[i].ffn);
		}
		free((*t)->blocks);
	}
	free(*t);
	*t = NULL;
}

void transformer_zero_grad(transformer *t)
{
	int i;

	if (!t || !t->blocks)
		return;
	for (i = 0; i < t->n_blocks; i++)
		tblock_zero_grad(&t->blocks[i]);
}

transformer_cache *transformer_cache_create_packed_dims(int B, int T,
	int n_blocks, int n_heads, int d_head)
{
	transformer_cache *c;
	int i;

	if (B <= 0 || T <= 0 || n_blocks <= 0)
		return NULL;
	c = malloc(sizeof(transformer_cache));
	if (!c)
		return NULL;
	c->T = B * T;
	c->B = B;
	c->Tseq = T;
	c->n_blocks = n_blocks;
	c->bc = calloc((size_t)n_blocks, sizeof(block_cache));
	if (!c->bc) {
		free(c);
		return NULL;
	}
	for (i = 0; i < n_blocks; i++) {
		block_cache *bc;

		bc = block_cache_create_packed(B, T, n_heads, d_head);
		if (!bc) {
			transformer_cache_free(&c);
			return NULL;
		}
		c->bc[i] = *bc;
		free(bc);
	}
	return c;
}

transformer_cache *transformer_cache_create(int T, const transformer *t)
{
	if (!t || t->n_blocks <= 0)
		return NULL;
	return transformer_cache_create_packed_dims(1, T, t->n_blocks,
	    t->blocks[0].attn->n_heads, t->blocks[0].attn->d_head);
}

transformer_cache *transformer_cache_create_packed(int B, int T,
	const transformer *t)
{
	if (!t || t->n_blocks <= 0)
		return NULL;
	return transformer_cache_create_packed_dims(B, T, t->n_blocks,
	    t->blocks[0].attn->n_heads, t->blocks[0].attn->d_head);
}

void transformer_cache_free(transformer_cache **c)
{
	int i;
	block_cache *bc;

	if (!c || !*c)
		return;
	if ((*c)->bc) {
		for (i = 0; i < (*c)->n_blocks; i++) {
			bc = &(*c)->bc[i];
			mat_free(&bc->x_in);
			mat_free(&bc->ln1_out);
			mat_free(&bc->attn_out);
			mat_free(&bc->x_mid);
			mat_free(&bc->ln2_out);
			mat_free(&bc->ffn_out);
			ln_cache_free(&bc->ln1c);
			attn_cache_free(&bc->attnc);
			attn_pcache_free(&bc->attnpc);
			ln_cache_free(&bc->ln2c);
			ffn_cache_free(&bc->ffnc);
		}
		free((*c)->bc);
	}
	free(*c);
	*c = NULL;
}

int transformer_forward_cache(mat *x, transformer *t, transformer_cache *c)
{
	int i;

	if (!x || !t || !c)
		return -1;
	if (resid_inplace_on() && t->n_blocks > 0) {
		block_cache_repin(&c->bc[0]);
		copy_mat(c->bc[0].x_in, x);
		for (i = 0; i < t->n_blocks; i++) {
			mat *out = i + 1 < t->n_blocks ? c->bc[i + 1].x_in : x;

			if (i > 0)
				block_cache_repin(&c->bc[i]);
			mat_try_pin(out);
			if (tblock_forward_body(out, &t->blocks[i], &c->bc[i]) != 0)
				return -1;
			if (block_ckpt_layer(i))
				block_cache_drop_device(&c->bc[i]);
		}
		return 0;
	}
	for (i = 0; i < t->n_blocks; i++) {
		if (tblock_forward_cache(x, &t->blocks[i], &c->bc[i]) != 0)
			return -1;
		/* S11: drop selected layers; x_in stays for recompute. */
		if (block_ckpt_layer(i))
			block_cache_drop_device(&c->bc[i]);
	}
	return 0;
}

int transformer_forward(mat *x, transformer *t)
{
	transformer_cache *c;
	int rc;

	if (!x || !t)
		return -1;
	c = transformer_cache_create(x->rows, t);
	if (!c)
		return -1;
	rc = transformer_forward_cache(x, t, c);
	transformer_cache_free(&c);
	return rc;
}

/*
** *dx_out = d(transformer input), in a buffer owned here (valid until the
** next call). B4 path: one accumulator for every block, no final copy.
*/
int transformer_backward_ref(const mat **dx_out, const mat *dout,
	transformer *t, transformer_cache *c)
{
	static mat *s_cur;
	static mat *s_prev;
	static mat *s_scratch;
	static int s_T;
	static int s_have_scratch;
	mat *cur;
	mat *prev;
	mat *swap;
	mat *scratch;
	int i;
	int want_scratch;

	if (!dx_out || !dout || !t || !c)
		return -1;
	want_scratch = g_act_checkpoint ? 1 : 0;
	if (s_T != c->T || !s_cur || !s_prev
	    || want_scratch != s_have_scratch) {
		mat_free(&s_cur);
		mat_free(&s_prev);
		mat_free(&s_scratch);
		s_T = 0;
		s_have_scratch = 0;
		s_cur = mat_create_dev(c->T, LM_D_MODEL);
		s_prev = mat_create_dev(c->T, LM_D_MODEL);
		s_scratch = want_scratch
		    ? mat_create_dev(c->T, LM_D_MODEL) : NULL;
		if (!s_cur || !s_prev || (want_scratch && !s_scratch)) {
			mat_free(&s_cur);
			mat_free(&s_prev);
			mat_free(&s_scratch);
			return -1;
		}
		mat_try_pin(s_cur);
		mat_try_pin(s_prev);
		if (s_scratch)
			mat_try_pin(s_scratch);
		s_T = c->T;
		s_have_scratch = want_scratch;
	}
	cur = s_cur;
	prev = s_prev;
	scratch = s_scratch;
	if (resid_inplace_on()) {
		/* The first block copies dout into cur (its d_mid copy). */
		for (i = t->n_blocks - 1; i >= 0; i--) {
			if (block_ckpt_layer(i)) {
				copy_mat(scratch, c->bc[i].x_in);
				if (tblock_forward_cache(scratch, &t->blocks[i],
				    &c->bc[i]) != 0)
					return -1;
			}
			if (tblock_backward_impl(cur,
			    i == t->n_blocks - 1 ? dout : cur, &t->blocks[i],
			    &c->bc[i], 1) != 0)
				return -1;
			if (block_ckpt_layer(i))
				block_cache_drop_device(&c->bc[i]);
		}
		if (t->n_blocks <= 0)
			copy_mat(cur, dout);
		*dx_out = cur;
		return 0;
	}
	copy_mat(cur, dout);
	for (i = t->n_blocks - 1; i >= 0; i--) {
		if (block_ckpt_layer(i)) {
			/* Recompute dropped-layer activations from saved x_in. */
			copy_mat(scratch, c->bc[i].x_in);
			if (tblock_forward_cache(scratch, &t->blocks[i],
			    &c->bc[i]) != 0)
				return -1;
		}
		if (tblock_backward(prev, cur, &t->blocks[i], &c->bc[i]) != 0)
			return -1;
		if (block_ckpt_layer(i))
			block_cache_drop_device(&c->bc[i]);
		swap = cur;
		cur = prev;
		prev = swap;
	}
	*dx_out = cur;
	return 0;
}

int transformer_backward(mat *dx, const mat *dout, transformer *t,
	transformer_cache *c)
{
	const mat *g;

	if (!dx)
		return -1;
	if (transformer_backward_ref(&g, dout, t, c) != 0)
		return -1;
	copy_mat(dx, g);
	return 0;
}
