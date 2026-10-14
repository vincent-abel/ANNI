/*
**	Optional GPU backend (CUDA via make cuda; stub otherwise)
*/
#ifndef GPU_H__
#define GPU_H__

#include "real.h"
#include "mat.h"
#include <stddef.h>

int	gpu_available(void);
const char	*gpu_backend_name(void);
/* Free / total device memory in bytes (cudaMemGetInfo). 0 on success, -1 = no GPU. */
int	gpu_mem_info(size_t *free_bytes, size_t *total_bytes);

/* Eager (default): D2H after each op. Batch: leave results on device when d_data set. */
#define GPU_SYNC_EAGER	0
#define GPU_SYNC_BATCH	1
void	gpu_set_sync_mode(int mode);
int	gpu_sync_mode(void);

/* S5: compute dtype for cuBLAS GemmEx (weights/Adam masters stay FP32). */
#define GPU_PREC_FP32	0
#define GPU_PREC_BF16	1
#define GPU_PREC_FP16	2
void	gpu_set_compute_prec(int prec);	/* FP32/BF16/FP16 */
int	gpu_compute_prec(void);
/*
** Block until all queued device work finishes. In GPU_SYNC_BATCH mode the
** individual GEMM/outer-add launches do not sync; call this once per
** microbatch (e.g. before reading device-authoritative grads). No-op on CPU.
*/
void	gpu_device_sync(void);

void	*gpu_malloc(size_t nbytes);
void	gpu_free(void *p);
int	gpu_memcpy_h2d(void *dst, const void *src, size_t nbytes);
int	gpu_memcpy_d2h(void *dst, const void *src, size_t nbytes);
int	gpu_memcpy_d2d(void *dst, const void *src, size_t nbytes);
int	gpu_memset(void *dst, int value, size_t nbytes);

/*
** Try GPU mat_mul; returns 0 on success, -1 to fall back to CPU.
** Uses mat.d_data when both inputs/outputs are on device.
*/
int	gpu_mat_mul(mat *c, const mat *a, const mat *b);
/* dA = dC * B^T, dB = A^T * dC (NULL skips that output). */
int	gpu_mat_mul_backward(mat *da, mat *db, const mat *dc,
		const mat *a, const mat *b);
int	gpu_mat_mul_bt(mat *c, const mat *a, const mat *b);
/* c = a^T b (a: T x m, b: T x n -> c: m x n); cuBLAS OP_T when built in. */
int	gpu_mat_mul_at(mat *c, const mat *a, const mat *b);
int	gpu_mat_outer_add(mat *dw, const mat *a, const mat *b);
/* Row LayerNorm backward; 0 = done on device, -1 = use CPU. */
int	gpu_layernorm_backward(mat *dx, const mat *dy, const ann_real *gamma,
		const ann_real *x_hat, const ann_real *inv_std, int T, int d,
		ann_real *dgamma, ann_real *dbeta);
/*
** Flat Adam step on host buffers (param/grad/m/v length n); uploads, runs,
** downloads. grad is scaled by grad_scale before the moment update.
** Updates param/m/v in place; zeros grad. Returns 0 on success, -1 = CPU.
*/
int	gpu_adam_step(ann_real *param, ann_real *grad, ann_real *m, ann_real *v, int n,
		double lr, double beta1, double beta2, double eps, int t,
		double grad_scale);
/*
** Same update on device pointers (no copies): d_param/d_grad/d_m/d_v are
** gpu_malloc'd length n. g = grad_scale * d_grad; d_grad is zeroed after.
** Returns 0 on success, -1 = CPU (stub or invalid args).
*/
int	gpu_adam_step_dev(ann_real *d_param, ann_real *d_grad, ann_real *d_m,
		ann_real *d_v, int n, double lr, double beta1, double beta2,
		double eps, int t, double grad_scale);
/*
** One decode-step MHA: QKV proj, append K/V at row len-1, attend, Wo.
** k_cache/v_cache are max_T x d_model (host). Returns 0 on success, -1 = CPU.
*/
int	gpu_kv_attn_decode(ann_real *out, const ann_real *x,
		const ann_real *wq, const ann_real *wk, const ann_real *wv,
		const ann_real *wo, ann_real *k_cache, ann_real *v_cache,
		int len, int max_T, int d_model, int n_heads, int d_head);

/*
** Track K: elementwise kernels. Every pointer below may be a host pointer
** (staged through persistent device scratch, uploaded/downloaded around the
** launch) or a device pointer from gpu_malloc / mat.d_data (used in place, no
** copies; synchronised only in GPU_SYNC_EAGER). All return 0 on success and
** -1 when the caller must use its CPU path (stub, no device, bad args).
*/

/* In-place row softmax over a row-major n_rows x n_cols block. */
int	gpu_softmax_rows(ann_real *rows, int n_rows, int n_cols);
/*
** Row softmax backward: dx = y * (dy - sum(y * dy)) per row. y is the
** softmax output. dx may alias dy (or y) for in-place use.
*/
int	gpu_softmax_bwd_rows(ann_real *dx, const ann_real *y,
		const ann_real *dy, int n_rows, int n_cols);
/*
** Row LayerNorm forward over T x d: x_hat = (x-mean)*inv_std,
** y = x_hat*gamma + beta. y may alias x. x_hat / inv_std may be NULL.
** Uses eps = 1e-5; gpu_layernorm_forward_ex takes eps and also stores the
** per-row mean (mean may be NULL).
*/
int	gpu_layernorm_forward(ann_real *y, const ann_real *x,
		const ann_real *gamma, const ann_real *beta,
		ann_real *x_hat, ann_real *inv_std, int T, int d);
int	gpu_layernorm_forward_ex(ann_real *y, const ann_real *x,
		const ann_real *gamma, const ann_real *beta,
		ann_real *x_hat, ann_real *mean, ann_real *inv_std,
		int T, int d, double eps);
/* x[r, c] += bias[c] for a row-major rows x cols block. */
int	gpu_bias_row_add(ann_real *x, const ann_real *bias, int rows, int cols);
/*
** S1: After row-softmax, subtract one-hot(tgt), accumulate column sums into
** db_host[V], and optionally write mean CE (-log p[tgt]). probs stays device
** when already on device (no full logits D2H). tgt is host ints, length rows.
*/
int	gpu_ce_delta_from_softmax(ann_real *probs, const int *tgt, int rows,
		int V, ann_real *db_host, double *mean_ce);
/*
** S2: Add sinusoidal posenc in place to packed (B*T) x D; position t restarts
** each sequence. x may be device-resident.
*/
int	gpu_posenc_add_packed(ann_real *x, int B, int T, int D);
/*
** S3: db_host[cols] += column sums of rows x cols matrix (device preferred).
*/
int	gpu_mat_col_sum_add(ann_real *db_host, const ann_real *x, int rows,
		int cols);
/* x[i] = max(x[i], 0) in place over n elements. */
int	gpu_relu_forward(ann_real *x, int n);
/*
** ReLU backward in place: dx[i] = 0 where ref[i] <= 0, else unchanged.
** Mask convention: ref is the pre-activation (z) or the post-ReLU output /
** a 0-1 mask -- all three agree because z > 0 <=> relu(z) > 0 <=> mask != 0.
*/
int	gpu_relu_backward(ann_real *dx, const ann_real *x_pre_relu_or_mask, int n);

/*
** M1: fused causal MHA over B packed sequences (B*T) x (H*dh), one launch per
** phase for every (b, h, i). w is the (B*H*T) x T softmax-weight buffer (must
** be pinned: device-resident, host copy never read). T <= 128. Return 0 on
** success, -1 = caller must use the per-head path (stub, no device, bad args).
** bwd consumes the w written by fwd; dq/dk/dv are overwritten.
*/
int	gpu_attn_causal_fwd(mat *concat, mat *w, const mat *q, const mat *k,
		const mat *v, int B, int T, int H, int dh);
/*
** S7: causal flash forward. Writes per-row logsumexp into lse ((B*H*T) x 1)
** and does not materialize T×T scores. lse must already be device-resident.
** Same causal softmax as the tiled backward (j > i masked, plus seqlens).
*/
int	gpu_attn_flash_fwd(mat *concat, const mat *q, const mat *k,
		const mat *v, int B, int T, int H, int dh, mat *lse);
/* FA-2 tiled bwd: P = exp(S - L) from this forward's lse. No T×T buffer. */
int	gpu_attn_tiled_bwd(mat *dq, mat *dk, mat *dv, const mat *dconcat,
		const mat *concat, const mat *q, const mat *k, const mat *v,
		const mat *lse, int B, int T, int H, int dh);
/* Recompute that causal logsumexp into lse. Does not write a score tensor. */
int	gpu_attn_recompute_w(mat *lse, const mat *q, const mat *k, int B, int T,
		int H, int dh);
/*
** Milliseconds of one tiled flash backward, or -1 if not measured yet.
** Timed once, outside CUDA graph capture. This is not a cuDNN SDPA launch.
*/
float	gpu_attn_flash_bwd_ms(void);
int	gpu_attn_causal_bwd(mat *dq, mat *dk, mat *dv, const mat *dconcat,
		const mat *w, const mat *q, const mat *k, const mat *v,
		int B, int T, int H, int dh);
/*
** TC path (40TF wave): pack heads + strided-batched GemmEx QK/AV + fused
** causal softmax. w is (B*H*T) x T (pinned). Prefer this over row-wise kernels.
*/
int	gpu_attn_tc_fwd(mat *concat, mat *w, const mat *q, const mat *k,
		const mat *v, int B, int T, int H, int dh);
int	gpu_attn_tc_bwd(mat *dq, mat *dk, mat *dv, const mat *dconcat,
		mat *w, const mat *q, const mat *k, const mat *v,
		int B, int T, int H, int dh);
/* One GEMM X@ [Wq|Wk|Wv] then split into q,k,v (D x D weights). */
int	gpu_attn_qkv_fused(mat *q, mat *k, mat *v, const mat *x,
		const mat *wq, const mat *wk, const mat *wv);
/* Pack CE gather: dst[b*(T-1)+t] = src[b*T+t] for t in [0,T-1). */
int	gpu_pack_seq_rows(ann_real *dst, const ann_real *src, int B, int T,
		int D);
int	gpu_unpack_seq_rows(ann_real *dst, const ann_real *src, int B, int T,
		int D);
/* S8: x += bias then ReLU in one launch (rows x cols). */
int	gpu_bias_relu_forward(ann_real *x, const ann_real *bias, int rows,
		int cols);
/* E8: optional host seqlens[B] in [1,T]; NULL clears. Active for next fused attn. */
void	gpu_attn_set_seqlens(const int *seqlens);

/*
** R1 residency helpers: prefer D2D when twins exist. Return 0 on success,
** -1 = CPU fallback. After success, dst is device-authoritative when pinned.
*/
int	gpu_mat_copy(mat *dst, const mat *src);
/* dst += src (elementwise). */
int	gpu_mat_add(mat *dst, const mat *src);
/* y += a * x (elementwise). */
int	gpu_mat_axpy(mat *y, double a, const mat *x);

/*
** R2: embed gather / grad scatter. table is vocab x D (pinned). ids length n
** on host (uploaded). out / dout are n x D mats.
*/
int	gpu_embed_lookup(mat *out, const mat *table, const int *ids, int n);
int	gpu_embed_scatter(mat *dtable, const mat *dout, const int *ids, int n);

/* E5: page-locked host alloc for staging; NULL on stub/failure. */
void	*gpu_host_alloc(size_t nbytes);
void	gpu_host_free(void *p);

/* R8/E6: secondary copy stream (NULL/0 = default). */
void	*gpu_copy_stream(void);
void	gpu_stream_sync_copy(void);

/*
** R12: capture/replay helpers for a fixed microbatch. begin/end wrap the
** default stream; replay returns -1 if no graph. Capture fails if host ops
** remain in the region (expected until full residency).
*/
int	gpu_graph_begin(void);
int	gpu_graph_end(void);
int	gpu_graph_replay(void);
void	gpu_graph_destroy(void);

#endif
