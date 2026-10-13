/*
**	CPU stub for optional GPU backend (default build)
*/
#include <stdlib.h>
#include "gpu.h"

static int g_sync_mode;

int gpu_available(void)
{
	return 0;
}

const char *gpu_backend_name(void)
{
	return "cpu";
}

int gpu_mem_info(size_t *free_bytes, size_t *total_bytes)
{
	if (free_bytes)
		*free_bytes = 0;
	if (total_bytes)
		*total_bytes = 0;
	return -1;
}

void gpu_set_sync_mode(int mode)
{
	g_sync_mode = mode;
}

int gpu_sync_mode(void)
{
	return g_sync_mode;
}

void gpu_device_sync(void)
{
}

void *gpu_malloc(size_t nbytes)
{
	(void)nbytes;
	return NULL;
}

void gpu_free(void *p)
{
	(void)p;
}

int gpu_memcpy_h2d(void *dst, const void *src, size_t nbytes)
{
	(void)dst;
	(void)src;
	(void)nbytes;
	return -1;
}

int gpu_memcpy_d2h(void *dst, const void *src, size_t nbytes)
{
	(void)dst;
	(void)src;
	(void)nbytes;
	return -1;
}

int gpu_memcpy_d2d(void *dst, const void *src, size_t nbytes)
{
	(void)dst;
	(void)src;
	(void)nbytes;
	return -1;
}

int gpu_memset(void *dst, int value, size_t nbytes)
{
	(void)dst;
	(void)value;
	(void)nbytes;
	return -1;
}

int gpu_mat_mul(mat *c, const mat *a, const mat *b)
{
	(void)c;
	(void)a;
	(void)b;
	return -1;
}

int gpu_mat_mul_backward(mat *da, mat *db, const mat *dc,
	const mat *a, const mat *b)
{
	(void)da;
	(void)db;
	(void)dc;
	(void)a;
	(void)b;
	return -1;
}

int gpu_mat_mul_bt(mat *c, const mat *a, const mat *b)
{
	(void)c;
	(void)a;
	(void)b;
	return -1;
}

int gpu_mat_mul_at(mat *c, const mat *a, const mat *b)
{
	(void)c;
	(void)a;
	(void)b;
	return -1;
}

int gpu_mat_outer_add(mat *dw, const mat *a, const mat *b)
{
	(void)dw;
	(void)a;
	(void)b;
	return -1;
}

int gpu_layernorm_backward(mat *dx, const mat *dy, const ann_real *gamma,
	const ann_real *x_hat, const ann_real *inv_std, int T, int d,
	ann_real *dgamma, ann_real *dbeta)
{
	(void)dx;
	(void)dy;
	(void)gamma;
	(void)x_hat;
	(void)inv_std;
	(void)T;
	(void)d;
	(void)dgamma;
	(void)dbeta;
	return -1;
}

int gpu_softmax_rows(ann_real *rows, int n_rows, int n_cols)
{
	(void)rows;
	(void)n_rows;
	(void)n_cols;
	return -1;
}

int gpu_softmax_bwd_rows(ann_real *dx, const ann_real *y, const ann_real *dy,
	int n_rows, int n_cols)
{
	(void)dx;
	(void)y;
	(void)dy;
	(void)n_rows;
	(void)n_cols;
	return -1;
}

int gpu_layernorm_forward(ann_real *y, const ann_real *x, const ann_real *gamma,
	const ann_real *beta, ann_real *x_hat, ann_real *inv_std, int T, int d)
{
	(void)y;
	(void)x;
	(void)gamma;
	(void)beta;
	(void)x_hat;
	(void)inv_std;
	(void)T;
	(void)d;
	return -1;
}

int gpu_layernorm_forward_ex(ann_real *y, const ann_real *x,
	const ann_real *gamma, const ann_real *beta, ann_real *x_hat,
	ann_real *mean, ann_real *inv_std, int T, int d, double eps)
{
	(void)y;
	(void)x;
	(void)gamma;
	(void)beta;
	(void)x_hat;
	(void)mean;
	(void)inv_std;
	(void)T;
	(void)d;
	(void)eps;
	return -1;
}

void gpu_set_compute_prec(int prec)
{
	(void)prec;
}

int gpu_compute_prec(void)
{
	return GPU_PREC_FP32;
}

int gpu_bias_row_add(ann_real *x, const ann_real *bias, int rows, int cols)
{
	(void)x;
	(void)bias;
	(void)rows;
	(void)cols;
	return -1;
}

int gpu_attn_flash_fwd(mat *concat, const mat *q, const mat *k, const mat *v,
	int B, int T, int H, int dh)
{
	(void)concat;
	(void)q;
	(void)k;
	(void)v;
	(void)B;
	(void)T;
	(void)H;
	(void)dh;
	return -1;
}

int gpu_attn_tiled_bwd(mat *dq, mat *dk, mat *dv, const mat *dconcat,
	const mat *concat, const mat *q, const mat *k, const mat *v, int B,
	int T, int H, int dh)
{
	(void)dq;
	(void)dk;
	(void)dv;
	(void)dconcat;
	(void)concat;
	(void)q;
	(void)k;
	(void)v;
	(void)B;
	(void)T;
	(void)H;
	(void)dh;
	return -1;
}

int gpu_attn_tc_fwd(mat *concat, mat *w, const mat *q, const mat *k,
	const mat *v, int B, int T, int H, int dh)
{
	(void)concat;
	(void)w;
	(void)q;
	(void)k;
	(void)v;
	(void)B;
	(void)T;
	(void)H;
	(void)dh;
	return -1;
}

int gpu_attn_tc_bwd(mat *dq, mat *dk, mat *dv, const mat *dconcat, mat *w,
	const mat *q, const mat *k, const mat *v, int B, int T, int H, int dh)
{
	(void)dq;
	(void)dk;
	(void)dv;
	(void)dconcat;
	(void)w;
	(void)q;
	(void)k;
	(void)v;
	(void)B;
	(void)T;
	(void)H;
	(void)dh;
	return -1;
}

int gpu_pack_seq_rows(ann_real *dst, const ann_real *src, int B, int T, int D)
{
	(void)dst;
	(void)src;
	(void)B;
	(void)T;
	(void)D;
	return -1;
}

int gpu_unpack_seq_rows(ann_real *dst, const ann_real *src, int B, int T,
	int D)
{
	(void)dst;
	(void)src;
	(void)B;
	(void)T;
	(void)D;
	return -1;
}

int gpu_attn_recompute_w(mat *w, const mat *q, const mat *k, int B, int T,
	int H, int dh)
{
	(void)w;
	(void)q;
	(void)k;
	(void)B;
	(void)T;
	(void)H;
	(void)dh;
	return -1;
}

int gpu_bias_relu_forward(ann_real *x, const ann_real *bias, int rows, int cols)
{
	(void)x;
	(void)bias;
	(void)rows;
	(void)cols;
	return -1;
}

int gpu_ce_delta_from_softmax(ann_real *probs, const int *tgt, int rows, int V,
	ann_real *db_host, double *mean_ce)
{
	(void)probs;
	(void)tgt;
	(void)rows;
	(void)V;
	(void)db_host;
	(void)mean_ce;
	return -1;
}

int gpu_posenc_add_packed(ann_real *x, int B, int T, int D)
{
	(void)x;
	(void)B;
	(void)T;
	(void)D;
	return -1;
}

int gpu_mat_col_sum_add(ann_real *db_host, const ann_real *x, int rows, int cols)
{
	(void)db_host;
	(void)x;
	(void)rows;
	(void)cols;
	return -1;
}

int gpu_relu_forward(ann_real *x, int n)
{
	(void)x;
	(void)n;
	return -1;
}

int gpu_relu_backward(ann_real *dx, const ann_real *x_pre_relu_or_mask, int n)
{
	(void)dx;
	(void)x_pre_relu_or_mask;
	(void)n;
	return -1;
}

int gpu_adam_step(ann_real *param, ann_real *grad, ann_real *m, ann_real *v, int n,
	double lr, double beta1, double beta2, double eps, int t, double grad_scale)
{
	(void)param;
	(void)grad;
	(void)m;
	(void)v;
	(void)n;
	(void)lr;
	(void)beta1;
	(void)beta2;
	(void)eps;
	(void)t;
	(void)grad_scale;
	return -1;
}

int gpu_adam_step_dev(ann_real *d_param, ann_real *d_grad, ann_real *d_m,
	ann_real *d_v, int n, double lr, double beta1, double beta2, double eps,
	int t, double grad_scale)
{
	(void)d_param;
	(void)d_grad;
	(void)d_m;
	(void)d_v;
	(void)n;
	(void)lr;
	(void)beta1;
	(void)beta2;
	(void)eps;
	(void)t;
	(void)grad_scale;
	return -1;
}

int gpu_kv_attn_decode(ann_real *out, const ann_real *x,
	const ann_real *wq, const ann_real *wk, const ann_real *wv,
	const ann_real *wo, ann_real *k_cache, ann_real *v_cache,
	int len, int max_T, int d_model, int n_heads, int d_head)
{
	(void)out;
	(void)x;
	(void)wq;
	(void)wk;
	(void)wv;
	(void)wo;
	(void)k_cache;
	(void)v_cache;
	(void)len;
	(void)max_T;
	(void)d_model;
	(void)n_heads;
	(void)d_head;
	return -1;
}

void gpu_attn_set_seqlens(const int *seqlens)
{
	(void)seqlens;
}

int gpu_attn_causal_fwd(mat *concat, mat *w, const mat *q, const mat *k,
	const mat *v, int B, int T, int H, int dh)
{
	(void)concat;
	(void)w;
	(void)q;
	(void)k;
	(void)v;
	(void)B;
	(void)T;
	(void)H;
	(void)dh;
	return -1;
}

int gpu_attn_causal_bwd(mat *dq, mat *dk, mat *dv, const mat *dconcat,
	const mat *w, const mat *q, const mat *k, const mat *v,
	int B, int T, int H, int dh)
{
	(void)dq;
	(void)dk;
	(void)dv;
	(void)dconcat;
	(void)w;
	(void)q;
	(void)k;
	(void)v;
	(void)B;
	(void)T;
	(void)H;
	(void)dh;
	return -1;
}

int gpu_mat_copy(mat *dst, const mat *src)
{
	(void)dst;
	(void)src;
	return -1;
}

int gpu_mat_add(mat *dst, const mat *src)
{
	(void)dst;
	(void)src;
	return -1;
}

int gpu_mat_axpy(mat *y, double a, const mat *x)
{
	(void)y;
	(void)a;
	(void)x;
	return -1;
}

int gpu_embed_lookup(mat *out, const mat *table, const int *ids, int n)
{
	(void)out;
	(void)table;
	(void)ids;
	(void)n;
	return -1;
}

int gpu_embed_scatter(mat *dtable, const mat *dout, const int *ids, int n)
{
	(void)dtable;
	(void)dout;
	(void)ids;
	(void)n;
	return -1;
}

void *gpu_host_alloc(size_t nbytes)
{
	(void)nbytes;
	return NULL;
}

void gpu_host_free(void *p)
{
	(void)p;
}

void *gpu_copy_stream(void)
{
	return NULL;
}

void gpu_stream_sync_copy(void)
{
}

int gpu_graph_begin(void)
{
	return -1;
}

int gpu_graph_end(void)
{
	return -1;
}

int gpu_graph_replay(void)
{
	return -1;
}

void gpu_graph_destroy(void)
{
}
