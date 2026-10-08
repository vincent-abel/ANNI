/*
**	Optional GPU backend (CUDA via make cuda; stub otherwise)
*/
#ifndef GPU_H__
#define GPU_H__

#include "mat.h"
#include <stddef.h>

int	gpu_available(void);
const char	*gpu_backend_name(void);

/* Eager (default): D2H after each op. Batch: leave results on device when d_data set. */
#define GPU_SYNC_EAGER	0
#define GPU_SYNC_BATCH	1
void	gpu_set_sync_mode(int mode);
int	gpu_sync_mode(void);

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
int	gpu_mat_outer_add(mat *dw, const mat *a, const mat *b);
/* Row LayerNorm backward; 0 = done on device, -1 = use CPU. */
int	gpu_layernorm_backward(mat *dx, const mat *dy, const double *gamma,
		const double *x_hat, const double *inv_std, int T, int d,
		double *dgamma, double *dbeta);
/*
** Flat Adam step on device buffers (param/grad/m/v length n).
** Updates param/m/v in place; zeros grad. Returns 0 on success, -1 = CPU.
*/
int	gpu_adam_step(double *param, double *grad, double *m, double *v, int n,
		double lr, double beta1, double beta2, double eps, int t);
/*
** One decode-step MHA: QKV proj, append K/V at row len-1, attend, Wo.
** k_cache/v_cache are max_T x d_model (host). Returns 0 on success, -1 = CPU.
*/
int	gpu_kv_attn_decode(double *out, const double *x,
		const double *wq, const double *wk, const double *wv,
		const double *wo, double *k_cache, double *v_cache,
		int len, int max_T, int d_model, int n_heads, int d_head);

#endif
