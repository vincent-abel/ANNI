/*
**	Language-model shared sizes (toy decoder-only).
*/
#ifndef LM_CONFIG_H__
#define LM_CONFIG_H__

#define LM_D_MODEL	32
#define LM_N_HEADS	4
#define LM_D_HEAD	(LM_D_MODEL / LM_N_HEADS)
#define LM_D_FF		64
#define LM_MAX_CTX	32
#define LM_N_BLOCKS	2
#define LM_BYTE_VOCAB	256

#endif
