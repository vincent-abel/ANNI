/*
**	Language-model shared sizes (toy decoder-only, scaled toward Target).
*/
#ifndef LM_CONFIG_H__
#define LM_CONFIG_H__

#define LM_D_MODEL	256
#define LM_N_HEADS	8
#define LM_D_HEAD	(LM_D_MODEL / LM_N_HEADS)
#define LM_D_FF		1024
#define LM_MAX_CTX	128
#define LM_N_BLOCKS	4
#define LM_BYTE_VOCAB	256

#endif
