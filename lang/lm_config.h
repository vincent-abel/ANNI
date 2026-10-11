/*
**	Language-model shared sizes (Target-scale decoder-only).
**	R5/R5b/R6/R7: D=512, heads=16, FF=2048, ctx=256, blocks=6.
*/
#ifndef LM_CONFIG_H__
#define LM_CONFIG_H__

#define LM_D_MODEL	512
#define LM_N_HEADS	16
#define LM_D_HEAD	(LM_D_MODEL / LM_N_HEADS)
#define LM_D_FF		2048
#define LM_MAX_CTX	256
#define LM_N_BLOCKS	6
#define LM_BYTE_VOCAB	256

#endif
