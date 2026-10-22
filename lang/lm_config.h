/*
**	Language-model shared sizes (40TF target: D=1024 / 12 blocks).
**	Breaks older D=512 LMCF checkpoints — retrain. Ctx stays 256.
*/
#ifndef LM_CONFIG_H__
#define LM_CONFIG_H__

#ifndef LM_D_MODEL
#define LM_D_MODEL	1024
#endif
#ifndef LM_N_HEADS
#define LM_N_HEADS	16
#endif
#define LM_D_HEAD	(LM_D_MODEL / LM_N_HEADS)
#ifndef LM_D_FF
#define LM_D_FF	4096
#endif
#ifndef LM_MAX_CTX
#define LM_MAX_CTX	256
#endif
#ifndef LM_N_BLOCKS
#define LM_N_BLOCKS	12
#endif
#define LM_BYTE_VOCAB	256

#endif
