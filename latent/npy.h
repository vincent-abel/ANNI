/*
**	Minimal .npy reader and writer, float32 and float16, C order
*/
#ifndef LATENT_NPY_H__
#define LATENT_NPY_H__

#include <stdint.h>
#include <stdio.h>

#define NPY_MAX_DIMS	3

typedef struct	npy {
	FILE	*fp;
	int	half;			/* 1 = float16, 0 = float32 */
	int	ndim;
	long	shape[NPY_MAX_DIMS];
	long	data_off;		/* byte offset of the first value */
} npy;

/* Writes the header; the caller then appends the values. shape: ndim longs. */
FILE		*npy_create(const char *path, int half, int ndim, const long *shape);
int		npy_open(npy *n, const char *path);
void		npy_close(npy *n);
/* count values from value index `first`, converted to float32. */
int		npy_read_f32(npy *n, long first, long count, float *out);
uint16_t	f32_to_f16(float v);
float		f16_to_f32(uint16_t h);

#endif
