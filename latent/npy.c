/*
**	Minimal .npy reader and writer, float32 and float16, C order
*/
#include <stdlib.h>
#include <string.h>
#include "npy.h"

#define HEADER_ALIGN	64
#define CHUNK		(1 << 16)

FILE *npy_create(const char *path, int half, int ndim, const long *shape)
{
	static const char magic[] = "\x93NUMPY\x01\x00";
	char dict[256];
	char dims[96];
	unsigned char len[2];
	FILE *fp;
	int used;
	int n;
	int i;

	used = 0;
	for (i = 0; i < ndim; i++)
		used += snprintf(dims + used, sizeof(dims) - (size_t)used,
		    "%ld, ", shape[i]);
	n = snprintf(dict, sizeof(dict),
	    "{'descr': '<f%d', 'fortran_order': False, 'shape': (%s), }",
	    half ? 2 : 4, dims);
	while ((10 + n + 1) % HEADER_ALIGN != 0)
		dict[n++] = ' ';
	dict[n++] = '\n';
	fp = fopen(path, "wb");
	if (!fp)
		return NULL;
	len[0] = (unsigned char)(n & 0xff);
	len[1] = (unsigned char)(n >> 8);
	if (fwrite(magic, 1, 8, fp) != 8 || fwrite(len, 1, 2, fp) != 2
	    || fwrite(dict, 1, (size_t)n, fp) != (size_t)n) {
		fclose(fp);
		return NULL;
	}
	return fp;
}

int npy_open(npy *n, const char *path)
{
	unsigned char head[10];
	char dict[1024];
	const char *p;
	int len;

	memset(n, 0, sizeof(*n));
	n->fp = fopen(path, "rb");
	if (!n->fp)
		return -1;
	if (fread(head, 1, 10, n->fp) != 10 || memcmp(head, "\x93NUMPY", 6) != 0)
		goto bad;
	len = head[8] | head[9] << 8;
	if (len >= (int)sizeof(dict)
	    || fread(dict, 1, (size_t)len, n->fp) != (size_t)len)
		goto bad;
	dict[len] = '\0';
	n->data_off = 10 + len;
	if (strstr(dict, "'<f2'"))
		n->half = 1;
	else if (!strstr(dict, "'<f4'"))
		goto bad;
	p = strstr(dict, "'shape': (");
	if (!p)
		goto bad;
	p += strlen("'shape': (");
	while (*p >= '0' && *p <= '9' && n->ndim < NPY_MAX_DIMS) {
		char *end;

		n->shape[n->ndim++] = strtol(p, &end, 10);
		p = end;
		while (*p == ',' || *p == ' ')
			p++;
	}
	if (n->ndim == 0)
		goto bad;
	return 0;
bad:
	npy_close(n);
	return -1;
}

void npy_close(npy *n)
{
	if (n->fp)
		fclose(n->fp);
	n->fp = NULL;
}

int npy_read_f32(npy *n, long first, long count, float *out)
{
	static uint16_t buf[CHUNK];
	long done;
	long i;

	if (fseek(n->fp, n->data_off + first * (n->half ? 2 : 4), SEEK_SET) != 0)
		return -1;
	if (!n->half)
		return fread(out, 4, (size_t)count, n->fp) == (size_t)count ? 0 : -1;
	for (done = 0; done < count; done += i) {
		long step = count - done < CHUNK ? count - done : CHUNK;

		if (fread(buf, 2, (size_t)step, n->fp) != (size_t)step)
			return -1;
		for (i = 0; i < step; i++)
			out[done + i] = f16_to_f32(buf[i]);
	}
	return 0;
}

uint16_t f32_to_f16(float v)
{
	uint32_t x;
	uint32_t sign;
	int exp;
	uint32_t man;

	memcpy(&x, &v, 4);
	sign = x >> 16 & 0x8000u;
	exp = (int)(x >> 23 & 0xff) - 127 + 15;
	man = x & 0x7fffffu;
	if (exp >= 31)			/* too large, inf, nan: largest finite */
		return (uint16_t)(sign | 0x7bffu);
	if (exp <= 0) {			/* subnormal or zero */
		if (exp < -10)
			return (uint16_t)sign;
		man |= 0x800000u;
		return (uint16_t)(sign | (man >> (14 - exp)));
	}
	return (uint16_t)(sign | (uint32_t)exp << 10 | man >> 13);
}

float f16_to_f32(uint16_t h)
{
	uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
	uint32_t exp = h >> 10 & 0x1f;
	uint32_t man = h & 0x3ffu;
	uint32_t x;
	float v;

	if (exp == 0) {
		v = (float)man / (1 << 24);
		return sign ? -v : v;
	}
	x = sign | (exp + 127 - 15) << 23 | man << 13;
	memcpy(&v, &x, 4);
	return v;
}
