/*
**	CPU stub for optional GPU backend (default build)
*/
#include "gpu.h"

int gpu_available(void)
{
	return 0;
}

const char *gpu_backend_name(void)
{
	return "cpu";
}

int gpu_mat_mul(mat *c, const mat *a, const mat *b)
{
	(void)c;
	(void)a;
	(void)b;
	return -1;
}
