/*
**	Ops Code
*/
#include "main.h"

int vec_softmax(vec *out, const vec *in)
{
	int i;
	double max_v;
	double sum;

	if (!out || !in || !out->data || !in->data || out->n != in->n || in->n <= 0)
		return -1;

	max_v = in->data[0];
	for (i = 1; i < in->n; i++) {
		if (in->data[i] > max_v)
			max_v = in->data[i];
	}

	sum = 0.0;
	for (i = 0; i < in->n; i++) {
		out->data[i] = exp(in->data[i] - max_v);
		sum += out->data[i];
	}
	if (sum == 0.0)
		return -1;
	for (i = 0; i < in->n; i++)
		out->data[i] /= sum;
	return 0;
}
