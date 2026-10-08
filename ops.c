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

double vec_cross_entropy(const vec *probs, int class_index)
{
	double p;

	if (!probs || !probs->data || class_index < 0 || class_index >= probs->n)
		return -1.0;
	p = probs->data[class_index];
	if (p < 1e-15)
		p = 1e-15;
	return -log(p);
}

double vec_softmax_cross_entropy(const vec *logits, int class_index)
{
	int i;
	double max_v;
	double sum;
	double log_prob;

	if (!logits || !logits->data || class_index < 0 || class_index >= logits->n)
		return -1.0;

	max_v = logits->data[0];
	for (i = 1; i < logits->n; i++) {
		if (logits->data[i] > max_v)
			max_v = logits->data[i];
	}

	sum = 0.0;
	for (i = 0; i < logits->n; i++)
		sum += exp(logits->data[i] - max_v);
	log_prob = (logits->data[class_index] - max_v) - log(sum);
	return -log_prob;
}
