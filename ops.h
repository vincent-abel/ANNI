/*
**	Ops Header
*/
#ifndef OPS_H__
#define OPS_H__

#include "vec.h"

int	vec_softmax(vec *out, const vec *in);
double	vec_cross_entropy(const vec *probs, int class_index);
double	vec_softmax_cross_entropy(const vec *logits, int class_index);

#endif
