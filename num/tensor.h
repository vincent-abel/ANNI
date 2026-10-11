/*
**	Tensor Header
*/
#ifndef TENSOR_H__
#define TENSOR_H__

#include "vec.h"
#include "mat.h"

typedef struct	tensor {
	ann_real	*data;
	int	*shape;
	int	ndim;
	int	size;
} tensor;

tensor	*tensor_create(const int *shape, int ndim);
void	tensor_free(tensor **t);
void	tensor_fill(tensor *t, ann_real value);
ann_real	*tensor_at(tensor *t, const int *idx);
int	tensor_as_vec(vec *view, tensor *t);
int	tensor_as_mat(mat *view, tensor *t);

#endif
