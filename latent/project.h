/*
**	Projections of the latent space viewer: PCA, kNN graph, UMAP layout
*/
#ifndef LATENT_PROJECT_H__
#define LATENT_PROJECT_H__

#define KNN_K	15

/* x: n rows of d floats, changed in place to mean 0. out: n rows of k scores. */
int	pca_fit(float *x, long n, int d, int k, float *out);
/* The KNN_K nearest of each row of x (n x d), by cosine. */
int	knn_graph(const float *x, long n, int d, int *idx, float *dist);
/*
** 3D layout of a kNN graph, into pos (n x 3) which holds the starting layout.
** Fewer epochs suit a start that is already close, as the previous layer.
*/
int	umap_layout(const int *idx, const float *dist, long n, float *pos,
		int epochs);
/* Scale pos around its centre so the largest spread is `radius`. */
void	layout_normalize(float *pos, long n, float radius);

#endif
