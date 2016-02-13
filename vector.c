#include "vector.h"
#include "main.h"

/* INIT THE VECTOR ! */
void vector_init(vector *v)
{
	v->item = NULL;
	v->size = 0;
	v->count = 0;
}

/* COUNT ACCESSOR */
int vector_count(vector *v)
{
	return v->count;
}

/* ADD THINGS TO THE PRETTY VECTOR ! */
void vector_add(vector *v, void *e)
{
	if (v->size == 0) {
		v->size = 10;
		v->item = malloc(sizeof(void*) * v->size);
		memset(v->item, '\0', sizeof(void) * v->size);
	}

	/* condition to increase v->item:
	 last slot exhausted */
	if (v->size == v->count) {
		v->size *= 2;
		v->item = realloc(v->item, sizeof(void*) * v->size);
	}

	v->item[v->count] = e;
	v->count++;
}

void vector_set(vector *v, int index, void *e)
{
	if (index >= v->count) {
		return;
	}

	v->item[index] = e;
}

void *vector_get(vector *v, int index)
{
	if (index >= v->count) {
		return;
	}

	return v->item[index];
}

void vector_delete(vector *v, int index)
{
	int i;
	int j;
	if (index >= v->count) {
		return;
	}

	v->item[index] = NULL;

	void **newarr = (void**)malloc(sizeof(void*) * v->count * 2);
	for (i = 0, j = 0; i < v->count; i++) {
		if (v->item[i] != NULL) {
			newarr[j] = v->item[i];
			j++;
		}		
	}

	free(v->item);

	v->item = newarr;
	v->count--;
}

void vector_free(vector *v)
{
	free(v->item);
}
