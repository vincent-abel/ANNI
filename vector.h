#ifndef VECTOR_H__
#define VECTOR_H__

/*
*	A Nice vector containing it's size, count, and list of item. pretty neat.
*/
typedef struct vector_ {
    void** item;
    int size;
    int count;
} vector;

void vector_init(vector*);
int vector_count(vector*);
void vector_add(vector*, void*);
void vector_set(vector*, int, void*);
void *vector_get(vector*, int);
void vector_delete(vector*, int);
void vector_free(vector*);

#endif

