#ifndef NEURON_H__
#define NEURON_H__

typedef struct inputs_ {
	double i;
	double w;
	struct inputs *prev;

} inputs;

typedef struct neuron_ {
    void** item;
    int size;
    int count;
} neuron;


void Push(inputs **, double , double);
int Delete(inputs **);
int Size(inputs *);
void Clear(inputs **);
void Show(inputs *);

#endif
