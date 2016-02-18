/*
**	Neuron Code
*/
#include "main.h"
#define RESPONSE 1

double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void input_init(input **input,int size)
{
	while (size--)
	{
		input_push(input,randfrom(0,1),randfrom(-2,2));
	}
}

void input_push(input **p, double in, double weigh)
{
        input *new_input = malloc(sizeof(input));
        if (!new_input) exit(EXIT_FAILURE);     /* in case of allocation failure */
        new_input->i = in;
		new_input->w = weigh;
        new_input->prev = *p;
        *p = new_input;       /* the pointer is on the last element of the pile. */
}

void nxtinput_push(nxtinput **p, double *input, double weigh)
{
        nxtinput *new_input = malloc(sizeof(nxtinput));
        if (!new_input) exit(EXIT_FAILURE);     /* in case of allocation failure */
        new_input->i = input;
		new_input->w = weigh;
        new_input->prev = *p;
        *p = new_input;       /* the pointer is on the last element of the pile. */
}

int input_delete(input **p)
{
        input *tmp;
        if (!*p) return -1;     /* Return -1 in empty pile */
        tmp = (*p)->prev;
        free(*p);
        *p = tmp; 
	return 0;      /* the pointer is on the last element of the pile. */
}

int input_size(input *p)
{
        int n = 0;
        while (p) {
              n++;
              p = p->prev;
          }
        return n;
}

void input_clear(input **p)
{
        input *tmp;
        while (*p) {
             tmp = (*p)->prev;
             free(*p);
             *p = tmp;
          }
}

void input_show(input *p)
{

        while(p->prev)
          {
             printf("input : %lf | weigh : %lf |\n",p->i,p->w);
             p = p->prev;
          }
}

void neuron_output(neuron **n) // Sigmoid for the time being
{
	while ((*n)->prev)
	{
		(*n)->output=( 1 / ( 1 + exp(-neuron_Tinput(*n) / RESPONSE)));
	}
}

double neuron_Tinput(neuron *n)
{
	input *i=(*n->input);
	n->Tinput=0;
	
	while (i->prev)
	{
		n->Tinput += i->i * i->w;
	}
	return n->Tinput;
}
/* Don't bother what the ouput is we are just linking the input to the prev output by memory address (optimization RULEZ)
*/
/*void nlayer_setInput(nlayer *nlayer)
{
	while (nlayer->prev)
	{
		neuron *n=nlayer->neuron;
		neuron *np=nlayer->prev;
		while (n->prev)
		{
			nxtinput *i=np->input;
			while (i->prev)
			{
				i->input= &n->output;
				i=i->prev;
			}
			n=n.prev;
		}
		nlayer=nlayer.prev;
	}
}*/
