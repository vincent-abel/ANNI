#include "main.h"

void Push(inputs **p, double input, double weigh)
{
        inputs *new_input = malloc(sizeof(inputs));
        if (!new_input) exit(EXIT_FAILURE);     /* Sinc ase of allocation failure */
        new_input->i = input;
	new_input->w = weigh;
        new_input->prev = *p;
        *p = new_input;       /* the pointer is on the last element of the pile. */
}

int Delete(inputs **p)
{
        inputs *tmp;
        if (!*p) return -1;     /* Return -1 in empty pile */
        tmp = (*p)->prev;
        free(*p);
        *p = tmp; 
	return 0;      /* the pointer is on the last element of the pile. */
}

int Size(inputs *p)
{
        int n = 0;
        while (p) {
              n++;
              p = p->prev;
          }
        return n;
}

void Clear(inputs **p)
{
        inputs *tmp;
        while (*p) {
             tmp = (*p)->prev;
             free(*p);
             *p = tmp;
          }
}

void Show(inputs *p)
{
        while(p)
          {
             printf("input : %lf | weigh : %lf |\n",p->i,p->w);
             p = p->prev;
          }
}
