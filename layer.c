/* 
**  Layer Code 
*/
#include "main.h"

void nlayer_push(nlayer **p)
{
        nlayer *new_input = malloc(sizeof(nlayer));
        if (!new_input) exit(EXIT_FAILURE);     /* in case of allocation failure */
        new_input->neuron = 0;
        new_input->prev = *p;
        *p = new_input;       /* the pointer is on the last element of the pile. */
}

/* Don't bother what the ouput is we are just linking the input to the prev output by memory address (optimization RULEZ)
*/
void nlayer_setInput(nlayer *nlayer)
{
	while (nlayer)
	{
		neuron *n=nlayer->neuron;
		neuron_p *np=(neuron_p *)nlayer->neuron;
		np=np->prev;
		while (n)
		{
			nxtinput *i=np->input;
			while (i->prev)
			{
				i->i= &n->output;
				i=i->prev;
			}
			n=n->prev;
		}
		nlayer=nlayer->prev;
	}
}

void nlayer_show(nlayer *nlayer)
{
	int i=0;
		
	while(nlayer)
	{
		i++;
		//input_show(p->input);
		printf("\n///////////////Present Layer %i :\n",i);
		fflush(stdout);
        neuron_show(nlayer->neuron);
        printf("\nEnd of Layer %i /////////////////\n",i);
		nlayer = nlayer->prev;
	}
}