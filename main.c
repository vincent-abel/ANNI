#include "main.h"



int main(int argc, char *argv[])
{
	setbuf(stdout, NULL);
	srand(time(NULL));
	
	if (argc<=1) {
		printf("Usage : ./ann numberofinput numberoflayer ");
	}
	else {
		input *in = 0;	
		neuron *ne = 0;
		int tmp=atoi(argv[1]);	

		printf("%s\n",argv[1]);

		input_init(&in,tmp);
		neuron_push(&ne);
		ne->input = in;

		neuron_Tinput(ne);
		neuron_output(ne);
		neuron_show(ne);
	}

	return (1);
}
