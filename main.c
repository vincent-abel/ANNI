#include "main.h"



int main(int argc, char *argv[])
{
	setbuf(stdout, NULL);
	srand(time(NULL));
	
	if (argc<=2) {
		printf("Usage : ./ann numberofinput numberofneuron numberoflayer \n");
	}
	else {
		input *in = 0;	
		neuron *ne = 0;
		nlayer *nlayer = 0;
		int tmp=atoi(argv[1]);	
		int ineuron,neuron=atoi(argv[2]);
		int layer=atoi(argv[3]);

		printf("%i\n",layer);
		int init=input_init(&in,tmp);
		while(layer--){
						printf("Layer : %i\n",layer);
			ne=0;

			ineuron=neuron;

			while (ineuron--){

				neuron_push(&ne);
				if (init)
				{
					ne->input = in;
					init=0;
				}
			}

			nlayer_setInput(nlayer);
			neuron_Tinput(ne);
			neuron_output(ne);
			nlayer_push(&nlayer);
			nlayer->neuron=ne;
			neuron_show(nlayer->neuron);

		}
// test
//		nlayer_show(nlayer);
//		neuron_show(ne);
	}

	return (1);
}
