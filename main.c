#include "main.h"



int main(int argc, char *argv[])
{
setbuf(stdout, NULL);
	
	if(argc<=1){
		printf("Usage : ./ann numberofinput numberoflayer ");
	}
	else
	{
		input *in = 0;	
		//in.i=1234.2313;
		printf("%s\n",argv[1]);
		//printf("%lf\n",in.i);	
		int tmp=atoi(argv[1]);	
		input_init(&in,tmp);
		input_show(in);
	}

	return (1);
}
