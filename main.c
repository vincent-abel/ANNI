#include "main.h"



int main(int argc, char *argv[])
{
	
	if(argc<=1){
		printf("Usage : ./ann numberofinput numberoflayer ");
	}
	else
	{
		input in;	
		//in.i=1234.2313;
		printf("%s\n",argv[1]);
		//printf("%lf\n",in.i);		
		input_init(&in,(int)argv[1]);
	}

	return (1);
}
