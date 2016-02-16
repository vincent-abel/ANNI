#include "main.h"

int main(int argc, char *argv[])
{
	
	if(argc>0){
		for(int i=0;argc--;i++){
			printf("%s",argv[i]);
		}
	}
	printf("Test\n");
	return (1);
}
