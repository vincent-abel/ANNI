CC=gcc -g
CFLAGS=-W -Wall -ansi -pedantic -std=c99
LDFLAGS= -lm
EXEC=ann
SRC= $(wildcard *.c)
OBJ= $(SRC:.c=.o)

all: $(EXEC)

ann: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

main.o: main.h

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

.PHONY: clean mrproper

clean:
	rm -rf *.o

mrproper: clean
	rm -rf $(EXEC)
	rm -rf *~ 
