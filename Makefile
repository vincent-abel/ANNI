CC=gcc -g
CFLAGS=-W -Wall -ansi -pedantic -std=c99
LDFLAGS= -lm
EXEC=ann
LIB_SRC=neuron.c layer.c dataset.c vec.c
LIB_OBJ=$(LIB_SRC:.c=.o)
OBJ=$(LIB_OBJ) main.o
TEST=tests/test_ann

all: $(EXEC)

ann: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

main.o: main.h layer.h neuron.h dataset.h vec.h

$(LIB_OBJ): main.h layer.h neuron.h dataset.h vec.h

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

$(TEST): tests/test_ann.c $(LIB_OBJ)
	$(CC) $(CFLAGS) -o $@ tests/test_ann.c $(LIB_OBJ) $(LDFLAGS)

test: $(TEST)
	./$(TEST)

.PHONY: clean mrproper test

clean:
	rm -rf *.o

mrproper: clean
	rm -rf $(EXEC)
	rm -rf $(TEST)
	rm -rf *~
